import test from 'node:test';
import assert from 'node:assert/strict';
import { createDiscMesh } from '../src/mesh.js';
import { CorotationalFEM } from '../src/sim/corotationalFEM.js';
import { ProjectiveDynamics } from '../src/sim/projectiveDynamics.js';

const close = (a, b, tol, msg) => assert.ok(Math.abs(a - b) <= tol, `${msg ?? ''} ${a} vs ${b} (tol ${tol})`);

test('disc mesh: CCW triangles, area -> pi r^2, boundary ring', () => {
  const m = createDiscMesh(0.5, 8, [0, 1]);
  assert.equal(m.n, 1 + 3 * 8 * 9);
  let sum = 0;
  for (const a of m.area) {
    assert.ok(a > 0);
    sum += a;
  }
  close(sum, Math.PI * 0.25, 0.01 * Math.PI * 0.25, 'area');
  assert.equal(m.boundary.length, 48);
  assert.equal(m.fixedIdx.length, 0);
});

const params = { E: 1e6, nu: 0.3, density: 1000, damping: 0 };
const contact = { enabled: true, floorY: 0, k: 1e6 };

function ball(S, rings = 6) {
  const mesh = createDiscMesh(0.5, rings, [0, 1.5]);
  const s = new S(mesh, params);
  s.setContact(contact);
  return s;
}

test('contact energy = 1/2 k sum m d^2 over penetrating vertices', () => {
  const s = ball(CorotationalFEM, 3);
  const x = Float64Array.from(s.x);
  let expect = 0;
  for (let i = 0; i < s.n; i++) {
    x[2 * i + 1] -= 1.2; // bottom of the ball goes 0.2 below the floor
    const d = -x[2 * i + 1];
    if (d > 0) expect += 0.5 * contact.k * s.mass[i] * d * d;
  }
  assert.ok(expect > 0);
  close(s.contactEnergy(x), expect, 1e-9 * expect);
  s.setContact({ enabled: false });
  assert.equal(s.contactEnergy(x), 0);
});

test('PD gradient matches finite differences of the objective (with contact)', () => {
  const s = ball(ProjectiveDynamics, 3);
  s.step(1 / 240); // builds the factorisation
  const h2 = (1 / 240) ** 2;
  const rnd = (() => {
    let t = 5;
    return () => ((t = (Math.imul(t, 1664525) + 1013904223) >>> 0) / 2 ** 32) - 0.5;
  })();
  const x = Float64Array.from(s.mesh.X);
  for (let i = 0; i < x.length; i++) x[i] += 0.02 * rnd();
  for (let i = 0; i < s.n; i++) x[2 * i + 1] -= 1.5; // put the ball into the floor
  const y = Float64Array.from(x, (v) => v + 0.01 * rnd());
  const nf = s._nf;
  const g = new Float64Array(2 * nf);
  s._gradient(x, y, h2, g);
  let checked = 0;
  for (const v of [0, 5, 11, 17, 30]) {
    for (const comp of [0, 1]) {
      const eps = 1e-6;
      const xp = Float64Array.from(x);
      const xm = Float64Array.from(x);
      xp[2 * v + comp] += eps;
      xm[2 * v + comp] -= eps;
      const fd = (s._objective(xp, y, h2) - s._objective(xm, y, h2)) / (2 * eps);
      const an = g[comp * nf + s._fidx[v]];
      close(fd, an, 1e-4 * Math.max(1, Math.abs(an)), `vertex ${v} comp ${comp}`);
      checked++;
    }
  }
  assert.equal(checked, 10);
});

for (const [name, S] of [
  ['FEM', CorotationalFEM],
  ['PD', ProjectiveDynamics],
]) {
  test(`${name}: dropped ball bounces, never gains energy, comes to rest on the floor`, () => {
    const s = ball(S);
    if (S === ProjectiveDynamics) s.iterations = 20;
    const e0 = s.energies().total;
    let maxE = e0;
    let apex = null;
    let prevVy = 0;
    let minY = Infinity;
    const h = 1 / 240;
    for (let i = 0; i < 240 * 4; i++) {
      s.step(h);
      const c = s.centerOfMass();
      if (apex === null && prevVy > 0 && c.vy <= 0) apex = c.y;
      prevVy = c.vy;
      maxE = Math.max(maxE, s.energies().total);
      for (let k = 0; k < s.n; k++) minY = Math.min(minY, s.x[2 * k + 1]);
    }
    assert.ok(apex !== null, 'ball never rebounded');
    assert.ok(apex > 0.7 && apex < 1.3, `first apex ${apex}`);
    assert.ok(maxE <= e0 * 1.001, `energy grew: ${maxE / e0}`);
    assert.ok(minY > -0.05, `penetration ${-minY}`);
    for (const v of s.x) assert.ok(Number.isFinite(v));
    // after 4 s the ball is (nearly) resting: centre of mass close to radius above the floor
    assert.ok(s.centerOfMass().y < 0.62, `centre height ${s.centerOfMass().y}`);
  });
}

test('FEM and PD agree on the first rebound height (same contact energy)', () => {
  const run = (S, it) => {
    const s = ball(S);
    if (it) s.iterations = it;
    let prevVy = 0;
    for (let i = 0; i < 240 * 2; i++) {
      s.step(1 / 240);
      const c = s.centerOfMass();
      if (prevVy > 0 && c.vy <= 0) return c.y;
      prevVy = c.vy;
    }
    return NaN;
  };
  const a = run(CorotationalFEM);
  const b = run(ProjectiveDynamics, 30);
  console.log(`  first apex  FEM=${a.toFixed(4)}  PD(30 it)=${b.toFixed(4)}`);
  close(a, b, 0.03, 'apex');
});
