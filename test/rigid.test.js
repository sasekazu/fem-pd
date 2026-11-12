import test from 'node:test';
import assert from 'node:assert/strict';
import { createBlockMesh, createDiscMesh } from '../src/mesh.js';
import { CorotationalFEM } from '../src/sim/corotationalFEM.js';
import { ProjectiveDynamics } from '../src/sim/projectiveDynamics.js';
import { ContactModel } from '../src/contact.js';
import { HalfPlane, Polygon, boxVertices, regularVertices, starVertices, lShapeVertices } from '../src/rigid.js';

const close = (a, b, tol, msg) => assert.ok(Math.abs(a - b) <= tol, `${msg ?? ''} ${a} vs ${b} (tol ${tol})`);

const rng = (seed) => {
  let t = seed;
  return () => ((t = (Math.imul(t, 1664525) + 1013904223) >>> 0) / 2 ** 32) - 0.5;
};

const q = [0, 0];

test('HalfPlane: signed distance and normal', () => {
  const f = new HalfPlane(0, 1, 0.3);
  close(f.query(5, 1.3, q), 1, 1e-15);
  assert.deepEqual(q, [0, 1]);
  close(f.query(0, -0.2, q), -0.5, 1e-15);
  assert.equal(f.bounds(), null);
  const w = new HalfPlane(3, 4, 5); // normalised: n = (0.6, 0.8), d = 1
  close(w.query(2, 0, q), 0.2, 1e-15);
  close(q[0], 0.6, 1e-15);
  close(q[1], 0.8, 1e-15);
});

test('Polygon (box): signed distance, normal, pose', () => {
  const b = new Polygon(boxVertices(2, 1), { x: 1, y: 2 }); // x in [0, 2], y in [1.5, 2.5]
  close(b.query(1, 2.3, q), -0.2, 1e-14, 'inside');
  close(q[0], 0, 1e-14);
  close(q[1], 1, 1e-14); // nearest face is the top one
  close(b.query(3.5, 2, q), 1.5, 1e-14, 'outside, side face');
  close(q[0], 1, 1e-14);
  close(b.query(3, 3, q), Math.hypot(1, 0.5), 1e-14, 'outside, corner');
  close(q[0], 1 / Math.hypot(1, 0.5), 1e-14);
  close(q[1], 0.5 / Math.hypot(1, 0.5), 1e-14);
  assert.deepEqual(b.bounds(), [0, 1.5, 2, 2.5]);
  assert.ok(b.contains(1, 2) && !b.contains(2.01, 2));
  // rotate by 90 degrees about (1, 2): extents become x in [0.5, 1.5], y in [1, 3]
  b.setPose(1, 2, Math.PI / 2);
  close(b.query(1.8, 2, q), 0.3, 1e-14);
  close(q[0], 1, 1e-14);
  close(q[1], 0, 1e-14);
  const bb = b.bounds();
  close(bb[0], 0.5, 1e-14);
  close(bb[3], 3, 1e-14);
});

test('Polygon: clockwise input is reordered, area is right', () => {
  const ccw = boxVertices(2, 1);
  const cw = [];
  for (let i = 3; i >= 0; i--) cw.push(ccw[2 * i], ccw[2 * i + 1]); // same corners, opposite winding
  const p = new Polygon(cw);
  close(p.area, 2, 1e-15);
  close(p.query(0, 0.2, q), -0.3, 1e-14);
  close(p.query(0, 0.9, q), 0.4, 1e-14);
  assert.throws(() => new Polygon([0, 0, 1, 1]));
});

test('Polygon (concave L-shape): notch is outside, distances to both notch faces', () => {
  // 1 x 1 box centred at the origin with the upper right 0.5 x 0.5 corner removed
  const L = new Polygon(lShapeVertices(1, 1, 0.5, 0.5));
  assert.ok(L.contains(-0.25, 0.25), 'upper left arm');
  assert.ok(L.contains(0.25, -0.25), 'lower right arm');
  assert.ok(!L.contains(0.25, 0.25), 'notch');
  // point in the notch, 0.1 above the notch floor (y = 0) and 0.4 from the notch wall (x = 0)
  close(L.query(0.4, 0.1, q), 0.1, 1e-14);
  close(q[1], 1, 1e-14); // pushes up out of the notch floor
  // a point just inside the lower arm next to the notch floor
  close(L.query(0.4, -0.05, q), -0.05, 1e-14);
  close(q[1], 1, 1e-14);
});

test('Polygon: unit normal, surface point at p - phi n, gradient matches finite differences', () => {
  const shapes = [
    new Polygon(starVertices(5, 0.55, 0.24, 0.3), { x: 0.2, y: -0.1, angle: 0.7 }),
    new Polygon(lShapeVertices(0.9, 0.8, 0.45, 0.4), { x: -0.3, y: 0.2, angle: -1.1 }),
    new Polygon(regularVertices(3, 0.5), { angle: 2 }),
  ];
  const r = rng(11);
  const e = 1e-7;
  let n = 0;
  for (const poly of shapes) {
    for (let k = 0; k < 400; k++) {
      const px = 1.6 * r();
      const py = 1.6 * r();
      const phi = poly.query(px, py, q);
      close(Math.hypot(q[0], q[1]), 1, 1e-12, 'unit normal');
      // moving along -phi * n lands on the surface
      const s = poly.query(px - phi * q[0], py - phi * q[1], [0, 0]);
      close(s, 0, 1e-9, 'surface point');
      // finite-difference gradient (skip the measure-zero kinks of the distance field)
      const nx = q[0];
      const ny = q[1];
      const gx = (poly.query(px + e, py, q) - poly.query(px - e, py, q)) / (2 * e);
      const gy = (poly.query(px, py + e, q) - poly.query(px, py - e, q)) / (2 * e);
      if (Math.hypot(gx - nx, gy - ny) > 1e-4) {
        // a kink: both one-sided candidates must be valid unit gradients
        assert.ok(Math.hypot(gx, gy) < 1.01, 'kink gradient');
        continue;
      }
      n++;
    }
  }
  assert.ok(n > 1100, `only ${n} smooth samples matched`);
});

test('ContactModel: energy = 1/2 k sum m phi^2 and net force is -dE/d(translation)', () => {
  const mesh = createDiscMesh(0.5, 4, [0, 1.0]);
  const s = new CorotationalFEM(mesh, { E: 1e6, nu: 0.3 });
  const star = new Polygon(starVertices(5, 0.7, 0.3, 0.4), { x: 0.1, y: 0.9, angle: 0.3 });
  s.setContact({ enabled: true, floorY: 0.6, k: 3e5 });
  s.setColliders([star]);
  const x = Float64Array.from(s.x);
  const ct = s.contact;
  let expect = 0;
  const phi = (c, i) => c.query(x[2 * i], x[2 * i + 1], q);
  for (const c of ct.colliders()) for (let i = 0; i < s.n; i++) {
    const d = phi(c, i);
    if (d < 0) expect += 0.5 * ct.k * s.mass[i] * d * d;
  }
  assert.ok(expect > 0);
  close(s.contactEnergy(x), expect, 1e-10 * expect);
  // the net force on the soft body equals -dE/d(uniform translation)
  const forces = ct.colliderForces(x, s.mass, s.n);
  assert.equal(forces.length, 2);
  assert.ok(forces[0].count > 0 && forces[1].count > 0);
  const tr = (dx, dy) => {
    const xt = Float64Array.from(x, (v, i) => v + (i % 2 ? dy : dx));
    return s.contactEnergy(xt);
  };
  const eps = 1e-7;
  const fx = forces[0].fx + forces[1].fx;
  const fy = forces[0].fy + forces[1].fy;
  close(-(tr(eps, 0) - tr(-eps, 0)) / (2 * eps), fx, 1e-5 * Math.abs(fx) + 1e-6, 'fx');
  close(-(tr(0, eps) - tr(0, -eps)) / (2 * eps), fy, 1e-5 * Math.abs(fy) + 1e-6, 'fy');
  // the floor alone pushes up, the star pushes the ball away from itself
  assert.ok(forces[0].fy > 0);
  s.setContact({ enabled: false });
  close(ct.colliders().length, 1, 0);
  assert.ok(s.contactEnergy(x) < expect);
});

test('PD gradient matches finite differences of the objective (rotated star + floor)', () => {
  const mesh = createDiscMesh(0.5, 3, [0, 0.9]);
  const s = new ProjectiveDynamics(mesh, { E: 1e6, nu: 0.3, density: 1000, damping: 0 });
  s.setContact({ enabled: true, floorY: 0.5, k: 1e6 });
  s.setColliders([new Polygon(starVertices(5, 0.8, 0.35, 0.2), { x: 0.05, y: 0.85, angle: 0.4 })]);
  s.step(1 / 240); // builds the factorisation
  const h2 = (1 / 240) ** 2;
  const r = rng(5);
  const x = Float64Array.from(s.mesh.X);
  for (let i = 0; i < x.length; i++) x[i] += 0.02 * r();
  const y = Float64Array.from(x, (v) => v + 0.01 * r());
  assert.ok(s.contact.collect(x, s.n) > 3, 'need active contacts');
  const nf = s._nf;
  const g = new Float64Array(2 * nf);
  s._gradient(x, y, h2, g);
  let checked = 0;
  for (let v = 0; v < s.n; v++) {
    for (const comp of [0, 1]) {
      const eps = 1e-7;
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
  assert.equal(checked, 2 * s.n);
});

// ---- a soft block pressed by a kinematic rigid body: both solvers must agree ----

const MAT = { E: 1e6, nu: 0.3, density: 1000, damping: 0.5 };
const H = 1 / 240;

/** Block 1 x 0.6 on the floor y = 0, rigid body lowered from `y0` to `y1` (origin height) at 0.1 m/s, then held. */
function press(S, body, y0, y1, { hold = 1.5, k = 1e7, iterations = 30 } = {}) {
  const mesh = createBlockMesh(1, 0.6, 14, 8, [0, 0]);
  const s = new S(mesh, MAT);
  if (S === ProjectiveDynamics) s.iterations = iterations;
  s.setContact({ enabled: true, floorY: 0, k });
  s.setColliders([body]);
  const steps = Math.round(Math.abs(y1 - y0) / (0.1 * H));
  let minDet = Infinity;
  for (let i = 0; i < steps + Math.round(hold / H); i++) {
    const y = i < steps ? y0 + ((y1 - y0) * i) / steps : y1;
    body.setPose(body.x, y, body.angle);
    s.step(H);
    minDet = Math.min(minDet, s.deformationStats().minDet);
  }
  return { s, minDet };
}

for (const [name, S] of [
  ['FEM', CorotationalFEM],
  ['PD', ProjectiveDynamics],
]) {
  test(`${name}: flat rigid box on a block -> force balance (weight + press = floor reaction), bounded penetration`, () => {
    const box = new Polygon(boxVertices(1.4, 0.4), { x: 0, y: 0.8 + 0.2 });
    const { s } = press(S, box, 0.8 + 0.2, 0.55 + 0.2); // lower the bottom face from 0.8 to 0.55 (block top at 0.6)
    const f = s.contact.colliderForces(s.x, s.mass, s.n);
    const weight = s.mass.reduce((a, b) => a + b, 0) * s.params.gravity;
    // static: floor reaction - box press - weight = 0
    close(f[0].fy - -f[1].fy - weight, 0, 0.02 * weight + 0.01 * Math.abs(f[1].fy), 'force balance');
    // 5 cm compression of a 0.6 m block: sigma ~ E * 0.05/0.6, per metre of width ~ 1 m
    const expect = (MAT.E * 0.05) / 0.6;
    assert.ok(-f[1].fy > 0.8 * expect && -f[1].fy < 1.3 * expect, `press force ${-f[1].fy} vs ~${expect}`);
    assert.ok(Math.abs(f[1].fx) < 0.02 * -f[1].fy, `lateral force ${f[1].fx}`);
    assert.ok(s.contact.maxPenetration(s.x, s.n, 1) < 3e-3, `penetration ${s.contact.maxPenetration(s.x, s.n, 1)}`);
    assert.ok(s.contact.maxPenetration(s.x, s.n, 0) < 1e-3);
    for (const v of s.x) assert.ok(Number.isFinite(v));
  });

  test(`${name}: wedge pressed into a block -> no element inverts, penetration stays small`, () => {
    // triangle with the tip pointing down; the tip is 0.5 below the origin
    const wedge = new Polygon(regularVertices(3, 0.5, -Math.PI / 2), { x: 0, y: 0.6 + 0.2 + 0.5 });
    const { s, minDet } = press(S, wedge, 0.6 + 0.2 + 0.5, 0.6 - 0.2 + 0.5); // tip 0.2 below the block top
    assert.ok(minDet > 0.1, `min det F ${minDet}`);
    const pen = s.contact.maxPenetration(s.x, s.n, 1);
    assert.ok(pen < 0.01, `penetration ${pen}`);
    const f = s.contact.colliderForces(s.x, s.mass, s.n)[1];
    assert.ok(f.count >= 2 && -f.fy > 0, `contacts ${f.count}, press ${-f.fy}`);
    close(f.fx / f.fy, 0, 0.05, 'symmetric wedge: no net lateral force');
  });
}

test('FEM and PD agree on the wedge press (force on the rigid body and block shape)', () => {
  // The tip is 7 cm off-centre: a centred tip on a frictionless floor is an unstable symmetric equilibrium
  // (the block squeezes out to the left or the right), so FEM and PD could legitimately pick opposite sides.
  const mk = () => new Polygon(regularVertices(3, 0.5, -Math.PI / 2), { x: 0.07, y: 0.6 + 0.2 + 0.5 });
  const a = press(CorotationalFEM, mk(), 1.3, 0.9).s;
  const b = press(ProjectiveDynamics, mk(), 1.3, 0.9).s;
  const fa = -a.contact.colliderForces(a.x, a.mass, a.n)[1].fy;
  const fb = -b.contact.colliderForces(b.x, b.mass, b.n)[1].fy;
  console.log(`  press force  FEM=${fa.toFixed(0)}  PD=${fb.toFixed(0)}`);
  close(fa, fb, 0.03 * fa, 'press force');
  let dmax = 0;
  for (let i = 0; i < a.x.length; i++) dmax = Math.max(dmax, Math.abs(a.x[i] - b.x[i]));
  assert.ok(dmax < 0.02, `max vertex difference ${dmax}`);
});

test('PD without L-BFGS (plain local/global + line search) gives the same wedge press', () => {
  const mk = () => new Polygon(regularVertices(3, 0.5, -Math.PI / 2), { x: 0.07, y: 1.3 });
  const run = (lbfgsM) => {
    const body = mk();
    const mesh = createBlockMesh(1, 0.6, 14, 8, [0, 0]);
    const s = new ProjectiveDynamics(mesh, MAT);
    s.iterations = 60;
    s.lbfgsM = lbfgsM;
    s.setContact({ enabled: true, floorY: 0, k: 1e7 });
    s.setColliders([body]);
    const steps = Math.round(0.4 / (0.1 * H));
    for (let i = 0; i < steps + 240; i++) {
      body.setPose(0.07, i < steps ? 1.3 - (0.4 * i) / steps : 0.9, 0);
      s.step(H);
    }
    return -s.contact.colliderForces(s.x, s.mass, s.n)[1].fy;
  };
  const a = run(6);
  const b = run(0);
  close(a, b, 0.01 * a, 'press force');
});

test('kinematic body only matters at its end-of-step pose: a body far away has no effect', () => {
  const mesh = createBlockMesh(1, 0.6, 8, 5, [0, 0]);
  const run = (withBody) => {
    const s = new CorotationalFEM(mesh, MAT);
    s.setContact({ enabled: true, floorY: 0, k: 1e7 });
    if (withBody) s.setColliders([new Polygon(boxVertices(1, 1), { x: 5, y: 5 })]);
    for (let i = 0; i < 60; i++) s.step(H);
    return s.x;
  };
  const a = run(false);
  const b = run(true);
  for (let i = 0; i < a.length; i++) assert.equal(a[i], b[i]);
});

test('ContactModel without any collider is a no-op', () => {
  const c = new ContactModel();
  assert.equal(c.enabled, false);
  assert.equal(c.collect(new Float64Array(4), 2), 0);
  assert.equal(c.energy(new Float64Array(4), new Float64Array(2), 2), 0);
});
