import test from 'node:test';
import assert from 'node:assert/strict';
import { createRectMesh } from '../src/mesh.js';
import { CorotationalFEM } from '../src/sim/corotationalFEM.js';
import { ProjectiveDynamics } from '../src/sim/projectiveDynamics.js';
import { StretchDriver } from '../src/stretch.js';

const close = (a, b, tol, msg) => assert.ok(Math.abs(a - b) <= tol, `${msg ?? ''} ${a} vs ${b} (tol ${tol})`);

function setup(S, nu, { E = 1e5, nx = 12, ny = 12, damping = 5, iters = 40 } = {}) {
  const mesh = createRectMesh(1, 1, nx, ny);
  const sim = new S(mesh, { E, nu, density: 1000, gravity: 0, damping });
  if (S === ProjectiveDynamics) sim.iterations = iters;
  return { mesh, sim };
}

test('internalForces equals the FEM elastic force (same exact gradient)', () => {
  const mesh = createRectMesh(2, 1, 6, 3);
  const fem = new CorotationalFEM(mesh, { E: 1e5, nu: 0.3 });
  const x = Float64Array.from(mesh.X);
  for (let i = 0; i < x.length; i++) x[i] += 0.05 * Math.sin(3 * i + 1);
  const a = new Float64Array(x.length);
  const b = new Float64Array(x.length);
  fem.elasticForce(x, a);
  fem.internalForces(x, b);
  for (let i = 0; i < a.length; i++) close(a[i], b[i], 1e-8 * Math.max(1, Math.abs(a[i])), `dof ${i}`);
});

for (const [name, S] of [
  ['FEM', CorotationalFEM],
  ['PD', ProjectiveDynamics],
]) {
  test(`${name}: grips follow the prescribed motion exactly`, () => {
    const { mesh, sim } = setup(S, 0.3);
    const drv = new StretchDriver(mesh, { speed: 0.5, maxStretch: 3 });
    drv.attach(sim);
    for (let i = 0; i < 60; i++) {
      drv.advance(1 / 120, sim);
      sim.step(1 / 120);
    }
    for (const i of mesh.rightIdx) close(sim.x[2 * i], mesh.X[2 * i] + drv.d, 1e-12);
    for (const i of mesh.leftIdx) close(sim.x[2 * i], mesh.X[2 * i] - drv.d, 1e-12);
    close(drv.stretch, 1 + 2 * 0.25, 1e-9); // 0.5 s * 0.5 m/s
  });

  test(`${name}: nu = 0 stretch gives the exact linear force E*H0*(s-1) up to s = 5`, () => {
    const E = 1e5;
    const { mesh, sim } = setup(S, 0, { E });
    const drv = new StretchDriver(mesh, { speed: 1, maxStretch: 5 });
    drv.attach(sim);
    const h = 1 / 120;
    for (let i = 0; i < 120 * 12; i++) {
      drv.advance(h, sim);
      sim.step(h);
    }
    assert.equal(drv.mode, 'hold');
    close(drv.stretch, 5, 1e-9);
    const F = sim.gripForce(mesh.rightIdx, +1);
    const expected = E * 1 * (5 - 1); // per unit depth, H0 = 1
    console.log(`  ${name}: grip force ${F.toFixed(1)} vs analytic ${expected}`);
    close(F / expected, 1, 0.01, 'force ratio');
    assert.equal(sim.deformationStats().inverted, 0);
  });

  test(`${name}: nu = 0.3 stretch to 2.5x and back recovers the original shape`, () => {
    const { mesh, sim } = setup(S, 0.3);
    const drv = new StretchDriver(mesh, { speed: 1, maxStretch: 2.5 });
    drv.attach(sim);
    const h = 1 / 120;
    for (let i = 0; i < 120 * 6 && drv.mode !== 'hold'; i++) {
      drv.advance(h, sim);
      sim.step(h);
    }
    assert.equal(sim.deformationStats().inverted, 0);
    drv.mode = 'unload';
    for (let i = 0; i < 120 * 8; i++) {
      drv.advance(h, sim);
      sim.step(h);
    }
    assert.equal(drv.d, 0);
    let err = 0;
    for (let i = 0; i < sim.x.length; i++) err = Math.max(err, Math.abs(sim.x[i] - mesh.X[i]));
    assert.ok(err < 2e-3, `max deviation from rest ${err}`);
  });
}
