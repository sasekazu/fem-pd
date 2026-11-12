import test from 'node:test';
import assert from 'node:assert/strict';
import { polarRotation, lame } from '../src/math2d.js';
import { createRectMesh } from '../src/mesh.js';
import { CorotationalFEM } from '../src/sim/corotationalFEM.js';
import { ProjectiveDynamics } from '../src/sim/projectiveDynamics.js';

const close = (a, b, tol, msg) => assert.ok(Math.abs(a - b) <= tol, `${msg ?? ''} ${a} vs ${b} (tol ${tol})`);

test('polarRotation: F = R S with S symmetric, det R = 1', () => {
  const out = [0, 0];
  for (const F of [
    [1.2, 0.3, -0.1, 0.9],
    [0.2, -1.1, 1.0, 0.1],
    [-1, 0, 0, 1], // inverted: still a proper rotation
  ]) {
    polarRotation(...F, out);
    const [c, s] = out;
    close(c * c + s * s, 1, 1e-12, 'unit');
    // S = R^T F
    const s01 = c * F[1] + s * F[3];
    const s10 = -s * F[0] + c * F[2];
    if (F[0] * F[3] - F[1] * F[2] > 0) close(s01, s10, 1e-12, 'symmetric S');
  }
});

test('mesh: areas sum to rectangle, all triangles CCW', () => {
  const m = createRectMesh(4, 1, 40, 8);
  assert.equal(m.n, 41 * 9);
  assert.equal(m.tris.length / 3, 2 * 40 * 8);
  let sum = 0;
  for (const a of m.area) {
    assert.ok(a > 0);
    sum += a;
  }
  close(sum, 4, 1e-12, 'total area');
  assert.equal(m.fixedIdx.length, 9);
  for (const i of m.fixedIdx) close(m.X[2 * i], 0, 0);
});

test('lame', () => {
  const { mu, lambda } = lame(1000, 0);
  close(mu, 500, 1e-12);
  close(lambda, 0, 1e-12);
});

test('FEM: zero elastic force at rest', () => {
  const mesh = createRectMesh(4, 1, 12, 3);
  const fem = new CorotationalFEM(mesh);
  const f = new Float64Array(2 * mesh.n);
  fem.elasticForce(fem.x, f);
  for (const v of f) close(v, 0, 1e-8);
});

test('FEM: zero elastic force under rigid rotation + translation (corotational property)', () => {
  const mesh = createRectMesh(4, 1, 12, 3);
  const fem = new CorotationalFEM(mesh);
  const th = 0.9;
  const c = Math.cos(th);
  const s = Math.sin(th);
  const x = new Float64Array(2 * mesh.n);
  for (let i = 0; i < mesh.n; i++) {
    x[2 * i] = c * mesh.X[2 * i] - s * mesh.X[2 * i + 1] + 1.5;
    x[2 * i + 1] = s * mesh.X[2 * i] + c * mesh.X[2 * i + 1] - 0.7;
  }
  const f = new Float64Array(2 * mesh.n);
  fem.elasticForce(x, f);
  for (const v of f) close(v, 0, 1e-6);
});

test('FEM: warped stiffness matches finite-difference of force for small perturbations of rest', () => {
  const mesh = createRectMesh(2, 1, 4, 2);
  const fem = new CorotationalFEM(mesh, { E: 1000, nu: 0.3 });
  const N = 2 * mesh.n;
  const f0 = new Float64Array(N);
  fem.elasticForce(fem.x, f0, true);
  const { rowPtr, colIdx, vals } = fem.csr;
  const eps = 1e-7;
  const f1 = new Float64Array(N);
  for (const dof of [3, 8, 11]) {
    const x = Float64Array.from(fem.x);
    x[dof] += eps;
    fem.elasticForce(x, f1);
    for (let r = 0; r < N; r++) {
      let k = 0;
      for (let q = rowPtr[r]; q < rowPtr[r + 1]; q++) if (colIdx[q] === dof) k = vals[q];
      const fd = -(f1[r] - f0[r]) / eps;
      close(fd, k, 1e-4 * Math.max(1, Math.abs(k)), `K[${r},${dof}]`);
    }
  }
});

test('PD: rest state is a fixed point without gravity', () => {
  const mesh = createRectMesh(4, 1, 12, 3);
  const pd = new ProjectiveDynamics(mesh, { gravity: 0 });
  for (let i = 0; i < 5; i++) pd.step(1 / 60);
  for (let i = 0; i < pd.x.length; i++) close(pd.x[i], mesh.X[i], 1e-9);
});

test('FEM: rest state is a fixed point without gravity', () => {
  const mesh = createRectMesh(4, 1, 12, 3);
  const fem = new CorotationalFEM(mesh, { gravity: 0 });
  for (let i = 0; i < 5; i++) fem.step(1 / 60);
  for (let i = 0; i < fem.x.length; i++) close(fem.x[i], mesh.X[i], 1e-9);
});

function settle(sim, steps, h = 1 / 60) {
  for (let i = 0; i < steps; i++) sim.step(h);
  return sim.tip[1] - sim.mesh.X[2 * sim.mesh.tipIdx + 1];
}

test('static sag: FEM, PD and the Euler-Bernoulli cantilever agree (stiff beam, nu = 0)', () => {
  const E = 5e8;
  const rho = 1000;
  const g = 9.8;
  const params = { E, nu: 0, density: rho, gravity: g, damping: 20 };
  const mesh = createRectMesh(4, 1, 24, 6);
  const fem = new CorotationalFEM(mesh, params);
  const pd = new ProjectiveDynamics(mesh, params);
  pd.iterations = 30;
  const dFem = settle(fem, 300);
  const dPd = settle(pd, 300);
  const L = 4;
  const t = 1;
  const analytic = -(rho * g * t * L ** 4) / (8 * E * (t ** 3 / 12));
  console.log(`  tip deflection  FEM=${dFem.toExponential(4)}  PD=${dPd.toExponential(4)}  Euler-Bernoulli=${analytic.toExponential(4)}`);
  // FEM and PD solve the same static problem (lambda = 0 -> identical energy)
  close(dFem / dPd, 1, 0.01, 'FEM vs PD');
  // coarse linear triangles are stiffer than beam theory (shear locking) but must be the right order
  assert.ok(dFem / analytic > 0.5 && dFem / analytic < 1.2, `ratio to analytic ${dFem / analytic}`);
});

test('static sag with nu = 0.3: FEM and PD agree to within a few percent', () => {
  const params = { E: 5e6, nu: 0.3, damping: 20 };
  const mesh = createRectMesh(4, 1, 24, 6);
  const fem = new CorotationalFEM(mesh, params);
  const pd = new ProjectiveDynamics(mesh, params);
  pd.iterations = 30;
  const dFem = settle(fem, 400);
  const dPd = settle(pd, 400);
  console.log(`  large deflection tip: FEM=${dFem.toFixed(4)} PD=${dPd.toFixed(4)}`);
  close(dFem / dPd, 1, 0.05, 'FEM vs PD');
  for (const v of fem.x) assert.ok(Number.isFinite(v));
  for (const v of pd.x) assert.ok(Number.isFinite(v));
});
