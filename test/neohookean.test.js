import test from 'node:test';
import assert from 'node:assert/strict';
import { svd2 } from '../src/math2d.js';
import { snhEnergy, snhPK1, snhHessian } from '../src/materials.js';

function close(a, b, tol, msg) {
  assert.ok(Math.abs(a - b) <= tol * (1 + Math.abs(a) + Math.abs(b)), `${msg ?? ''} ${a} vs ${b}`);
}

const MU = 1.9e6;
const LAMBDA = 2.8e6;
const SAMPLES = [
  [1, 0, 0, 1],
  [1.3, 0.2, -0.1, 0.8],
  [0.5, 0.3, 0.2, 0.9],
  [2.0, 0.1, 0.4, 1.5],
  [0.2, 0.9, 0.7, -0.3], // inverted (det < 0)
  [-1, 0, 0, 1], // reflection
  [0.01, 0, 0, 0.01], // nearly collapsed
];

function eigenvaluesSym4(H) {
  // cyclic Jacobi on a copy
  const a = Array.from({ length: 4 }, (_, i) => Array.from({ length: 4 }, (_, j) => H[4 * i + j]));
  for (let sweep = 0; sweep < 60; sweep++) {
    let off = 0;
    for (let i = 0; i < 4; i++) for (let j = i + 1; j < 4; j++) off += a[i][j] ** 2;
    if (off < 1e-24) break;
    for (let p = 0; p < 3; p++) {
      for (let q = p + 1; q < 4; q++) {
        if (Math.abs(a[p][q]) < 1e-300) continue;
        const th = (a[q][q] - a[p][p]) / (2 * a[p][q]);
        const t = Math.sign(th || 1) / (Math.abs(th) + Math.sqrt(th * th + 1));
        const c = 1 / Math.sqrt(t * t + 1);
        const s = t * c;
        for (let k = 0; k < 4; k++) {
          const akp = a[k][p];
          const akq = a[k][q];
          a[k][p] = c * akp - s * akq;
          a[k][q] = s * akp + c * akq;
        }
        for (let k = 0; k < 4; k++) {
          const apk = a[p][k];
          const aqk = a[q][k];
          a[p][k] = c * apk - s * aqk;
          a[q][k] = s * apk + c * aqk;
        }
      }
    }
  }
  return [a[0][0], a[1][1], a[2][2], a[3][3]].sort((x, y) => x - y);
}

test('svd2 reconstructs F with proper rotations and signed singular values', () => {
  const o = new Float64Array(6);
  for (const [a, b, c, d] of SAMPLES) {
    svd2(a, b, c, d, o);
    const [cp, sp, ct, st, s0, s1] = o;
    // F = Rot(phi) diag(s0,s1) Rot(theta)
    const m00 = cp * s0 * ct - sp * s1 * st;
    const m01 = cp * s0 * -st - sp * s1 * ct;
    const m10 = sp * s0 * ct + cp * s1 * st;
    const m11 = sp * s0 * -st + cp * s1 * ct;
    close(m00, a, 1e-12, 'm00');
    close(m01, b, 1e-12, 'm01');
    close(m10, c, 1e-12, 'm10');
    close(m11, d, 1e-12, 'm11');
    assert.ok(s0 >= Math.abs(s1) - 1e-14);
    assert.equal(s1 < 0, a * d - b * c < 0);
  }
});

test('Stable NH: rest state has zero energy and zero stress', () => {
  close(snhEnergy(MU, LAMBDA, 1, 0, 0, 1), 0, 1e-14);
  const P = snhPK1(MU, LAMBDA, 1, 0, 0, 1, new Float64Array(4));
  for (const v of P) close(v, 0, 1e-14);
});

test('Stable NH: rotation invariant, finite for inverted F', () => {
  const [a, b, c, d] = SAMPLES[1];
  const th = 0.7;
  const cs = Math.cos(th);
  const sn = Math.sin(th);
  const e0 = snhEnergy(MU, LAMBDA, a, b, c, d);
  const e1 = snhEnergy(MU, LAMBDA, cs * a - sn * c, cs * b - sn * d, sn * a + cs * c, sn * b + cs * d);
  close(e0, e1, 1e-12);
  for (const F of SAMPLES) assert.ok(Number.isFinite(snhEnergy(MU, LAMBDA, ...F)));
});

test('Stable NH: PK1 equals the finite-difference gradient of the energy', () => {
  const P = new Float64Array(4);
  for (const F of SAMPLES) {
    snhPK1(MU, LAMBDA, ...F, P);
    for (let k = 0; k < 4; k++) {
      const h = 1e-6;
      const Fp = F.slice();
      const Fm = F.slice();
      Fp[k] += h;
      Fm[k] -= h;
      const fd = (snhEnergy(MU, LAMBDA, ...Fp) - snhEnergy(MU, LAMBDA, ...Fm)) / (2 * h);
      // central-difference round-off scales with |Psi|/h, so compare against the stress scale mu
      assert.ok(Math.abs(P[k] - fd) <= 1e-6 * MU * (1 + Math.abs(F[0]) + Math.abs(F[3])), `P[${k}] ${P[k]} vs ${fd}`);
    }
  }
});

test('Stable NH: unprojected Hessian equals the finite-difference Jacobian of PK1', () => {
  const H = new Float64Array(16);
  const Pp = new Float64Array(4);
  const Pm = new Float64Array(4);
  for (const F of SAMPLES) {
    snhHessian(MU, LAMBDA, ...F, H, false);
    for (let k = 0; k < 4; k++) {
      const h = 1e-6;
      const Fp = F.slice();
      const Fm = F.slice();
      Fp[k] += h;
      Fm[k] -= h;
      snhPK1(MU, LAMBDA, ...Fp, Pp);
      snhPK1(MU, LAMBDA, ...Fm, Pm);
      for (let i = 0; i < 4; i++) close(H[4 * i + k], (Pp[i] - Pm[i]) / (2 * h), 1e-5, `H[${i},${k}]`);
    }
  }
});

test('Stable NH: projected Hessian is PSD and equals the exact one when already PSD', () => {
  const Hp = new Float64Array(16);
  const He = new Float64Array(16);
  for (const F of SAMPLES) {
    snhHessian(MU, LAMBDA, ...F, Hp, true);
    snhHessian(MU, LAMBDA, ...F, He, false);
    for (let i = 0; i < 4; i++) for (let j = 0; j < 4; j++) close(Hp[4 * i + j], Hp[4 * j + i], 1e-9, 'sym');
    const ev = eigenvaluesSym4(Hp);
    const scale = Math.abs(ev[3]) + 1;
    assert.ok(ev[0] >= -1e-9 * scale, `min eigenvalue ${ev[0]} for F=${F}`);
    const evE = eigenvaluesSym4(He);
    if (evE[0] >= -1e-9 * scale) {
      for (let i = 0; i < 16; i++) close(Hp[i], He[i], 1e-8, 'unchanged');
    }
  }
});

test('Stable NH: Hessian at rest has eigenvalues {0, 2mu, 2mu, 2lambda}', () => {
  const H = new Float64Array(16);
  snhHessian(MU, LAMBDA, 1, 0, 0, 1, H, true);
  const ev = eigenvaluesSym4(H);
  const want = [0, 2 * MU, 2 * MU, 2 * LAMBDA].sort((a, b) => a - b);
  for (let i = 0; i < 4; i++) close(ev[i], want[i], 1e-9, `ev${i}`);
});

// ---------------------------------------------------------------------------------------------
// HyperelasticFEM (Newton)
// ---------------------------------------------------------------------------------------------
import { createRectMesh } from '../src/mesh.js';
import { CorotationalFEM } from '../src/sim/corotationalFEM.js';
import { HyperelasticFEM } from '../src/sim/hyperelasticFEM.js';
import { csrMul } from '../src/linalg.js';

function perturbed(mesh, amp) {
  const x = Float64Array.from(mesh.X);
  for (let i = 0; i < x.length; i++) x[i] += amp * Math.sin(3 * i + 1);
  return x;
}

test('HyperelasticFEM: elasticForce equals the negative gradient of elasticEnergy and internalForces', () => {
  const mesh = createRectMesh(2, 1, 6, 3);
  const fem = new HyperelasticFEM(mesh, { E: 1e5, nu: 0.3 });
  const x = perturbed(mesh, 0.1);
  const f = new Float64Array(x.length);
  const fi = new Float64Array(x.length);
  fem.elasticForce(x, f);
  fem.internalForces(x, fi);
  for (let i = 0; i < f.length; i++) close(f[i], fi[i], 1e-9, `internalForces dof ${i}`);
  for (let i = 0; i < x.length; i++) {
    const h = 1e-6;
    const xp = x.slice();
    const xm = x.slice();
    xp[i] += h;
    xm[i] -= h;
    const fd = -(fem.elasticEnergy(xp) - fem.elasticEnergy(xm)) / (2 * h);
    assert.ok(Math.abs(f[i] - fd) <= 1e-5 * (1 + Math.abs(f[i])), `dof ${i}: ${f[i]} vs ${fd}`);
  }
});

test('HyperelasticFEM: assembled tangent stiffness matches the finite-difference Jacobian of the force', () => {
  const mesh = createRectMesh(2, 1, 4, 2);
  const fem = new HyperelasticFEM(mesh, { E: 1e5, nu: 0.3 });
  fem.projectHessian = false; // the projection (intentionally) differs from the exact Jacobian for J < 1
  const x = perturbed(mesh, 0.05);
  const f = new Float64Array(x.length);
  fem.elasticForce(x, f, true);
  const { rowPtr, colIdx, vals } = fem.csr;
  const N = x.length;
  const dir = new Float64Array(N);
  const Kd = new Float64Array(N);
  const fp = new Float64Array(N);
  const fm = new Float64Array(N);
  for (let j = 0; j < N; j++) {
    dir.fill(0);
    dir[j] = 1;
    csrMul(rowPtr, colIdx, vals, dir, Kd);
    const h = 1e-6;
    const xp = x.slice();
    const xm = x.slice();
    xp[j] += h;
    xm[j] -= h;
    fem.elasticForce(xp, fp);
    fem.elasticForce(xm, fm);
    for (let i = 0; i < N; i++) {
      const fd = -(fp[i] - fm[i]) / (2 * h);
      assert.ok(Math.abs(Kd[i] - fd) <= 1e-5 * (1e5 + Math.abs(fd)), `K[${i},${j}] ${Kd[i]} vs ${fd}`);
    }
  }
});

function settleSag(sim, steps, h = 1 / 120) {
  for (let i = 0; i < steps; i++) sim.step(h);
  return sim.tip[1] - sim.mesh.X[2 * sim.mesh.tipIdx + 1];
}

test('HyperelasticFEM: small-deformation sag matches corotational FEM (same linear limit)', () => {
  const params = { E: 5e8, nu: 0.3, damping: 20 };
  const mesh = createRectMesh(4, 1, 24, 6);
  const co = new CorotationalFEM(mesh, params);
  const nh = new HyperelasticFEM(mesh, params);
  const dCo = settleSag(co, 300);
  const dNh = settleSag(nh, 300);
  console.log(`  tip deflection  corotational=${dCo.toExponential(4)}  neo-hookean=${dNh.toExponential(4)}`);
  assert.ok(Math.abs(dNh / dCo - 1) < 0.01, `ratio ${dNh / dCo}`);
});

test('HyperelasticFEM: rest state is a fixed point without gravity', () => {
  const mesh = createRectMesh(4, 1, 12, 3);
  const nh = new HyperelasticFEM(mesh, { E: 1e6, nu: 0.3, gravity: 0 });
  for (let i = 0; i < 20; i++) nh.step(1 / 60);
  for (let i = 0; i < nh.x.length; i++) assert.ok(Math.abs(nh.x[i] - mesh.X[i]) < 1e-9);
});

// ---------------------------------------------------------------------------------------------
// Projective Dynamics with Neo-Hookean (constant PD matrix + L-BFGS)
// ---------------------------------------------------------------------------------------------
import { ProjectiveDynamics } from '../src/sim/projectiveDynamics.js';
import { StretchDriver } from '../src/stretch.js';

test('PD (Neo-Hookean): gradient matches finite differences of the objective, incl. inverted elements', () => {
  const mesh = createRectMesh(2, 1, 6, 3);
  const pd = new ProjectiveDynamics(mesh, { E: 1e5, nu: 0.3, model: 'neohookean' });
  pd.setContact({ enabled: true, floorY: 0.1, k: 1e4 });
  pd.step(1 / 240); // builds the factorisation
  const h2 = (1 / 240) ** 2;
  const x = perturbed(mesh, 0.3); // large perturbation: some elements are strongly compressed / inverted
  const y = Float64Array.from(x, (v, i) => v + 0.01 * Math.cos(i));
  const nf = pd._nf;
  const g = new Float64Array(2 * nf);
  pd._gradient(x, y, h2, g);
  assert.ok(pd.deformationStats(x).minDet < 0.5, 'test setup should be strongly deformed');
  const free = [];
  for (let i = 0; i < mesh.n; i++) if (pd._fidx[i] >= 0) free.push(i);
  for (const v of [free[0], free[3], free[7], free[12], free[free.length - 1]]) {
    for (const comp of [0, 1]) {
      const eps = 1e-6;
      const xp = Float64Array.from(x);
      const xm = Float64Array.from(x);
      xp[2 * v + comp] += eps;
      xm[2 * v + comp] -= eps;
      const fd = (pd._objective(xp, y, h2) - pd._objective(xm, y, h2)) / (2 * eps);
      const an = g[comp * nf + pd._fidx[v]];
      assert.ok(Math.abs(fd - an) <= 1e-4 * Math.max(1, Math.abs(an)), `vertex ${v} comp ${comp}: ${an} vs ${fd}`);
    }
  }
});

test('PD (Neo-Hookean) and Newton FEM minimise the same objective: static sags agree', () => {
  const params = { E: 5e6, nu: 0.3, damping: 20, model: 'neohookean' };
  const mesh = createRectMesh(4, 1, 24, 6);
  const fem = new HyperelasticFEM(mesh, params);
  const pd = new ProjectiveDynamics(mesh, params);
  pd.iterations = 30;
  const dFem = settleSag(fem, 400);
  const dPd = settleSag(pd, 400);
  console.log(`  large-deflection tip: Newton FEM=${dFem.toFixed(4)}  PD L-BFGS=${dPd.toFixed(4)}`);
  assert.ok(Math.abs(dFem / dPd - 1) < 0.01, `ratio ${dFem / dPd}`);
  for (const v of pd.x) assert.ok(Number.isFinite(v));
});

for (const [name, S] of [
  ['Newton FEM', HyperelasticFEM],
  ['PD L-BFGS', ProjectiveDynamics],
]) {
  test(`${name}: nu = 0.4 stretch to 4x never inverts (corotational inverts at 2.5x) and recovers`, () => {
    const mesh = createRectMesh(1, 1, 10, 10);
    const sim = new S(mesh, { E: 1e5, nu: 0.4, density: 1000, gravity: 0, damping: 5, model: 'neohookean' });
    if (S === ProjectiveDynamics) sim.iterations = 30;
    const drv = new StretchDriver(mesh, { speed: 1, maxStretch: 4 });
    drv.attach(sim);
    const h = 1 / 120;
    let maxInv = 0;
    for (let i = 0; i < 120 * 8 && drv.mode !== 'hold'; i++) {
      drv.advance(h, sim);
      sim.step(h);
      maxInv = Math.max(maxInv, sim.deformationStats().inverted);
    }
    assert.equal(drv.mode, 'hold');
    assert.equal(maxInv, 0);
    drv.mode = 'unload';
    for (let i = 0; i < 120 * 10; i++) {
      drv.advance(h, sim);
      sim.step(h);
    }
    assert.equal(drv.d, 0);
    let err = 0;
    for (let i = 0; i < sim.x.length; i++) err = Math.max(err, Math.abs(sim.x[i] - mesh.X[i]));
    assert.ok(err < 2e-3, `max deviation from rest ${err}`);
  });
}

test('Neo-Hookean recovers from an inverted (folded) element configuration', () => {
  // Flip a block of vertices through the plane so several triangles have det F < 0.
  const mesh = createRectMesh(1, 1, 6, 6);
  const fem = new HyperelasticFEM(mesh, { E: 1e5, nu: 0.3, density: 1000, gravity: 0, damping: 10 });
  fem.setPinned([]); // free body
  for (let i = 0; i < fem.n; i++) {
    const px = fem.x[2 * i];
    if (px > 0.5) fem.x[2 * i] = 1 - px; // fold the right half over the left half
  }
  assert.ok(fem.deformationStats().inverted > 0, 'setup must contain inverted elements');
  for (let i = 0; i < 120 * 6; i++) fem.step(1 / 120);
  assert.equal(fem.deformationStats().inverted, 0);
  for (const v of fem.x) assert.ok(Number.isFinite(v));
});
