import { svd2 } from './math2d.js';

/**
 * Stable Neo-Hookean (Smith, de Goes & Kim 2018), 2D form:
 *
 *   Psi(F) = mu/2 (||F||^2 - 2) - mu (J - 1) + lambda/2 (J - 1)^2,   J = det F
 *
 * Psi(I) = 0 and P(I) = 0 (rest-stable), and Psi stays finite for J <= 0, so inverted elements
 * are pushed back instead of blowing up. F is passed row-major as scalars f00 f01 f10 f11.
 * All functions are allocation-free apart from the caller-provided `out` arrays.
 */

/** Strain energy density (per unit rest area). */
export function snhEnergy(mu, lambda, f00, f01, f10, f11) {
  const J = f00 * f11 - f01 * f10;
  const n2 = f00 * f00 + f01 * f01 + f10 * f10 + f11 * f11;
  return 0.5 * mu * (n2 - 2) - mu * (J - 1) + 0.5 * lambda * (J - 1) * (J - 1);
}

/** First Piola-Kirchhoff stress  P = mu F + (lambda (J - 1) - mu) cof(F), row-major into out[0..3]. */
export function snhPK1(mu, lambda, f00, f01, f10, f11, out) {
  const J = f00 * f11 - f01 * f10;
  const s = lambda * (J - 1) - mu;
  out[0] = mu * f00 + s * f11;
  out[1] = mu * f01 - s * f10;
  out[2] = mu * f10 - s * f01;
  out[3] = mu * f11 + s * f00;
  return out;
}

const _svd = new Float64Array(6);

/**
 * Hessian d^2 Psi / dF^2 as a symmetric 4x4 (row-major, index order f00 f01 f10 f11) into out[0..15].
 *
 *   H = mu I + lambda g g^T + s H_J,   g = vec(cof F),  s = lambda (J - 1) - mu
 *
 * With project = true negative eigenvalues are clamped to zero (Teran et al. 2005; Smith et al.
 * 2019), which keeps the Newton system positive definite. In the frame F = U diag(s0,s1) V^T the
 * eigen-system is closed-form:
 *   twist  [[0,-1],[1,0]]/sqrt2 : lambda (J - 1)
 *   flip   [[0, 1],[1,0]]/sqrt2 : 2 mu - lambda (J - 1)
 *   scale  span{diag(1,0), diag(0,1)} : 2x2 block [[mu+l s1^2, l s0 s1 + s], [., mu + l s0^2]]
 */
export function snhHessian(mu, lambda, f00, f01, f10, f11, out, project = true) {
  const J = f00 * f11 - f01 * f10;
  const s = lambda * (J - 1) - mu;
  if (!project) {
    const g0 = f11;
    const g1 = -f10;
    const g2 = -f01;
    const g3 = f00;
    const g = [g0, g1, g2, g3];
    for (let i = 0; i < 4; i++) {
      for (let j = 0; j < 4; j++) out[4 * i + j] = lambda * g[i] * g[j] + (i === j ? mu : 0);
    }
    // H_J: (0,3) = (3,0) = 1 ; (1,2) = (2,1) = -1
    out[3] += s;
    out[12] += s;
    out[6] -= s;
    out[9] -= s;
    return out;
  }

  svd2(f00, f01, f10, f11, _svd);
  const cp = _svd[0];
  const sp = _svd[1];
  const ct = _svd[2];
  const st = _svd[3];
  const s0 = _svd[4];
  const s1 = _svd[5];
  out.fill(0);

  // accumulate  w * vec(U A V^T) vec(U A V^T)^T  for a Sigma-frame mode matrix A = [[a b],[c d]]
  const addMode = (w, a, b, c, d) => {
    if (w <= 0) return;
    // B = U A, then M = B V^T with U = Rot(phi), V^T = Rot(theta)
    const b00 = cp * a - sp * c;
    const b01 = cp * b - sp * d;
    const b10 = sp * a + cp * c;
    const b11 = sp * b + cp * d;
    const m0 = b00 * ct + b01 * st;
    const m1 = -b00 * st + b01 * ct;
    const m2 = b10 * ct + b11 * st;
    const m3 = -b10 * st + b11 * ct;
    const m = [m0, m1, m2, m3];
    for (let i = 0; i < 4; i++) {
      for (let j = 0; j < 4; j++) out[4 * i + j] += w * m[i] * m[j];
    }
  };

  const R2 = Math.SQRT1_2;
  addMode(lambda * (J - 1), 0, -R2, R2, 0); // twist
  addMode(2 * mu - lambda * (J - 1), 0, R2, R2, 0); // flip

  // scaling block
  const p = mu + lambda * s1 * s1;
  const r = mu + lambda * s0 * s0;
  const q = lambda * s0 * s1 + s;
  const mean = 0.5 * (p + r);
  const rad = Math.hypot(0.5 * (p - r), q);
  const psi = 0.5 * Math.atan2(2 * q, p - r);
  const c = Math.cos(psi);
  const sn = Math.sin(psi);
  addMode(mean + rad, c, 0, 0, sn);
  addMode(mean - rad, -sn, 0, 0, c);
  return out;
}

/**
 * Unified material interface used by the solvers. `model` is 'corotational' or 'neohookean'.
 * Only the Stable Neo-Hookean lives here; the corotational path keeps its original code in base.js.
 */
export const StableNeoHookean = {
  energy: snhEnergy,
  pk1: snhPK1,
  hessian: snhHessian,
};
