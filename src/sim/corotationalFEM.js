import { SoftBodyBase } from './base.js';
import { polarRotation } from '../math2d.js';
import { pcg, csrMul } from '../linalg.js';

/**
 * Corotational linear-triangle FEM (Mueller & Gross 2004 style).
 *
 *  - per-element polar decomposition F = R S gives the element rotation R
 *  - elastic force  f_e = -R Ke (R^T x_e - X_e)          (exact gradient for fixed R)
 *  - warped stiffness K'_e = R Ke R^T
 *  - implicit Euler, one linear solve per step:
 *        (M + h^2 K') dv = h (f_el + f_ext) - h^2 K' v
 *    solved with Jacobi-preconditioned CG.
 *  - velocity damping v <- v / (1 + h d) (identical in the PD solver)
 */
export class CorotationalFEM extends SoftBodyBase {
  constructor(mesh, params = {}) {
    super(mesh, params);
    this.cg = { tol: 1e-6, maxIter: 300 };
    this.contactPasses = 3; // max re-linearisations of the contact active set per step
    this._buildPattern();
    this._buildElementStiffness();
    const N = 2 * this.n;
    this._rhs = new Float64Array(N);
    this._rhsBase = new Float64Array(N);
    this._kC = new Float64Array(N);
    this._dv = new Float64Array(N);
    this._tmp = new Float64Array(N);
    this._f = new Float64Array(N);
    this._diag = new Float64Array(N);
    this._free = new Uint8Array(N);
    this._work = { r: new Float64Array(N), z: new Float64Array(N), p: new Float64Array(N), Ap: new Float64Array(N) };
    this.onPinnedChanged();
  }

  onParamsChanged() {
    this._buildElementStiffness();
  }

  onPinnedChanged() {
    if (!this._free) return;
    for (let i = 0; i < this.n; i++) {
      this._free[2 * i] = this._free[2 * i + 1] = this.pinned[i] ? 0 : 1;
    }
  }

  /** CSR sparsity pattern (scalar DOFs) and per-element scatter offsets. */
  _buildPattern() {
    const { tris, n } = this.mesh;
    const m = tris.length / 3;
    const nbr = Array.from({ length: n }, () => new Set());
    for (let e = 0; e < m; e++) {
      for (let a = 0; a < 3; a++) for (let b = 0; b < 3; b++) nbr[tris[3 * e + a]].add(tris[3 * e + b]);
    }
    const sorted = nbr.map((s) => Array.from(s).sort((p, q) => p - q));
    const rowPtr = new Int32Array(2 * n + 1);
    for (let i = 0; i < n; i++) {
      rowPtr[2 * i + 1] = rowPtr[2 * i] + 2 * sorted[i].length;
      rowPtr[2 * i + 2] = rowPtr[2 * i + 1] + 2 * sorted[i].length;
    }
    const colIdx = new Int32Array(rowPtr[2 * n]);
    for (let i = 0; i < n; i++) {
      sorted[i].forEach((j, q) => {
        for (let k = 0; k < 2; k++) {
          colIdx[rowPtr[2 * i + k] + 2 * q] = 2 * j;
          colIdx[rowPtr[2 * i + k] + 2 * q + 1] = 2 * j + 1;
        }
      });
    }
    // scatter[e][a][b] -> offsets of the 2x2 block (row 0 start, row 1 start)
    const scatter = new Int32Array(m * 9 * 2);
    for (let e = 0; e < m; e++) {
      for (let a = 0; a < 3; a++) {
        const i = tris[3 * e + a];
        for (let b = 0; b < 3; b++) {
          const q = sorted[i].indexOf(tris[3 * e + b]);
          scatter[(e * 9 + a * 3 + b) * 2] = rowPtr[2 * i] + 2 * q;
          scatter[(e * 9 + a * 3 + b) * 2 + 1] = rowPtr[2 * i + 1] + 2 * q;
        }
      }
    }
    this.csr = { rowPtr, colIdx, vals: new Float64Array(colIdx.length), diag: new Float64Array(2 * n) };
    this._scatter = scatter;
  }

  /** Ke (6x6, row-major per element) of linear elasticity for the rest triangle. */
  _buildElementStiffness() {
    const { Dinv, area } = this.mesh;
    const m = area.length;
    const { mu, lambda } = this;
    const Ke = (this.Ke = new Float64Array(36 * m));
    const eps = new Float64Array(18); // 6 dofs x (e00, e01, e11)
    for (let e = 0; e < m; e++) {
      const g = [
        [-(Dinv[4 * e] + Dinv[4 * e + 2]), -(Dinv[4 * e + 1] + Dinv[4 * e + 3])],
        [Dinv[4 * e], Dinv[4 * e + 1]],
        [Dinv[4 * e + 2], Dinv[4 * e + 3]],
      ];
      for (let j = 0; j < 3; j++) {
        // dof (j, x): H = [[g0, g1], [0, 0]]
        eps[3 * (2 * j)] = g[j][0];
        eps[3 * (2 * j) + 1] = 0.5 * g[j][1];
        eps[3 * (2 * j) + 2] = 0;
        // dof (j, y): H = [[0, 0], [g0, g1]]
        eps[3 * (2 * j + 1)] = 0;
        eps[3 * (2 * j + 1) + 1] = 0.5 * g[j][0];
        eps[3 * (2 * j + 1) + 2] = g[j][1];
      }
      for (let p = 0; p < 6; p++) {
        for (let q = 0; q < 6; q++) {
          const dot = eps[3 * p] * eps[3 * q] + 2 * eps[3 * p + 1] * eps[3 * q + 1] + eps[3 * p + 2] * eps[3 * q + 2];
          const trp = eps[3 * p] + eps[3 * p + 2];
          const trq = eps[3 * q] + eps[3 * q + 2];
          Ke[36 * e + 6 * p + q] = area[e] * (2 * mu * dot + lambda * trp * trq);
        }
      }
    }
  }

  /** Element rotation from the current positions. */
  _rotation(x, e, out) {
    const { tris, Dinv } = this.mesh;
    const i0 = tris[3 * e];
    const i1 = tris[3 * e + 1];
    const i2 = tris[3 * e + 2];
    const d1x = x[2 * i1] - x[2 * i0];
    const d1y = x[2 * i1 + 1] - x[2 * i0 + 1];
    const d2x = x[2 * i2] - x[2 * i0];
    const d2y = x[2 * i2 + 1] - x[2 * i0 + 1];
    const a = Dinv[4 * e];
    const b = Dinv[4 * e + 1];
    const c = Dinv[4 * e + 2];
    const d = Dinv[4 * e + 3];
    return polarRotation(d1x * a + d2x * c, d1x * b + d2x * d, d1y * a + d2y * c, d1y * b + d2y * d, out);
  }

  /**
   * Elastic force at positions x (output f, overwritten). If `assemble` is true
   * also rebuilds the warped stiffness matrix in this.csr.
   */
  elasticForce(x, f, assemble = false) {
    const { tris, X } = this.mesh;
    const m = tris.length / 3;
    const Ke = this.Ke;
    const { vals, diag } = this.csr;
    f.fill(0);
    if (assemble) {
      vals.fill(0);
      diag.fill(0);
    }
    const rot = [1, 0];
    const u = new Float64Array(6);
    const fl = new Float64Array(6);
    for (let e = 0; e < m; e++) {
      this._rotation(x, e, rot);
      const c = rot[0];
      const s = rot[1];
      for (let a = 0; a < 3; a++) {
        const i = tris[3 * e + a];
        // u_a = R^T x_a - X_a
        u[2 * a] = c * x[2 * i] + s * x[2 * i + 1] - X[2 * i];
        u[2 * a + 1] = -s * x[2 * i] + c * x[2 * i + 1] - X[2 * i + 1];
      }
      const o = 36 * e;
      for (let p = 0; p < 6; p++) {
        let acc = 0;
        for (let q = 0; q < 6; q++) acc += Ke[o + 6 * p + q] * u[q];
        fl[p] = -acc;
      }
      for (let a = 0; a < 3; a++) {
        const i = tris[3 * e + a];
        f[2 * i] += c * fl[2 * a] - s * fl[2 * a + 1];
        f[2 * i + 1] += s * fl[2 * a] + c * fl[2 * a + 1];
      }
      if (!assemble) continue;
      // K'_ab = R Ke_ab R^T  (2x2 blocks)
      for (let a = 0; a < 3; a++) {
        for (let b = 0; b < 3; b++) {
          const k00 = Ke[o + 6 * (2 * a) + 2 * b];
          const k01 = Ke[o + 6 * (2 * a) + 2 * b + 1];
          const k10 = Ke[o + 6 * (2 * a + 1) + 2 * b];
          const k11 = Ke[o + 6 * (2 * a + 1) + 2 * b + 1];
          // T = R K
          const t00 = c * k00 - s * k10;
          const t01 = c * k01 - s * k11;
          const t10 = s * k00 + c * k10;
          const t11 = s * k01 + c * k11;
          // (T R^T): R^T = [[c, s], [-s, c]]
          const r00 = t00 * c - t01 * s;
          const r01 = t00 * s + t01 * c;
          const r10 = t10 * c - t11 * s;
          const r11 = t10 * s + t11 * c;
          const base0 = this._scatter[(e * 9 + a * 3 + b) * 2];
          const base1 = this._scatter[(e * 9 + a * 3 + b) * 2 + 1];
          vals[base0] += r00;
          vals[base0 + 1] += r01;
          vals[base1] += r10;
          vals[base1 + 1] += r11;
          if (a === b) {
            const i = tris[3 * e + a];
            diag[2 * i] += r00;
            diag[2 * i + 1] += r11;
          }
        }
      }
    }
    return f;
  }

  step(h) {
    const t0 = performance.now();
    const { x, v, mass, n } = this;
    const { gravity, damping } = this.params;
    const N = 2 * n;
    const { rowPtr, colIdx, vals, diag: kdiag } = this.csr;

    // velocity damping (implicit): v <- v / (1 + h d)
    const damp = 1 / (1 + h * damping);
    for (let i = 0; i < N; i++) v[i] *= damp;
    // prescribed motion: pinned vertices move to their target within this step (dv = 0 there)
    for (let i = 0; i < n; i++) {
      if (this.pinned[i]) {
        v[2 * i] = (this.pinTarget[2 * i] - x[2 * i]) / h;
        v[2 * i + 1] = (this.pinTarget[2 * i + 1] - x[2 * i + 1]) / h;
      }
    }

    const f = this.elasticForce(x, this._f, true);
    // rhs = h (f_el + f_ext + m g) - h^2 K' v
    const Kv = this._tmp;
    csrMul(rowPtr, colIdx, vals, v, Kv);
    const rhsBase = this._rhsBase;
    for (let i = 0; i < n; i++) {
      rhsBase[2 * i] = h * (f[2 * i] + this.fext[2 * i]) - h * h * Kv[2 * i];
      rhsBase[2 * i + 1] = h * (f[2 * i + 1] + this.fext[2 * i + 1] - mass[i] * gravity) - h * h * Kv[2 * i + 1];
    }
    const h2 = h * h;
    const kC = this._kC; // contact stiffness on the y-DOFs (diagonal)
    const rhs = this._rhs;
    const diag = this._diag;
    const apply = (p, out) => {
      csrMul(rowPtr, colIdx, vals, p, out);
      for (let i = 0; i < n; i++) {
        out[2 * i] = mass[i] * p[2 * i] + h2 * out[2 * i];
        out[2 * i + 1] = (mass[i] + h2 * kC[2 * i + 1]) * p[2 * i + 1] + h2 * out[2 * i + 1];
      }
    };

    // Floor contact: penalty force k m_i (floorY - y)_+ evaluated at the predicted end-of-step
    // position x + h (v + dv) and linearised there (adds h^2 k m_i to the A diagonal of active
    // vertices). The active set is re-checked after the solve (a few passes at most).
    const { enabled, floorY, k } = this.contact;
    const dv = this._dv;
    dv.fill(0);
    let iters = 0;
    const passes = enabled ? this.contactPasses : 1;
    for (let pass = 0; pass < passes; pass++) {
      kC.fill(0);
      rhs.set(rhsBase);
      if (enabled) {
        for (let i = 0; i < n; i++) {
          const yl = x[2 * i + 1] + h * (v[2 * i + 1] + dv[2 * i + 1]);
          if (floorY - yl > 0) {
            const ki = k * mass[i];
            kC[2 * i + 1] = ki;
            // h f_c(x_l) + h^2 K_c dv_l
            rhs[2 * i + 1] += h * ki * (floorY - yl) + h2 * ki * dv[2 * i + 1];
          }
        }
      }
      for (let i = 0; i < n; i++) {
        diag[2 * i] = mass[i] + h2 * kdiag[2 * i];
        diag[2 * i + 1] = mass[i] + h2 * (kdiag[2 * i + 1] + kC[2 * i + 1]);
      }
      iters += pcg({
        apply,
        diag,
        b: rhs,
        x: dv,
        free: this._free,
        tol: this.cg.tol,
        maxIter: this.cg.maxIter,
        work: this._work,
      });
      if (!enabled) break;
      // did the active set at the new predicted position change?
      let changed = false;
      for (let i = 0; i < n; i++) {
        const now = floorY - (x[2 * i + 1] + h * (v[2 * i + 1] + dv[2 * i + 1])) > 0 ? 1 : 0;
        if ((kC[2 * i + 1] > 0 ? 1 : 0) !== now) changed = true;
      }
      if (!changed) break;
    }
    for (let i = 0; i < N; i++) {
      v[i] += dv[i];
      x[i] += h * v[i];
    }
    for (let i = 0; i < n; i++) {
      if (this.pinned[i]) {
        x[2 * i] = this.pinTarget[2 * i];
        x[2 * i + 1] = this.pinTarget[2 * i + 1];
      }
    }
    this.stats.iterations = iters;
    this.stats.stepMs = performance.now() - t0;
  }
}
