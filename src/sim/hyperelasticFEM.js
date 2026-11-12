import { SoftBodyBase } from './base.js';
import { pcg, csrMul } from '../linalg.js';
import { snhPK1, snhHessian } from '../materials.js';
import { buildCsrPattern } from './csrPattern.js';

/**
 * Linear-triangle FEM with the Stable Neo-Hookean material and a full Newton solver.
 *
 * Implicit Euler is posed as the minimisation of the incremental potential
 *
 *   g(x) = sum_i m_i / (2 h^2) |x_i - y_i|^2 + sum_e area_e Psi(F_e(x)) + E_contact(x),
 *   y = x_n + h v_n + h^2 (f_ext / m - g e_y)
 *
 * (the very same problem the Projective Dynamics solver minimises). Every Newton iteration
 * assembles the exact gradient and the element Hessians with their negative eigenvalues clamped
 * (positive definite even for inverted elements), solves
 *
 *   (M / h^2 + K) dx = -grad g
 *
 * with Jacobi-preconditioned CG and backtracks on g.
 */
export class HyperelasticFEM extends SoftBodyBase {
  constructor(mesh, params = {}) {
    super(mesh, { ...params, model: 'neohookean' });
    this.cg = { tol: 1e-6, maxIter: 300 };
    this.newton = { tol: 1e-9, maxIter: 10 };
    this.projectHessian = true; // clamp negative eigenvalues of the element Hessians (needed for CG)
    const { csr, scatter } = buildCsrPattern(mesh);
    this.csr = csr;
    this._scatter = scatter;
    const N = 2 * this.n;
    this._xn = new Float64Array(N);
    this._y = new Float64Array(N);
    this._grad = new Float64Array(N);
    this._dx = new Float64Array(N);
    this._rhs = new Float64Array(N);
    this._xt = new Float64Array(N);
    this._diag = new Float64Array(N);
    this._kC = new Float64Array(N);
    this._free = new Uint8Array(N);
    this._work = { r: new Float64Array(N), z: new Float64Array(N), p: new Float64Array(N), Ap: new Float64Array(N) };
    this._pk = new Float64Array(4);
    this._H = new Float64Array(16);
    this.onPinnedChanged();
  }

  onPinnedChanged() {
    if (!this._free) return;
    for (let i = 0; i < this.n; i++) {
      this._free[2 * i] = this._free[2 * i + 1] = this.pinned[i] ? 0 : 1;
    }
  }

  /**
   * Elastic force f = -dE/dx into f (overwritten). If `assemble`, also rebuilds the projected
   * (PSD) tangent stiffness K in this.csr.
   */
  elasticForce(x, f, assemble = false) {
    const { tris, Dinv, area } = this.mesh;
    const m = area.length;
    const { mu } = this;
    const lambda = this.lambdaNH; // see SoftBodyBase._updateMaterial
    const { vals, diag } = this.csr;
    const pk = this._pk;
    const H = this._H;
    const gs = new Float64Array(6);
    const Ke = new Float64Array(36);
    f.fill(0);
    if (assemble) {
      vals.fill(0);
      diag.fill(0);
    }
    for (let e = 0; e < m; e++) {
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
      const f00 = d1x * a + d2x * c;
      const f01 = d1x * b + d2x * d;
      const f10 = d1y * a + d2y * c;
      const f11 = d1y * b + d2y * d;
      // shape-function gradients g0 = -(g1 + g2), g1 = (a, b), g2 = (c, d)
      gs[0] = -(a + c);
      gs[1] = -(b + d);
      gs[2] = a;
      gs[3] = b;
      gs[4] = c;
      gs[5] = d;
      const A = area[e];
      snhPK1(mu, lambda, f00, f01, f10, f11, pk);
      for (let k = 0; k < 3; k++) {
        const i = tris[3 * e + k];
        f[2 * i] -= A * (pk[0] * gs[2 * k] + pk[1] * gs[2 * k + 1]);
        f[2 * i + 1] -= A * (pk[2] * gs[2 * k] + pk[3] * gs[2 * k + 1]);
      }
      if (!assemble) continue;
      snhHessian(mu, lambda, f00, f01, f10, f11, H, this.projectHessian);
      // K_e[(a,j),(b,l)] = A sum_{c,d} H[(j,c),(l,d)] g_{a,c} g_{b,d}
      for (let p = 0; p < 3; p++) {
        for (let j = 0; j < 2; j++) {
          for (let q = 0; q < 3; q++) {
            for (let l = 0; l < 2; l++) {
              let s = 0;
              for (let cc = 0; cc < 2; cc++) {
                for (let dd = 0; dd < 2; dd++) s += H[4 * (2 * j + cc) + 2 * l + dd] * gs[2 * p + cc] * gs[2 * q + dd];
              }
              Ke[6 * (2 * p + j) + 2 * q + l] = A * s;
            }
          }
        }
      }
      for (let p = 0; p < 3; p++) {
        for (let q = 0; q < 3; q++) {
          const base0 = this._scatter[(e * 9 + p * 3 + q) * 2];
          const base1 = this._scatter[(e * 9 + p * 3 + q) * 2 + 1];
          vals[base0] += Ke[6 * (2 * p) + 2 * q];
          vals[base0 + 1] += Ke[6 * (2 * p) + 2 * q + 1];
          vals[base1] += Ke[6 * (2 * p + 1) + 2 * q];
          vals[base1 + 1] += Ke[6 * (2 * p + 1) + 2 * q + 1];
          if (p === q) {
            const i = tris[3 * e + p];
            diag[2 * i] += Ke[6 * (2 * p) + 2 * p];
            diag[2 * i + 1] += Ke[6 * (2 * p + 1) + 2 * p + 1];
          }
        }
      }
    }
    return f;
  }

  /** Incremental potential g(x) for the current inertial target y. */
  _objective(x, y, h2) {
    const { mass, n } = this;
    let e = 0;
    for (let i = 0; i < n; i++) {
      const dx = x[2 * i] - y[2 * i];
      const dy = x[2 * i + 1] - y[2 * i + 1];
      e += (mass[i] * (dx * dx + dy * dy)) / (2 * h2);
    }
    return e + this.elasticEnergy(x) + this.contactEnergy(x);
  }

  step(h) {
    const t0 = performance.now();
    const { x, v, mass, n } = this;
    const { gravity, damping } = this.params;
    const N = 2 * n;
    const h2 = h * h;
    const { rowPtr, colIdx, vals, diag: kdiag } = this.csr;

    const damp = 1 / (1 + h * damping);
    for (let i = 0; i < N; i++) v[i] *= damp;

    const xn = this._xn;
    xn.set(x);
    const y = this._y;
    for (let i = 0; i < n; i++) {
      y[2 * i] = x[2 * i] + h * v[2 * i] + h2 * (this.fext[2 * i] / mass[i]);
      y[2 * i + 1] = x[2 * i + 1] + h * v[2 * i + 1] + h2 * (this.fext[2 * i + 1] / mass[i] - gravity);
    }
    // initial guess y; pinned vertices follow their prescribed target
    for (let i = 0; i < n; i++) {
      if (this.pinned[i]) {
        x[2 * i] = this.pinTarget[2 * i];
        x[2 * i + 1] = this.pinTarget[2 * i + 1];
      } else {
        x[2 * i] = y[2 * i];
        x[2 * i + 1] = y[2 * i + 1];
      }
    }

    let eRef = 0;
    for (let i = 0; i < N; i++) eRef += (mass[i >> 1] * (y[i] - xn[i]) ** 2) / (2 * h2);

    const { enabled, floorY, k } = this.contact;
    const grad = this._grad;
    const dx = this._dx;
    const rhs = this._rhs;
    const kC = this._kC;
    const diag = this._diag;
    const xt = this._xt;
    const free = this._free;
    const apply = (p, out) => {
      csrMul(rowPtr, colIdx, vals, p, out);
      for (let i = 0; i < n; i++) {
        out[2 * i] += (mass[i] / h2) * p[2 * i];
        out[2 * i + 1] += (mass[i] / h2 + kC[2 * i + 1]) * p[2 * i + 1];
      }
    };

    let cgIters = 0;
    let newtonIters = 0;
    let lineSearchSteps = 0;
    let g0 = this._objective(x, y, h2);
    for (let it = 0; it < this.newton.maxIter; it++) {
      // gradient of g = M (x - y) / h^2 - f_el + contact gradient ; K, kC assembled at x
      this.elasticForce(x, grad, true);
      kC.fill(0);
      for (let i = 0; i < n; i++) {
        grad[2 * i] = (mass[i] * (x[2 * i] - y[2 * i])) / h2 - grad[2 * i];
        grad[2 * i + 1] = (mass[i] * (x[2 * i + 1] - y[2 * i + 1])) / h2 - grad[2 * i + 1];
        if (enabled) {
          const pen = floorY - x[2 * i + 1];
          if (pen > 0) {
            grad[2 * i + 1] -= k * mass[i] * pen;
            kC[2 * i + 1] = k * mass[i];
          }
        }
      }
      for (let i = 0; i < n; i++) {
        diag[2 * i] = mass[i] / h2 + kdiag[2 * i];
        diag[2 * i + 1] = mass[i] / h2 + kdiag[2 * i + 1] + kC[2 * i + 1];
      }
      for (let i = 0; i < N; i++) rhs[i] = free[i] ? -grad[i] : 0;
      dx.fill(0);
      cgIters += pcg({ apply, diag, b: rhs, x: dx, free, tol: this.cg.tol, maxIter: this.cg.maxIter, work: this._work });
      // Newton decrement: -grad . dx  (twice the predicted energy decrease)
      let dec = 0;
      for (let i = 0; i < N; i++) dec -= grad[i] * dx[i];
      if (!(dec > 1e-300) || dec <= this.newton.tol * eRef) break;
      newtonIters++;
      // backtracking line search on g (Armijo)
      let alpha = 1;
      let accepted = false;
      for (let ls = 0; ls < 20; ls++) {
        for (let i = 0; i < N; i++) xt[i] = x[i] + alpha * dx[i];
        const gt = this._objective(xt, y, h2);
        if (gt <= g0 - 1e-4 * alpha * dec) {
          g0 = gt;
          accepted = true;
          break;
        }
        alpha *= 0.5;
        lineSearchSteps++;
      }
      if (!accepted) break;
      x.set(xt);
    }

    for (let i = 0; i < N; i++) v[i] = (x[i] - xn[i]) / h;
    this.stats.iterations = newtonIters;
    this.stats.cgIterations = cgIters;
    this.stats.lineSearch = lineSearchSteps;
    this.stats.stepMs = performance.now() - t0;
  }
}
