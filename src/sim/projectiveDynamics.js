import { SparseCholesky } from '../sparse/cholesky.js';
import { SoftBodyBase } from './base.js';
import { polarRotation } from '../math2d.js';

/**
 * Projective Dynamics (Bouaziz et al. 2014), local/global solver.
 *
 * Per triangle two strain terms share the deformation gradient F = Ds Dm^-1 (R = polar rotation of F):
 *   ARAP     energy (w_a/2)||F - R||^2                  w_a = 2 mu      target R
 *   volume   energy (w_v/2)||F - P_v||^2                w_v = 2 lambda  target P_v = F - ((tr(R^T F) - 2)/2) R
 * Both are weighted by the rest area and together equal the corotational energy
 * mu||F-R||^2 + lambda/2 (tr(R^T F) - 2)^2 for fixed R, so the fixed points of the
 * local/global iteration coincide with the FEM equilibrium for any lambda.
 * Since both terms act on the same F, the local step uses the weighted average target.
 *
 * Global step: solve  (M/h^2 + sum_i w_i area_i G_i^T G_i) x = M y / h^2 + sum_i w_i area_i G_i^T P_i
 * (x and y components decouple with the same scalar matrix) using a sparse
 * Cholesky factor (RCM ordering, see sparse/cholesky.js) computed once for the
 * free (non-pinned) vertices.
 */
export class ProjectiveDynamics extends SoftBodyBase {
  constructor(mesh, params = {}) {
    super(mesh, params);
    this.iterations = 10;
    this._h = null; // step size the factorization was built for
    this._dirty = true;
    const n = mesh.n;
    this._y = new Float64Array(2 * n);
    this._xn = new Float64Array(2 * n);
    this._rhs = new Float64Array(2 * n);
    this._xt = new Float64Array(2 * n);
    this._tg = [1, 0, 0, 1];
    this._rot = [1, 0];
    this.lbfgsM = 6; // L-BFGS history used when floor contact is enabled (0 = plain PD + line search)
    this._G = this._buildGradients();
  }

  onParamsChanged() {
    this._dirty = true;
  }

  onPinnedChanged() {
    this._dirty = true;
  }

  /** Shape-function gradients per triangle: g[j] = (dN_j/dX0, dN_j/dX1). */
  _buildGradients() {
    const { Dinv, area } = this.mesh;
    const m = area.length;
    const G = new Float64Array(6 * m);
    for (let e = 0; e < m; e++) {
      const a = Dinv[4 * e];
      const b = Dinv[4 * e + 1];
      const c = Dinv[4 * e + 2];
      const d = Dinv[4 * e + 3];
      G[6 * e] = -(a + c);
      G[6 * e + 1] = -(b + d);
      G[6 * e + 2] = a;
      G[6 * e + 3] = b;
      G[6 * e + 4] = c;
      G[6 * e + 5] = d;
    }
    return G;
  }

  /** (Re)build the global matrix and its Cholesky factor for step size h. */
  _factorize(h) {
    const { tris, area, n } = this.mesh;
    const m = area.length;
    const wa = 2 * this.mu;
    const wv = 2 * this.lambda;
    const w = wa + wv;
    // free-vertex numbering
    const fidx = new Int32Array(n).fill(-1);
    let nf = 0;
    for (let i = 0; i < n; i++) if (!this.pinned[i]) fidx[i] = nf++;
    this._fidx = fidx;
    this._nf = nf;

    // assemble the global matrix as symmetric triplets over the free vertices
    const ti = [];
    const tj = [];
    const tv = [];
    for (let i = 0; i < n; i++) {
      if (fidx[i] >= 0) {
        ti.push(fidx[i]);
        tj.push(fidx[i]);
        tv.push(this.mass[i] / (h * h));
      }
    }
    // Contribution of pinned vertices to the right-hand side: -L_{free,pinned} x_pinned
    // (recomputed every step from the prescribed pin targets: see _updateBFixed)
    const cFa = [];
    const cIb = [];
    const cL = [];
    for (let e = 0; e < m; e++) {
      const g = this._G;
      const o = 6 * e;
      for (let a = 0; a < 3; a++) {
        const ia = tris[3 * e + a];
        const fa = fidx[ia];
        if (fa < 0) continue;
        for (let b = 0; b < 3; b++) {
          const ib = tris[3 * e + b];
          const lab = w * area[e] * (g[o + 2 * a] * g[o + 2 * b] + g[o + 2 * a + 1] * g[o + 2 * b + 1]);
          const fb = fidx[ib];
          if (fb >= 0) {
            ti.push(fa);
            tj.push(fb);
            tv.push(lab);
          } else {
            cFa.push(fa);
            cIb.push(ib);
            cL.push(lab);
          }
        }
      }
    }
    this._chol = new SparseCholesky(nf, Int32Array.from(ti), Int32Array.from(tj), Float64Array.from(tv));
    this._cFa = Int32Array.from(cFa);
    this._cIb = Int32Array.from(cIb);
    this._cL = Float64Array.from(cL);
    this._bFixed = new Float64Array(2 * nf);
    this._zx = new Float64Array(nf);
    this._zy = new Float64Array(nf);
    this._bx = new Float64Array(nf);
    this._by = new Float64Array(nf);
    this._h = h;
    this._dirty = false;
  }

  /** bFixed = -L_{free,pinned} x_pinned for the current pin targets. */
  _updateBFixed() {
    const b = this._bFixed;
    b.fill(0);
    const t = this.pinTarget;
    for (let k = 0; k < this._cL.length; k++) {
      const fa = this._cFa[k];
      const ib = this._cIb[k];
      b[2 * fa] -= this._cL[k] * t[2 * ib];
      b[2 * fa + 1] -= this._cL[k] * t[2 * ib + 1];
    }
  }

  /** Local step for one element: weighted target of the ARAP and volume terms (row-major 2x2). */
  _target(f00, f01, f10, f11, out) {
    const wa = 2 * this.mu;
    const wv = 2 * this.lambda;
    const rot = this._rot;
    polarRotation(f00, f01, f10, f11, rot);
    const c = rot[0];
    const s = rot[1];
    let p00 = c;
    let p01 = -s;
    let p10 = s;
    let p11 = c;
    if (wv > 0) {
      // volumetric term (lambda/2)(tr(R^T F) - 2)^2: with R fixed it is quadratic along R.
      // P_v = F - ((tr(R^T F) - 2) / 2) R is bounded and smooth (no 1/sqrt(det) blow-up)
      // and with weight w_v = 2 lambda reproduces the energy exactly.
      const t = 0.5 * (c * (f00 + f11) + s * (f10 - f01) - 2);
      const rv = wv / (wa + wv);
      p00 += rv * (f00 - t * c - p00);
      p01 += rv * (f01 + t * s - p01);
      p10 += rv * (f10 - t * s - p10);
      p11 += rv * (f11 - t * c - p11);
    }
    out[0] = p00;
    out[1] = p01;
    out[2] = p10;
    out[3] = p11;
    return out;
  }

  /**
   * Gradient of the implicit-Euler objective at x over the free DOFs, layout [x-block(nf), y-block(nf)]:
   *   M (x - y)/h^2 + sum_e area w G^T (F - P) + contact gradient.
   */
  _gradient(x, y, h2, g) {
    const { tris, area, n } = this.mesh;
    const { mass } = this;
    const nf = this._nf;
    const fidx = this._fidx;
    const G = this._G;
    const w = 2 * this.mu + 2 * this.lambda;
    const tg = this._tg;
    const { enabled, floorY, k } = this.contact;
    g.fill(0);
    for (let i = 0; i < n; i++) {
      const f = fidx[i];
      if (f < 0) continue;
      g[f] += (mass[i] * (x[2 * i] - y[2 * i])) / h2;
      g[nf + f] += (mass[i] * (x[2 * i + 1] - y[2 * i + 1])) / h2;
      if (enabled) {
        const d = floorY - x[2 * i + 1];
        if (d > 0) g[nf + f] -= k * mass[i] * d;
      }
    }
    for (let e = 0; e < area.length; e++) {
      const i0 = tris[3 * e];
      const i1 = tris[3 * e + 1];
      const i2 = tris[3 * e + 2];
      const d1x = x[2 * i1] - x[2 * i0];
      const d1y = x[2 * i1 + 1] - x[2 * i0 + 1];
      const d2x = x[2 * i2] - x[2 * i0];
      const d2y = x[2 * i2 + 1] - x[2 * i0 + 1];
      const o = 6 * e;
      const f00 = d1x * G[o + 2] + d2x * G[o + 4];
      const f01 = d1x * G[o + 3] + d2x * G[o + 5];
      const f10 = d1y * G[o + 2] + d2y * G[o + 4];
      const f11 = d1y * G[o + 3] + d2y * G[o + 5];
      this._target(f00, f01, f10, f11, tg);
      const we = w * area[e];
      const r00 = f00 - tg[0];
      const r01 = f01 - tg[1];
      const r10 = f10 - tg[2];
      const r11 = f11 - tg[3];
      for (let a = 0; a < 3; a++) {
        const fa = fidx[tris[3 * e + a]];
        if (fa < 0) continue;
        g[fa] += we * (G[o + 2 * a] * r00 + G[o + 2 * a + 1] * r01);
        g[nf + fa] += we * (G[o + 2 * a] * r10 + G[o + 2 * a + 1] * r11);
      }
    }
    return g;
  }

  /** r = A^-1 q for a vector in the [x-block, y-block] layout. */
  _applyH0(q, r) {
    const nf = this._nf;
    this._chol.solve(q.subarray(0, nf), r.subarray(0, nf));
    this._chol.solve(q.subarray(nf), r.subarray(nf));
  }

  /**
   * Quasi-Newton minimisation of the implicit-Euler objective (Liu, Bouaziz & Kavan 2016):
   * the constant PD matrix A = M/h^2 + L is the initial inverse-Hessian of L-BFGS, with an
   * Armijo backtracking line search. Floor contact only appears in the objective/gradient.
   * Returns the number of line-search backtracks.
   */
  _solveLBFGS(x, y, h2, eRef) {
    const { n } = this;
    const nf = this._nf;
    const fidx = this._fidx;
    const m = this.lbfgsM;
    const dim = 2 * nf;
    if (!this._lb || this._lb.dim !== dim || this._lb.m !== m) {
      const mk = () => new Float64Array(dim);
      this._lb = {
        dim,
        m,
        S: Array.from({ length: m }, mk),
        Y: Array.from({ length: m }, mk),
        rho: new Float64Array(m),
        alpha: new Float64Array(m),
        g: mk(),
        gN: mk(),
        d: mk(),
        q: mk(),
        r: mk(),
      };
    }
    const lb = this._lb;
    const { S, Y, rho, alpha: al, g, gN, d, q, r } = lb;
    let count = 0; // stored pairs
    let head = 0; // next slot
    const xt = this._xt;
    let backtracks = 0;
    this._gradient(x, y, h2, g);
    let E = this._objective(x, y, h2);
    const dot = (a, b) => {
      let s = 0;
      for (let i = 0; i < dim; i++) s += a[i] * b[i];
      return s;
    };

    for (let it = 0; it < this.iterations; it++) {
      // two-loop recursion: d = -H g
      q.set(g);
      for (let c = 0; c < count; c++) {
        const idx = (head - 1 - c + m) % m;
        al[idx] = rho[idx] * dot(S[idx], q);
        const Yi = Y[idx];
        for (let i = 0; i < dim; i++) q[i] -= al[idx] * Yi[i];
      }
      this._applyH0(q, r);
      for (let c = count - 1; c >= 0; c--) {
        const idx = (head - 1 - c + m) % m;
        const beta = rho[idx] * dot(Y[idx], r);
        const Si = S[idx];
        for (let i = 0; i < dim; i++) r[i] += Si[i] * (al[idx] - beta);
      }
      for (let i = 0; i < dim; i++) d[i] = -r[i];
      let gd = dot(g, d);
      if (!(gd < 0)) {
        // not a descent direction: fall back to the plain PD direction and drop the history
        count = 0;
        head = 0;
        this._applyH0(g, r);
        for (let i = 0; i < dim; i++) d[i] = -r[i];
        gd = dot(g, d);
        if (!(gd < 0)) break; // converged
      }
      // converged: the Newton decrement is negligible against the energy scale of the step
      if (-gd < 1e-10 * eRef) break;
      // Armijo backtracking
      let a = 1;
      let Et = E;
      for (let ls = 0; ls < 20; ls++) {
        for (let i = 0; i < n; i++) {
          const f = fidx[i];
          if (f < 0) {
            xt[2 * i] = x[2 * i];
            xt[2 * i + 1] = x[2 * i + 1];
          } else {
            xt[2 * i] = x[2 * i] + a * d[f];
            xt[2 * i + 1] = x[2 * i + 1] + a * d[nf + f];
          }
        }
        Et = this._objective(xt, y, h2);
        if (Et <= E + 1e-4 * a * gd + 1e-13 * eRef) break;
        a *= 0.5;
        backtracks++;
      }
      this._gradient(xt, y, h2, gN);
      // history update
      const Si = S[head];
      const Yi = Y[head];
      let sy = 0;
      for (let i = 0; i < dim; i++) {
        Si[i] = a * d[i];
        Yi[i] = gN[i] - g[i];
        sy += Si[i] * Yi[i];
      }
      if (sy > 1e-14) {
        rho[head] = 1 / sy;
        head = (head + 1) % m;
        count = Math.min(count + 1, m);
      }
      x.set(xt);
      g.set(gN);
      E = Et;
    }
    return backtracks;
  }

  /** Implicit-Euler objective: inertia + corotational elasticity + floor penalty. */
  _objective(x, y, h2) {
    const { mass, n } = this;
    let e = 0;
    for (let i = 0; i < n; i++) {
      const dx = x[2 * i] - y[2 * i];
      const dy = x[2 * i + 1] - y[2 * i + 1];
      e += (0.5 * mass[i] * (dx * dx + dy * dy)) / h2;
    }
    return e + this.elasticEnergy(x) + this.contactEnergy(x);
  }

  step(h) {
    const t0 = performance.now();
    if (this._dirty || this._h !== h) this._factorize(h);
    this._updateBFixed();
    const { x, v, mass, n } = this;
    const { tris, area } = this.mesh;
    const m = area.length;
    const { gravity, damping } = this.params;
    const N = 2 * n;
    const wa = 2 * this.mu;
    const wv = 2 * this.lambda;
    const w = wa + wv;
    const G = this._G;
    const fidx = this._fidx;
    const h2 = h * h;

    const damp = 1 / (1 + h * damping);
    for (let i = 0; i < N; i++) v[i] *= damp;

    const xn = this._xn;
    xn.set(x);
    // inertial target y = x_n + h v + h^2 (g + f_ext / m)
    const y = this._y;
    for (let i = 0; i < n; i++) {
      y[2 * i] = x[2 * i] + h * v[2 * i] + h2 * (this.fext[2 * i] / mass[i]);
      y[2 * i + 1] = x[2 * i + 1] + h * v[2 * i + 1] + h2 * (this.fext[2 * i + 1] / mass[i] - gravity);
    }
    // initial guess: y (pinned vertices stay at rest)
    for (let i = 0; i < n; i++) {
      if (this.pinned[i]) {
        x[2 * i] = this.pinTarget[2 * i];
        x[2 * i + 1] = this.pinTarget[2 * i + 1];
      } else {
        x[2 * i] = y[2 * i];
        x[2 * i + 1] = y[2 * i + 1];
      }
    }

    const { enabled: contactOn, floorY, k: kc } = this.contact;
    let lineSearchSteps = 0;
    const useLBFGS = contactOn && this.lbfgsM > 0;
    if (useLBFGS) {
      // energy scale of the step (objective at x_n): used for convergence / round-off tolerances
      let eRef = 0;
      for (let i = 0; i < N; i++) eRef += (mass[i >> 1] * (y[i] - xn[i]) ** 2) / (2 * h2);
      lineSearchSteps = this._solveLBFGS(x, y, h2, eRef);
    }
    const tg = this._tg;
    const bx = this._bx;
    const by = this._by;
    for (let it = 0; it < (useLBFGS ? 0 : this.iterations); it++) {
      // ---- global RHS: M y / h^2 + pinned contribution ----
      for (let i = 0; i < n; i++) {
        const f = fidx[i];
        if (f < 0) continue;
        bx[f] = (mass[i] * y[2 * i]) / h2 + this._bFixed[2 * f];
        by[f] = (mass[i] * y[2 * i + 1]) / h2 + this._bFixed[2 * f + 1];
      }
      // ---- local step: project every triangle, scatter w area G^T P ----
      for (let e = 0; e < m; e++) {
        const i0 = tris[3 * e];
        const i1 = tris[3 * e + 1];
        const i2 = tris[3 * e + 2];
        const d1x = x[2 * i1] - x[2 * i0];
        const d1y = x[2 * i1 + 1] - x[2 * i0 + 1];
        const d2x = x[2 * i2] - x[2 * i0];
        const d2y = x[2 * i2 + 1] - x[2 * i0 + 1];
        const o = 6 * e;
        // F = Ds Dm^-1 ; rows of Dm^-1 are g1, g2
        const f00 = d1x * G[o + 2] + d2x * G[o + 4];
        const f01 = d1x * G[o + 3] + d2x * G[o + 5];
        const f10 = d1y * G[o + 2] + d2y * G[o + 4];
        const f11 = d1y * G[o + 3] + d2y * G[o + 5];
        this._target(f00, f01, f10, f11, tg);
        const p00 = tg[0];
        const p01 = tg[1];
        const p10 = tg[2];
        const p11 = tg[3];
        const we = w * area[e];
        for (let a = 0; a < 3; a++) {
          const fa = fidx[tris[3 * e + a]];
          if (fa < 0) continue;
          // sum_c G_{c a} P_{k c}
          bx[fa] += we * (G[o + 2 * a] * p00 + G[o + 2 * a + 1] * p01);
          by[fa] += we * (G[o + 2 * a] * p10 + G[o + 2 * a + 1] * p11);
        }
      }
      // ---- floor contact (Liu, Bouaziz & Kavan 2016): the penalty energy enters the gradient
      //      (-> right-hand side) but NOT the constant matrix, so no re-factorisation is needed.
      let anyActive = false;
      if (contactOn) {
        for (let i = 0; i < n; i++) {
          const f = fidx[i];
          if (f < 0) continue;
          const d = floorY - x[2 * i + 1];
          if (d > 0) {
            by[f] += kc * mass[i] * d;
            anyActive = true;
          }
        }
      }
      // ---- global step ----
      this._chol.solve(bx, this._zx);
      this._chol.solve(by, this._zy);
      let alpha = 1;
      if (contactOn) {
        let trialActive = false;
        for (let i = 0; i < n; i++) {
          const f = fidx[i];
          if (f >= 0 && floorY - this._zy[f] > 0) trialActive = true;
        }
        if (anyActive || trialActive) {
          // backtracking line search on the full objective (inertia + elasticity + contact):
          // the inexact Hessian (no contact stiffness) can overshoot.
          const xt = this._xt;
          const E0 = this._objective(x, y, h2);
          for (let ls = 0; ls < 12; ls++) {
            for (let i = 0; i < n; i++) {
              const f = fidx[i];
              if (f < 0) {
                xt[2 * i] = x[2 * i];
                xt[2 * i + 1] = x[2 * i + 1];
              } else {
                xt[2 * i] = x[2 * i] + alpha * (this._zx[f] - x[2 * i]);
                xt[2 * i + 1] = x[2 * i + 1] + alpha * (this._zy[f] - x[2 * i + 1]);
              }
            }
            if (this._objective(xt, y, h2) <= E0 + 1e-12 * Math.abs(E0)) break;
            alpha *= 0.5;
            lineSearchSteps++;
          }
          x.set(xt);
          continue;
        }
      }
      for (let i = 0; i < n; i++) {
        const f = fidx[i];
        if (f < 0) continue;
        x[2 * i] = this._zx[f];
        x[2 * i + 1] = this._zy[f];
      }
    }

    for (let i = 0; i < N; i++) v[i] = (x[i] - xn[i]) / h;
    this.stats.iterations = this.iterations;
    this.stats.lineSearch = lineSearchSteps;
    this.stats.stepMs = performance.now() - t0;
  }
}
