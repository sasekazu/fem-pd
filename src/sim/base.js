import { lame, polarRotation } from '../math2d.js';
import { snhEnergy, snhPK1 } from '../materials.js';

export const MODELS = ['corotational', 'neohookean'];

export const DEFAULT_PARAMS = {
  model: 'corotational', // constitutive model: 'corotational' | 'neohookean' (Stable Neo-Hookean)
  E: 5e6, // Young's modulus
  nu: 0.3, // Poisson ratio
  density: 1000, // mass per unit area (thickness = 1)
  gravity: 9.8,
  damping: 1.0, // velocity damping rate [1/s]
};

/**
 * Shared state for both solvers: positions x, velocities v, lumped masses,
 * pinned vertices, external forces and per-vertex strain for visualisation.
 *
 * Subclasses implement `step(h)` and may override `onParamsChanged()`.
 */
export class SoftBodyBase {
  constructor(mesh, params = {}) {
    this.mesh = mesh;
    this.n = mesh.n;
    this.params = { ...DEFAULT_PARAMS, ...params };
    this.x = new Float64Array(mesh.X);
    this.v = new Float64Array(2 * mesh.n);
    this.fext = new Float64Array(2 * mesh.n); // user forces (mouse etc.), not gravity
    this.pinned = new Uint8Array(mesh.n);
    this.pinTarget = new Float64Array(mesh.X); // where pinned vertices must be (prescribed motion)
    this.stats = { stepMs: 0, iterations: 0 };
    // Floor contact: penalty energy  sum_i 1/2 k m_i max(0, floorY - y_i)^2  (k in 1/s^2, mass-scaled)
    this.contact = { enabled: false, floorY: 0, k: 1e5 };
    this.vertexStrain = new Float32Array(mesh.n);
    this.setPinned(mesh.fixedIdx);
    this._updateMaterial();
  }

  _updateMaterial() {
    const { E, nu, density } = this.params;
    const { mu, lambda } = lame(E, nu);
    this.mu = mu;
    this.lambda = lambda;
    // Stable Neo-Hookean parameter: the (J-1)^2 term of Psi_SNH is paired with -mu (J-1), which
    // lowers the rest bulk stiffness by mu. lambda + mu restores it, so that the rest Hessian equals
    // linear elasticity with the Lame parameters (mu, lambda) (2D analogue of Smith et al.'s
    // mu' = 4/3 mu, lambda' = lambda + 5/6 mu).
    this.lambdaNH = lambda + mu;
    // lumped mass: a third of each incident triangle's mass
    const { tris, area } = this.mesh;
    const mass = new Float64Array(this.n);
    for (let e = 0; e < area.length; e++) {
      const m = (density * area[e]) / 3;
      mass[tris[3 * e]] += m;
      mass[tris[3 * e + 1]] += m;
      mass[tris[3 * e + 2]] += m;
    }
    this.mass = mass;
  }

  setParams(p) {
    Object.assign(this.params, p);
    this._updateMaterial();
    this.onParamsChanged();
  }

  onParamsChanged() {}

  setContact(c) {
    Object.assign(this.contact, c);
  }

  /** Floor penalty energy at positions x. */
  contactEnergy(x) {
    const { enabled, floorY, k } = this.contact;
    if (!enabled) return 0;
    let e = 0;
    for (let i = 0; i < this.n; i++) {
      const d = floorY - x[2 * i + 1];
      if (d > 0) e += 0.5 * k * this.mass[i] * d * d;
    }
    return e;
  }

  get isNeoHookean() {
    return this.params.model === 'neohookean';
  }

  /**
   * Exact elastic energy. Corotational: sum_e area (mu ||F-R||^2 + lambda/2 (tr(R^T F) - 2)^2).
   * Stable Neo-Hookean: sum_e area Psi_SNH(F).
   */
  elasticEnergy(x) {
    const { tris, Dinv, area } = this.mesh;
    const { mu } = this;
    const lambda = this.isNeoHookean ? this.lambdaNH : this.lambda;
    const neo = this.isNeoHookean;
    const rot = [1, 0];
    let e = 0;
    for (let t = 0; t < area.length; t++) {
      const i0 = tris[3 * t];
      const i1 = tris[3 * t + 1];
      const i2 = tris[3 * t + 2];
      const d1x = x[2 * i1] - x[2 * i0];
      const d1y = x[2 * i1 + 1] - x[2 * i0 + 1];
      const d2x = x[2 * i2] - x[2 * i0];
      const d2y = x[2 * i2 + 1] - x[2 * i0 + 1];
      const a = Dinv[4 * t];
      const b = Dinv[4 * t + 1];
      const c = Dinv[4 * t + 2];
      const d = Dinv[4 * t + 3];
      const f00 = d1x * a + d2x * c;
      const f01 = d1x * b + d2x * d;
      const f10 = d1y * a + d2y * c;
      const f11 = d1y * b + d2y * d;
      if (neo) {
        e += area[t] * snhEnergy(mu, lambda, f00, f01, f10, f11);
        continue;
      }
      polarRotation(f00, f01, f10, f11, rot);
      const cs = rot[0];
      const sn = rot[1];
      const dev = (f00 - cs) ** 2 + (f01 + sn) ** 2 + (f10 - sn) ** 2 + (f11 - cs) ** 2;
      const tr = cs * (f00 + f11) + sn * (f10 - f01) - 2;
      e += area[t] * (mu * dev + 0.5 * lambda * tr * tr);
    }
    return e;
  }


  /**
   * Exact internal force at positions x (negative energy gradient): f_{a,k} = -area sum_c P_{kc} G_{ca}.
   * Corotational: P = 2 mu (F - R) + lambda (tr(R^T F) - 2) R (R held at its polar value, which is
   * the exact gradient). Stable Neo-Hookean: P = mu F + (lambda (J-1) - mu) cof F. Independent of the solver.
   */
  internalForces(x, out) {
    const { tris, Dinv, area } = this.mesh;
    const { mu } = this;
    const lambda = this.isNeoHookean ? this.lambdaNH : this.lambda;
    const neo = this.isNeoHookean;
    const rot = [1, 0];
    const pk = new Float64Array(4);
    out.fill(0);
    for (let t = 0; t < area.length; t++) {
      const i0 = tris[3 * t];
      const i1 = tris[3 * t + 1];
      const i2 = tris[3 * t + 2];
      const d1x = x[2 * i1] - x[2 * i0];
      const d1y = x[2 * i1 + 1] - x[2 * i0 + 1];
      const d2x = x[2 * i2] - x[2 * i0];
      const d2y = x[2 * i2 + 1] - x[2 * i0 + 1];
      const a = Dinv[4 * t];
      const b = Dinv[4 * t + 1];
      const c = Dinv[4 * t + 2];
      const d = Dinv[4 * t + 3];
      const f00 = d1x * a + d2x * c;
      const f01 = d1x * b + d2x * d;
      const f10 = d1y * a + d2y * c;
      const f11 = d1y * b + d2y * d;
      let p00;
      let p01;
      let p10;
      let p11;
      if (neo) {
        snhPK1(mu, lambda, f00, f01, f10, f11, pk);
        [p00, p01, p10, p11] = pk;
      } else {
        polarRotation(f00, f01, f10, f11, rot);
        const cs = rot[0];
        const sn = rot[1];
        const tr = cs * (f00 + f11) + sn * (f10 - f01) - 2;
        p00 = 2 * mu * (f00 - cs) + lambda * tr * cs;
        p01 = 2 * mu * (f01 + sn) - lambda * tr * sn;
        p10 = 2 * mu * (f10 - sn) + lambda * tr * sn;
        p11 = 2 * mu * (f11 - cs) + lambda * tr * cs;
      }
      // shape-function gradients: g1 = (a, b), g2 = (c, d), g0 = -(g1 + g2)
      const gs = [-(a + c), -(b + d), a, b, c, d];
      const A = area[t];
      const nodes = [i0, i1, i2];
      for (let k = 0; k < 3; k++) {
        const gx = gs[2 * k];
        const gy = gs[2 * k + 1];
        out[2 * nodes[k]] -= A * (p00 * gx + p01 * gy);
        out[2 * nodes[k] + 1] -= A * (p10 * gx + p11 * gy);
      }
    }
    return out;
  }

  /** Pulling force exerted by the body on the given grip vertices along +dir (dir = +1 or -1) in x. */
  gripForce(indices, dir) {
    if (!this._fi) this._fi = new Float64Array(2 * this.n);
    this.internalForces(this.x, this._fi);
    let s = 0;
    for (const i of indices) s -= dir * this._fi[2 * i];
    return s;
  }

  /** Deformation-gradient statistics: smallest det F and number of inverted elements (det F <= 0). */
  deformationStats(x = this.x) {
    const { tris, Dinv, area } = this.mesh;
    let minDet = Infinity;
    let inverted = 0;
    for (let t = 0; t < area.length; t++) {
      const i0 = tris[3 * t];
      const i1 = tris[3 * t + 1];
      const i2 = tris[3 * t + 2];
      const d1x = x[2 * i1] - x[2 * i0];
      const d1y = x[2 * i1 + 1] - x[2 * i0 + 1];
      const d2x = x[2 * i2] - x[2 * i0];
      const d2y = x[2 * i2 + 1] - x[2 * i0 + 1];
      const f00 = d1x * Dinv[4 * t] + d2x * Dinv[4 * t + 2];
      const f01 = d1x * Dinv[4 * t + 1] + d2x * Dinv[4 * t + 3];
      const f10 = d1y * Dinv[4 * t] + d2y * Dinv[4 * t + 2];
      const f11 = d1y * Dinv[4 * t + 1] + d2y * Dinv[4 * t + 3];
      const det = f00 * f11 - f01 * f10;
      if (det < minDet) minDet = det;
      if (det <= 0) inverted++;
    }
    return { minDet, inverted };
  }

  kineticEnergy() {
    let e = 0;
    for (let i = 0; i < this.n; i++) e += 0.5 * this.mass[i] * (this.v[2 * i] ** 2 + this.v[2 * i + 1] ** 2);
    return e;
  }

  gravityEnergy() {
    let e = 0;
    for (let i = 0; i < this.n; i++) e += this.mass[i] * this.params.gravity * this.x[2 * i + 1];
    return e;
  }

  /** Energy breakdown of the current state (total = kinetic + elastic + gravity + contact). */
  energies() {
    const kin = this.kineticEnergy();
    const ela = this.elasticEnergy(this.x);
    const grav = this.gravityEnergy();
    const con = this.contactEnergy(this.x);
    return { kin, ela, grav, con, total: kin + ela + grav + con };
  }

  /** Mass-weighted centre of mass height and vertical velocity. */
  centerOfMass() {
    let m = 0;
    let y = 0;
    let vy = 0;
    for (let i = 0; i < this.n; i++) {
      m += this.mass[i];
      y += this.mass[i] * this.x[2 * i + 1];
      vy += this.mass[i] * this.v[2 * i + 1];
    }
    return { y: y / m, vy: vy / m };
  }

  setPinned(indices) {
    this.pinned.fill(0);
    for (const i of indices) this.pinned[i] = 1;
    this.pinnedList = Array.from(indices);
    this.onPinnedChanged();
  }

  onPinnedChanged() {}

  /** Prescribe the position of a pinned vertex for the next step (kinematic boundary condition). */
  setPinTarget(i, px, py) {
    this.pinTarget[2 * i] = px;
    this.pinTarget[2 * i + 1] = py;
  }

  reset() {
    this.x.set(this.mesh.X);
    this.pinTarget.set(this.mesh.X);
    this.v.fill(0);
    this.fext.fill(0);
  }

  /** Add a user force (world space) on vertex i for the next step(s). */
  addForce(i, fx, fy) {
    this.fext[2 * i] += fx;
    this.fext[2 * i + 1] += fy;
  }

  /** Per-vertex strain measure ||F - R||_F averaged over incident triangles. */
  updateVertexStrain() {
    const { tris, Dinv } = this.mesh;
    const x = this.x;
    const acc = new Float32Array(this.n);
    const cnt = new Uint8Array(this.n);
    const rot = [1, 0];
    for (let e = 0; e < tris.length / 3; e++) {
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
      polarRotation(f00, f01, f10, f11, rot);
      const [cs, sn] = rot;
      const s = Math.hypot(f00 - cs, f01 + sn, f10 - sn, f11 - cs);
      for (const i of [i0, i1, i2]) {
        acc[i] += s;
        cnt[i]++;
      }
    }
    for (let i = 0; i < this.n; i++) this.vertexStrain[i] = cnt[i] ? acc[i] / cnt[i] : 0;
    return this.vertexStrain;
  }

  get tip() {
    const i = this.mesh.tipIdx;
    return [this.x[2 * i], this.x[2 * i + 1]];
  }
}
