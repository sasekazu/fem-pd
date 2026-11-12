import { HalfPlane } from './rigid.js';

/**
 * Penalty contact between the soft body's vertices and rigid colliders (see rigid.js).
 *
 * For every vertex i inside a collider (phi < 0):
 *   energy    E_i = 1/2 k m_i phi^2                       (k in 1/s^2, mass-scaled)
 *   gradient  dE_i/dx_i = k m_i phi n                     (n = grad phi, the outward normal)
 *   Hessian   ~ k m_i n n^T                               (Gauss-Newton: curvature term dropped; PSD)
 * The built-in floor is the half plane  phi = y - floorY  (the previous behaviour), so
 * `setContact({ enabled, floorY, k })` keeps working; extra rigid bodies are added with `bodies`.
 *
 * `collect(x, n)` fills the struct-of-arrays {idx, cid, phi, nx, ny}[0..count) with every active
 * (vertex, collider) pair; the solvers then read energy / forces / Hessian blocks from it.
 */
export class ContactModel {
  constructor() {
    this.k = 1e5;
    this.floorEnabled = false;
    this.floor = new HalfPlane(0, 1, 0);
    this.bodies = []; // additional (kinematic) colliders
    this.count = 0;
    this._grow(64);
    this._q = [0, 0];
  }

  _grow(cap) {
    const keep = (Old, New, len) => {
      const a = new New(cap);
      if (Old) a.set(Old.subarray(0, len));
      return a;
    };
    const c = this.count ?? 0;
    this.idx = keep(this.idx, Int32Array, c);
    this.cid = keep(this.cid, Int32Array, c);
    this.phi = keep(this.phi, Float64Array, c);
    this.nx = keep(this.nx, Float64Array, c);
    this.ny = keep(this.ny, Float64Array, c);
    this._cap = cap;
  }

  /** Accepts {enabled, floorY, k}; `enabled` and `floorY` refer to the built-in floor. */
  set({ enabled, floorY, k } = {}) {
    if (enabled !== undefined) this.floorEnabled = !!enabled;
    if (floorY !== undefined) this.floor.d = floorY;
    if (k !== undefined) this.k = k;
  }

  get floorY() {
    return this.floor.d;
  }

  /** Active colliders: the floor (if enabled) followed by the rigid bodies. */
  colliders() {
    return this.floorEnabled ? [this.floor, ...this.bodies] : this.bodies;
  }

  /** Is there anything to collide with? */
  get enabled() {
    return this.floorEnabled || this.bodies.length > 0;
  }

  /** Collect the active pairs of the n vertices at positions x (interleaved xy). Returns the count. */
  collect(x, n) {
    this.count = 0;
    if (!this.enabled) return 0;
    const list = this.colliders();
    const q = this._q;
    for (let c = 0; c < list.length; c++) {
      const col = list[c];
      const b = col.bounds();
      for (let i = 0; i < n; i++) {
        const px = x[2 * i];
        const py = x[2 * i + 1];
        if (b !== null && (px < b[0] || px > b[2] || py < b[1] || py > b[3])) continue; // broad phase
        const phi = col.query(px, py, q);
        if (!(phi < 0)) continue;
        if (this.count === this._cap) this._grow(2 * this._cap);
        const t = this.count++;
        this.idx[t] = i;
        this.cid[t] = c;
        this.phi[t] = phi;
        this.nx[t] = q[0];
        this.ny[t] = q[1];
      }
    }
    return this.count;
  }

  /** Penalty energy at positions x. */
  energy(x, mass, n) {
    const cnt = this.collect(x, n);
    let e = 0;
    for (let t = 0; t < cnt; t++) e += 0.5 * this.k * mass[this.idx[t]] * this.phi[t] ** 2;
    return e;
  }

  /** Copy of the current active (vertex, collider) pairs, for `sameActiveSet` after the next collect. */
  snapshot() {
    return { count: this.count, idx: this.idx.slice(0, this.count), cid: this.cid.slice(0, this.count) };
  }

  /** Does the current list hold exactly the pairs of the snapshot? */
  sameActiveSet(snap) {
    if (this.count !== snap.count) return false;
    for (let t = 0; t < this.count; t++) if (this.idx[t] !== snap.idx[t] || this.cid[t] !== snap.cid[t]) return false;
    return true;
  }

  /**
   * Net contact force exerted on the soft body by each collider (floor first when enabled), at positions x:
   * [{ fx, fy, count }]. (The force on the rigid body is the opposite.)
   */
  colliderForces(x, mass, n) {
    const cnt = this.collect(x, n);
    const list = this.colliders();
    const out = list.map(() => ({ fx: 0, fy: 0, count: 0 }));
    for (let t = 0; t < cnt; t++) {
      const f = -this.k * mass[this.idx[t]] * this.phi[t];
      const o = out[this.cid[t]];
      o.fx += f * this.nx[t];
      o.fy += f * this.ny[t];
      o.count++;
    }
    return out;
  }

  /** Deepest penetration (>= 0) of any vertex into the collider with list index c. */
  maxPenetration(x, n, c = null) {
    const cnt = this.collect(x, n);
    let d = 0;
    for (let t = 0; t < cnt; t++) if ((c === null || this.cid[t] === c) && -this.phi[t] > d) d = -this.phi[t];
    return d;
  }
}
