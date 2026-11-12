// Mouse interaction: grab a point of the body and drag it with a spring (pure logic, no three.js / DOM).
//
// The grabbed material point is stored as (triangle, barycentric weights), so the very same point and the
// very same world-space target can be applied to both solvers (they share the mesh topology).
//
// Force model (mass-scaled spring-damper, applied through SoftBodyBase.fext):
//   p = sum_j w_j x_j,   a = k (target - p) - c v_p,   f_j = m_j w_j a
// Acceleration of vertex j is at most |a| (independent of mesh resolution and of the body's mass), and the
// coupled system has a single stiff eigenvalue k |w|^2 <= k. The force is evaluated explicitly (it enters the
// right-hand side as an external force), so k must satisfy k h^2 < 4 for stability; `effectiveStiffness` caps it.

/**
 * Map a viewport position to world coordinates for an orthographic camera.
 * u, v in [0, 1] with (0, 0) at the top-left corner of the viewport.
 */
export function viewportToWorld(cam, u, v) {
  return [cam.left + u * (cam.right - cam.left), cam.top - v * (cam.top - cam.bottom)];
}

/**
 * Find the triangle of `mesh` (deformed vertex positions `x`) that contains (px, py).
 * Returns { tri, w: [w0, w1, w2] } with weights clamped to the triangle (sum 1), or null if the point is more
 * than `tol` (in barycentric units) outside every triangle. The most "inside" triangle wins on ties.
 */
export function locateTriangle(mesh, x, px, py, tol = 0.1) {
  const { tris } = mesh;
  let best = -1;
  let bestMin = -Infinity;
  let bw = null;
  for (let t = 0; t < tris.length / 3; t++) {
    const i0 = tris[3 * t];
    const i1 = tris[3 * t + 1];
    const i2 = tris[3 * t + 2];
    const ax = x[2 * i0];
    const ay = x[2 * i0 + 1];
    const e1x = x[2 * i1] - ax;
    const e1y = x[2 * i1 + 1] - ay;
    const e2x = x[2 * i2] - ax;
    const e2y = x[2 * i2 + 1] - ay;
    const det = e1x * e2y - e2x * e1y;
    if (!Number.isFinite(det) || Math.abs(det) < 1e-300) continue;
    const dx = px - ax;
    const dy = py - ay;
    const w1 = (dx * e2y - e2x * dy) / det;
    const w2 = (e1x * dy - dx * e1y) / det;
    const w0 = 1 - w1 - w2;
    const mn = Math.min(w0, w1, w2);
    if (mn > bestMin) {
      bestMin = mn;
      best = t;
      bw = [w0, w1, w2];
    }
  }
  if (best < 0 || bestMin < -tol) return null;
  const w = bw.map((a) => Math.max(0, a));
  const s = w[0] + w[1] + w[2];
  return { tri: best, w: w.map((a) => a / s) };
}

/** Largest stable mouse stiffness for sub-step size h (explicit force: k h^2 < 4, with some margin). */
export function effectiveStiffness(k, h) {
  return Math.min(k, 1 / (h * h));
}

export class MouseGrabber {
  /**
   * @param {object} opts
   * @param {number} [opts.stiffness]   spring stiffness k [1/s^2]
   * @param {number} [opts.damping]     damping ratio of the spring (1 = critical)
   * @param {number} [opts.maxDist]     cap on the |target - p| used for the force [m]
   */
  constructor({ stiffness = 1500, damping = 0.7, maxDist = 1.5 } = {}) {
    this.stiffness = stiffness;
    this.damping = damping;
    this.maxDist = maxDist;
    this.active = false;
    this.tri = -1;
    this.w = [0, 0, 0];
    this.target = [0, 0];
  }

  /** Grab the body of `sim` at world position (px, py). Returns false if nothing is there. */
  grab(sim, px, py) {
    const hit = locateTriangle(sim.mesh, sim.x, px, py);
    if (!hit) return false;
    this.active = true;
    this.tri = hit.tri;
    this.w = hit.w;
    this.target = [px, py];
    return true;
  }

  setTarget(px, py) {
    this.target = [px, py];
  }

  release() {
    this.active = false;
  }

  /** Current world position of the grabbed material point in `sim`. */
  point(sim) {
    const { tris } = sim.mesh;
    let px = 0;
    let py = 0;
    for (let k = 0; k < 3; k++) {
      const i = tris[3 * this.tri + k];
      px += this.w[k] * sim.x[2 * i];
      py += this.w[k] * sim.x[2 * i + 1];
    }
    return [px, py];
  }

  /**
   * Write the mouse force into `sim.fext` for the next step of size h. The grabber owns `fext`:
   * it is cleared here, so releasing the mouse removes the force.
   */
  applyForces(sim, h) {
    sim.fext.fill(0);
    if (!this.active) return;
    const { tris } = sim.mesh;
    const k = effectiveStiffness(this.stiffness, h);
    const c = 2 * this.damping * Math.sqrt(k);
    let px = 0;
    let py = 0;
    let vx = 0;
    let vy = 0;
    for (let a = 0; a < 3; a++) {
      const i = tris[3 * this.tri + a];
      px += this.w[a] * sim.x[2 * i];
      py += this.w[a] * sim.x[2 * i + 1];
      vx += this.w[a] * sim.v[2 * i];
      vy += this.w[a] * sim.v[2 * i + 1];
    }
    let dx = this.target[0] - px;
    let dy = this.target[1] - py;
    const dist = Math.hypot(dx, dy);
    if (dist > this.maxDist) {
      dx *= this.maxDist / dist;
      dy *= this.maxDist / dist;
    }
    const ax = k * dx - c * vx;
    const ay = k * dy - c * vy;
    for (let a = 0; a < 3; a++) {
      const i = tris[3 * this.tri + a];
      const s = sim.mass[i] * this.w[a];
      sim.addForce(i, s * ax, s * ay);
    }
  }
}
