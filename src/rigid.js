// Rigid colliders described by a signed distance function (SDF), in world space.
//
// A collider answers  query(px, py, out) -> phi  with the signed distance phi of the point to the
// surface (phi < 0: inside the rigid body) and writes the unit gradient (outward normal) to
// out[0], out[1]. Colliders are kinematic: their pose is set from outside (setPose) and is
// constant during a solver step. No DOM / three.js dependency.
//
//   bounds()  world-space AABB [minx, miny, maxx, maxy] of the region where phi <= 0, or null if unbounded.

/** Half plane {p : n.p - d < 0}; the floor is  n = (0, 1), d = floorY. */
export class HalfPlane {
  constructor(nx = 0, ny = 1, d = 0) {
    const len = Math.hypot(nx, ny);
    this.nx = nx / len;
    this.ny = ny / len;
    this.d = d / len;
  }

  bounds() {
    return null;
  }

  query(px, py, out) {
    out[0] = this.nx;
    out[1] = this.ny;
    return this.nx * px + this.ny * py - this.d;
  }
}

/**
 * Simple polygon (convex or concave, counter-clockwise after construction) with a rigid pose.
 * `vertices` is a flat [x0, y0, x1, y1, ...] list in the body frame; the world pose is
 * p_world = R(angle) p_body + (x, y).
 *
 * |phi| is the distance to the nearest edge; the sign comes from an even-odd ray crossing test, so
 * non-convex shapes work. Inside points have grad(phi) pointing to the nearest surface point; this is
 * discontinuous on the medial axis (deep inside), which the penalty method never needs.
 */
export class Polygon {
  constructor(vertices, pose = {}) {
    const v = Float64Array.from(vertices);
    if (v.length < 6 || v.length % 2) throw new Error('Polygon needs at least 3 vertices');
    // signed area -> enforce CCW
    let a2 = 0;
    const m = v.length / 2;
    for (let i = 0; i < m; i++) {
      const j = (i + 1) % m;
      a2 += v[2 * i] * v[2 * j + 1] - v[2 * j] * v[2 * i + 1];
    }
    if (a2 === 0) throw new Error('Polygon has zero area');
    if (a2 < 0) {
      for (let i = 0; i < Math.floor(m / 2); i++) {
        const j = m - 1 - i;
        for (let k = 0; k < 2; k++) [v[2 * i + k], v[2 * j + k]] = [v[2 * j + k], v[2 * i + k]];
      }
    }
    this.vertices = v; // body frame, CCW
    this.n = m;
    this.area = Math.abs(a2) / 2;
    this._world = new Float64Array(2 * m);
    this._bounds = [0, 0, 0, 0];
    this.setPose(pose.x ?? 0, pose.y ?? 0, pose.angle ?? 0);
  }

  /** Move the body to position (x, y) with rotation `angle` [rad] (kinematic). */
  setPose(x, y, angle = this.angle) {
    this.x = x;
    this.y = y;
    this.angle = angle;
    this._c = Math.cos(angle);
    this._s = Math.sin(angle);
    const b = this._bounds;
    b[0] = b[1] = Infinity;
    b[2] = b[3] = -Infinity;
    for (let i = 0; i < this.n; i++) {
      const bx = this.vertices[2 * i];
      const by = this.vertices[2 * i + 1];
      const wx = this._c * bx - this._s * by + x;
      const wy = this._s * bx + this._c * by + y;
      this._world[2 * i] = wx;
      this._world[2 * i + 1] = wy;
      if (wx < b[0]) b[0] = wx;
      if (wx > b[2]) b[2] = wx;
      if (wy < b[1]) b[1] = wy;
      if (wy > b[3]) b[3] = wy;
    }
    return this;
  }

  /** Vertices in world space (flat, CCW). */
  get worldVertices() {
    return this._world;
  }

  bounds() {
    return this._bounds;
  }

  query(px, py, out) {
    // world -> body frame
    const dx = px - this.x;
    const dy = py - this.y;
    const qx = this._c * dx + this._s * dy;
    const qy = -this._s * dx + this._c * dy;
    const v = this.vertices;
    const m = this.n;
    let best = Infinity;
    let bx = 0; // closest surface point minus q
    let by = 0;
    let ex = 0; // outward normal of the closest edge (fallback on the surface)
    let ey = 0;
    let inside = false;
    for (let i = 0; i < m; i++) {
      const j = i + 1 === m ? 0 : i + 1;
      const ax = v[2 * i];
      const ay = v[2 * i + 1];
      const tx = v[2 * j] - ax;
      const ty = v[2 * j + 1] - ay;
      // even-odd crossing of the ray towards +x
      if (ay > qy !== v[2 * j + 1] > qy) {
        const xc = ax + ((qy - ay) / ty) * tx;
        if (xc > qx) inside = !inside;
      }
      const l2 = tx * tx + ty * ty;
      let s = l2 > 0 ? ((qx - ax) * tx + (qy - ay) * ty) / l2 : 0;
      s = s < 0 ? 0 : s > 1 ? 1 : s;
      const rx = ax + s * tx - qx;
      const ry = ay + s * ty - qy;
      const d2 = rx * rx + ry * ry;
      if (d2 < best) {
        best = d2;
        bx = rx;
        by = ry;
        const len = Math.sqrt(l2);
        ex = ty / len;
        ey = -tx / len;
      }
    }
    const dist = Math.sqrt(best);
    let gx;
    let gy;
    if (dist < 1e-12) {
      gx = ex;
      gy = ey;
    } else if (inside) {
      gx = bx / dist; // towards the surface
      gy = by / dist;
    } else {
      gx = -bx / dist; // away from the surface
      gy = -by / dist;
    }
    // body frame -> world
    out[0] = this._c * gx - this._s * gy;
    out[1] = this._s * gx + this._c * gy;
    return inside ? -dist : dist;
  }

  /** Is the world point inside the polygon? */
  contains(px, py) {
    return this.query(px, py, [0, 0]) < 0;
  }
}

// ---- shape helpers (body frame centred on the origin) ----

/** Axis-aligned box of width w and height h centred at the origin. */
export function boxVertices(w, h) {
  return [-w / 2, -h / 2, w / 2, -h / 2, w / 2, h / 2, -w / 2, h / 2];
}

/** Regular n-gon of circumradius r; the first vertex points along `phase` [rad]. */
export function regularVertices(n, r, phase = 0) {
  const out = [];
  for (let i = 0; i < n; i++) out.push(r * Math.cos(phase + (2 * Math.PI * i) / n), r * Math.sin(phase + (2 * Math.PI * i) / n));
  return out;
}

/** Star with `points` tips (outer radius ro, inner radius ri); a tip points along `phase` [rad]. */
export function starVertices(points, ro, ri, phase = 0) {
  const out = [];
  for (let i = 0; i < 2 * points; i++) {
    const r = i % 2 === 0 ? ro : ri;
    const a = phase + (Math.PI * i) / points;
    out.push(r * Math.cos(a), r * Math.sin(a));
  }
  return out;
}

/** L-shaped (concave) polygon: a w x h box with the corner notch (nw x nh) removed at the upper right. */
export function lShapeVertices(w, h, nw, nh) {
  return [-w / 2, -h / 2, w / 2, -h / 2, w / 2, h / 2 - nh, w / 2 - nw, h / 2 - nh, w / 2 - nw, h / 2, -w / 2, h / 2];
}
