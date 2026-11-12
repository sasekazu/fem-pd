// Small 2D helpers. Matrices are passed as scalars (row-major: a b / c d).

/**
 * Closed-form 2x2 polar decomposition F = R S (R: rotation, S: symmetric).
 * Returns the rotation as [cos, sin]; R = [[cos, -sin], [sin, cos]].
 * Always yields a proper rotation (det R = +1), even for inverted F.
 */
export function polarRotation(a, b, c, d, out) {
  const x = a + d;
  const y = c - b;
  const len = Math.hypot(x, y);
  if (len < 1e-12) {
    out[0] = 1;
    out[1] = 0;
  } else {
    out[0] = x / len;
    out[1] = y / len;
  }
  return out;
}

/**
 * Signed 2x2 SVD  F = U diag(s0, s1) V^T  with U and V proper rotations (det = +1).
 * s0 >= |s1| and s1 < 0 exactly when det F < 0 (inverted).
 * Result: out = [cos(phi), sin(phi), cos(theta), sin(theta), s0, s1] where
 * U = Rot(phi) and V^T = Rot(theta), Rot(t) = [[cos t, -sin t], [sin t, cos t]].
 */
export function svd2(a, b, c, d, out) {
  const e = 0.5 * (a + d);
  const f = 0.5 * (a - d);
  const g = 0.5 * (c + b);
  const h = 0.5 * (c - b);
  const q = Math.hypot(e, h);
  const r = Math.hypot(f, g);
  const a1 = Math.atan2(g, f);
  const a2 = Math.atan2(h, e);
  const theta = 0.5 * (a2 - a1);
  const phi = 0.5 * (a2 + a1);
  out[0] = Math.cos(phi);
  out[1] = Math.sin(phi);
  out[2] = Math.cos(theta);
  out[3] = Math.sin(theta);
  out[4] = q + r;
  out[5] = q - r;
  return out;
}

/** Lamé parameters (plane-strain convention, same formula as 3D). */
export function lame(E, nu) {
  return {
    mu: E / (2 * (1 + nu)),
    lambda: (E * nu) / ((1 + nu) * (1 - 2 * nu)),
  };
}
