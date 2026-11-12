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

/** Lamé parameters (plane-strain convention, same formula as 3D). */
export function lame(E, nu) {
  return {
    mu: E / (2 * (1 + nu)),
    lambda: (E * nu) / ((1 + nu) * (1 - 2 * nu)),
  };
}
