// Minimal sparse linear algebra: CSR matrix-vector product and
// Jacobi-preconditioned conjugate gradients with a free-DOF mask.

/** y = A x for CSR (rowPtr, colIdx, vals). */
export function csrMul(rowPtr, colIdx, vals, x, y) {
  const n = rowPtr.length - 1;
  for (let r = 0; r < n; r++) {
    let s = 0;
    for (let k = rowPtr[r]; k < rowPtr[r + 1]; k++) s += vals[k] * x[colIdx[k]];
    y[r] = s;
  }
}

/**
 * Preconditioned CG for  A x = b  restricted to DOFs with free[i] === 1.
 * Constrained entries of x stay exactly 0 (x is the initial guess, assumed 0 there).
 *
 * @param apply   (p, out) => void   computes out = A p
 * @param diag    diagonal of A (Jacobi preconditioner)
 * @param work    optional preallocated {r,z,p,Ap}
 * @returns iteration count
 */
export function pcg({ apply, diag, b, x, free, tol = 1e-6, maxIter = 200, work }) {
  const n = b.length;
  const r = work?.r ?? new Float64Array(n);
  const z = work?.z ?? new Float64Array(n);
  const p = work?.p ?? new Float64Array(n);
  const Ap = work?.Ap ?? new Float64Array(n);

  apply(x, Ap);
  let bnorm = 0;
  for (let i = 0; i < n; i++) {
    if (free[i]) {
      r[i] = b[i] - Ap[i];
      bnorm += b[i] * b[i];
    } else {
      r[i] = 0;
    }
  }
  bnorm = Math.sqrt(bnorm);
  if (bnorm === 0) return 0;

  let rz = 0;
  for (let i = 0; i < n; i++) {
    z[i] = free[i] ? r[i] / diag[i] : 0;
    p[i] = z[i];
    rz += r[i] * z[i];
  }

  let it = 0;
  for (; it < maxIter; it++) {
    let rnorm = 0;
    for (let i = 0; i < n; i++) rnorm += r[i] * r[i];
    if (Math.sqrt(rnorm) <= tol * bnorm) break;

    apply(p, Ap);
    let pAp = 0;
    for (let i = 0; i < n; i++) {
      if (!free[i]) Ap[i] = 0;
      pAp += p[i] * Ap[i];
    }
    if (pAp <= 0) break;
    const alpha = rz / pAp;
    let rzNew = 0;
    for (let i = 0; i < n; i++) {
      x[i] += alpha * p[i];
      r[i] -= alpha * Ap[i];
      z[i] = free[i] ? r[i] / diag[i] : 0;
      rzNew += r[i] * z[i];
    }
    const beta = rzNew / rz;
    rz = rzNew;
    for (let i = 0; i < n; i++) p[i] = z[i] + beta * p[i];
  }
  return it;
}
