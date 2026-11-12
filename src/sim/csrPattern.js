/**
 * CSR sparsity pattern for a scalar-DOF stiffness matrix of a triangle mesh (DOFs interleaved x, y
 * per vertex) plus per-element scatter offsets.
 *
 * scatter[(e * 9 + a * 3 + b) * 2 + r] = offset into `vals` of the first entry of row r (0 = x, 1 = y)
 * of the 2x2 block (vertex a, vertex b) of element e.
 */
export function buildCsrPattern(mesh) {
  const { tris, n } = mesh;
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
  const csr = { rowPtr, colIdx, vals: new Float64Array(colIdx.length), diag: new Float64Array(2 * n) };
  return { csr, scatter };
}
