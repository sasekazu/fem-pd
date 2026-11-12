import test from 'node:test';
import assert from 'node:assert/strict';
import { Matrix, CholeskyDecomposition } from 'ml-matrix'; // dense reference (dev dependency)
import { SparseCholesky, rcm } from '../src/sparse/cholesky.js';
import { createRectMesh } from '../src/mesh.js';

function rng(seed) {
  let s = seed >>> 0;
  return () => ((s = (Math.imul(s, 1664525) + 1013904223) >>> 0) / 2 ** 32);
}

/** Symmetric positive definite triplets from a graph: weighted Laplacian + diagonal. */
function laplacianTriplets(n, edges, rand) {
  const ti = [];
  const tj = [];
  const tv = [];
  const diag = new Float64Array(n);
  for (let i = 0; i < n; i++) diag[i] = 0.5 + rand();
  for (const [a, b] of edges) {
    const w = 0.5 + rand();
    ti.push(a, b, a, b);
    tj.push(b, a, a, b);
    tv.push(-w, -w, w, w);
    diag[a] += 0;
  }
  for (let i = 0; i < n; i++) {
    ti.push(i);
    tj.push(i);
    tv.push(diag[i]);
  }
  return { ti: Int32Array.from(ti), tj: Int32Array.from(tj), tv: Float64Array.from(tv) };
}

function denseOf(n, { ti, tj, tv }) {
  const A = Matrix.zeros(n, n);
  for (let k = 0; k < ti.length; k++) A.set(ti[k], tj[k], A.get(ti[k], tj[k]) + tv[k]);
  return A;
}

function compare(n, trip, ordering) {
  const rand = rng(7);
  const b = Float64Array.from({ length: n }, () => rand() - 0.5);
  const ref = new CholeskyDecomposition(denseOf(n, trip)).solve(Matrix.columnVector(Array.from(b)));
  const chol = new SparseCholesky(n, trip.ti, trip.tj, trip.tv, { ordering });
  const x = new Float64Array(n);
  chol.solve(b, x);
  let err = 0;
  let norm = 0;
  for (let i = 0; i < n; i++) {
    err = Math.max(err, Math.abs(x[i] - ref.get(i, 0)));
    norm = Math.max(norm, Math.abs(ref.get(i, 0)));
  }
  assert.ok(err <= 1e-9 * Math.max(1, norm), `max error ${err} (ordering ${ordering})`);
  return chol;
}

test('rcm returns a valid permutation (including disconnected graphs)', () => {
  const ti = Int32Array.from([0, 1, 3, 4]);
  const tj = Int32Array.from([1, 2, 4, 5]); // components {0,1,2}, {3,4,5}, isolated 6
  const p = rcm(7, ti, tj);
  assert.deepEqual(Array.from(p).sort((a, b) => a - b), [0, 1, 2, 3, 4, 5, 6]);
});

test('sparse Cholesky matches dense solve: random sparse SPD graph', () => {
  const rand = rng(1);
  const n = 250;
  const edges = new Set();
  while (edges.size < 3 * n) {
    const a = Math.floor(rand() * n);
    const b = Math.floor(rand() * n);
    if (a < b) edges.add(a * n + b);
  }
  const trip = laplacianTriplets(n, [...edges].map((e) => [Math.floor(e / n), e % n]), rand);
  compare(n, trip, 'none');
  compare(n, trip, 'rcm');
});

test('sparse Cholesky on a triangle mesh with scrambled vertex numbering (unstructured-like)', () => {
  const mesh = createRectMesh(4, 1, 30, 6);
  const n = mesh.n;
  // random relabelling destroys the banded structure of the grid numbering
  const rand = rng(3);
  const label = Int32Array.from({ length: n }, (_, i) => i);
  for (let i = n - 1; i > 0; i--) {
    const j = Math.floor(rand() * (i + 1));
    [label[i], label[j]] = [label[j], label[i]];
  }
  const edges = [];
  for (let e = 0; e < mesh.tris.length / 3; e++) {
    const t = [0, 1, 2].map((k) => label[mesh.tris[3 * e + k]]);
    edges.push([t[0], t[1]], [t[1], t[2]], [t[2], t[0]]);
  }
  const trip = laplacianTriplets(n, edges, rand);
  const none = compare(n, trip, 'none');
  const withRcm = compare(n, trip, 'rcm');
  console.log(`  scrambled mesh n=${n}: nnz(L) none=${none.nnzL}  rcm=${withRcm.nnzL}  dense=${(n * (n + 1)) / 2}`);
  assert.ok(withRcm.nnzL < none.nnzL, 'RCM must reduce fill-in on a scrambled numbering');
});

test('sparse Cholesky rejects an indefinite matrix', () => {
  const ti = Int32Array.from([0, 0, 1, 1]);
  const tj = Int32Array.from([0, 1, 0, 1]);
  const tv = Float64Array.from([1, 2, 2, 1]);
  assert.throws(() => new SparseCholesky(2, ti, tj, tv), /positive definite/);
});
