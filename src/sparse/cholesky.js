// General sparse Cholesky (A = L L^T) for symmetric positive definite matrices.
//
//   1. fill-reducing ordering (Reverse Cuthill-McKee, see rcm())
//   2. elimination tree + symbolic column counts
//   3. up-looking numeric factorization (Davis, "Direct Methods for Sparse
//      Linear Systems", cs_chol), L stored as CSC with the diagonal first
//   4. forward / backward substitution with the permutation applied
//
// No assumption is made about the mesh: any symmetric sparsity pattern works.

/**
 * Reverse Cuthill-McKee ordering of the graph given by (ti, tj) pairs.
 * Returns perm with perm[newIndex] = oldIndex. Handles disconnected graphs.
 */
export function rcm(n, ti, tj) {
  // adjacency (CSR), symmetric, no self loops, no duplicates
  const deg0 = new Int32Array(n + 1);
  for (let k = 0; k < ti.length; k++) {
    if (ti[k] !== tj[k]) {
      deg0[ti[k] + 1]++;
      deg0[tj[k] + 1]++;
    }
  }
  for (let i = 0; i < n; i++) deg0[i + 1] += deg0[i];
  const fill = deg0.slice(0, n);
  const adjRaw = new Int32Array(deg0[n]);
  for (let k = 0; k < ti.length; k++) {
    if (ti[k] !== tj[k]) {
      adjRaw[fill[ti[k]]++] = tj[k];
      adjRaw[fill[tj[k]]++] = ti[k];
    }
  }
  const ptr = new Int32Array(n + 1);
  const adj = [];
  for (let i = 0; i < n; i++) {
    const nb = Array.from(new Set(adjRaw.subarray(deg0[i], deg0[i + 1])));
    adj.push(nb);
    ptr[i + 1] = ptr[i] + nb.length;
  }
  const deg = (i) => adj[i].length;

  const level = new Int32Array(n);
  // BFS from `start` over unvisited nodes; returns {order, depth}
  const visited = new Uint8Array(n);
  const bfsLevels = (start, mark) => {
    // mark: Int32Array stamp to avoid reallocating; here a local Set-free approach
    const queue = [start];
    level[start] = 0;
    mark[start] = 1;
    let head = 0;
    while (head < queue.length) {
      const u = queue[head++];
      for (const w of adj[u]) {
        if (!mark[w] && !visited[w]) {
          mark[w] = 1;
          level[w] = level[u] + 1;
          queue.push(w);
        }
      }
    }
    for (const q of queue) mark[q] = 0;
    return queue;
  };

  const mark = new Uint8Array(n);
  const order = [];
  // process components starting from the lowest-degree unvisited node
  const byDegree = Array.from({ length: n }, (_, i) => i).sort((a, b) => deg(a) - deg(b));
  for (const seed of byDegree) {
    if (visited[seed]) continue;
    // pseudo-peripheral start node
    let start = seed;
    let queue = bfsLevels(start, mark);
    let ecc = level[queue[queue.length - 1]];
    for (let iter = 0; iter < 8; iter++) {
      const lastLevel = ecc;
      let cand = -1;
      for (let q = queue.length - 1; q >= 0 && level[queue[q]] === lastLevel; q--) {
        if (cand < 0 || deg(queue[q]) < deg(cand)) cand = queue[q];
      }
      const q2 = bfsLevels(cand, mark);
      const e2 = level[q2[q2.length - 1]];
      if (e2 > ecc) {
        start = cand;
        queue = q2;
        ecc = e2;
      } else break;
    }
    // Cuthill-McKee BFS: neighbours visited in increasing degree
    const comp = [start];
    visited[start] = 1;
    for (let head = 0; head < comp.length; head++) {
      const u = comp[head];
      const nb = adj[u].filter((w) => !visited[w]).sort((a, b) => deg(a) - deg(b));
      for (const w of nb) {
        visited[w] = 1;
        comp.push(w);
      }
    }
    for (const c of comp) order.push(c);
  }
  order.reverse();
  return Int32Array.from(order);
}

export class SparseCholesky {
  /**
   * @param n          matrix dimension
   * @param ti,tj,tv   triplets of the FULL symmetric matrix (both (i,j) and (j,i));
   *                   duplicates are summed.
   * @param ordering   'rcm' (default) or 'none'
   */
  constructor(n, ti, tj, tv, { ordering = 'rcm' } = {}) {
    this.n = n;
    this.perm = ordering === 'rcm' ? rcm(n, ti, tj) : Int32Array.from({ length: n }, (_, i) => i);
    const iperm = new Int32Array(n);
    for (let k = 0; k < n; k++) iperm[this.perm[k]] = k;

    // upper triangle of P A P^T in CSC (rows may be unsorted, duplicates allowed)
    const cnt = new Int32Array(n + 1);
    for (let k = 0; k < ti.length; k++) {
      const a = iperm[ti[k]];
      const b = iperm[tj[k]];
      if (a <= b) cnt[b + 1]++;
    }
    for (let j = 0; j < n; j++) cnt[j + 1] += cnt[j];
    const Ap = cnt;
    const Ai = new Int32Array(Ap[n]);
    const Ax = new Float64Array(Ap[n]);
    const next = Ap.slice(0, n);
    for (let k = 0; k < ti.length; k++) {
      const a = iperm[ti[k]];
      const b = iperm[tj[k]];
      if (a <= b) {
        const p = next[b]++;
        Ai[p] = a;
        Ax[p] = tv[k];
      }
    }
    this._factor(Ap, Ai, Ax);
    this._work = new Float64Array(n);
  }

  _factor(Ap, Ai, Ax) {
    const n = this.n;
    // elimination tree
    const parent = new Int32Array(n).fill(-1);
    const ancestor = new Int32Array(n).fill(-1);
    for (let k = 0; k < n; k++) {
      for (let p = Ap[k]; p < Ap[k + 1]; p++) {
        let i = Ai[p];
        while (i !== -1 && i < k) {
          const inext = ancestor[i];
          ancestor[i] = k;
          if (inext === -1) parent[i] = k;
          i = inext;
        }
      }
    }
    const s = new Int32Array(n);
    const w = new Int32Array(n).fill(-1);
    // non-zero pattern of row k of L (excluding diagonal) in topological order
    const ereach = (k) => {
      let top = n;
      w[k] = k;
      for (let p = Ap[k]; p < Ap[k + 1]; p++) {
        let i = Ai[p];
        if (i > k) continue;
        let len = 0;
        for (; w[i] !== k; i = parent[i]) {
          s[len++] = i;
          w[i] = k;
        }
        while (len > 0) s[--top] = s[--len];
      }
      return top;
    };
    // symbolic: column counts of L
    const colCount = new Int32Array(n).fill(1); // diagonal
    for (let k = 0; k < n; k++) {
      for (let top = ereach(k); top < n; top++) colCount[s[top]]++;
    }
    const Lp = new Int32Array(n + 1);
    for (let j = 0; j < n; j++) Lp[j + 1] = Lp[j] + colCount[j];
    const Li = new Int32Array(Lp[n]);
    const Lx = new Float64Array(Lp[n]);
    const c = Lp.slice(0, n); // next free slot per column
    const x = new Float64Array(n);
    w.fill(-1);
    for (let k = 0; k < n; k++) {
      const top0 = ereach(k);
      for (let p = Ap[k]; p < Ap[k + 1]; p++) if (Ai[p] <= k) x[Ai[p]] += Ax[p];
      let d = x[k];
      x[k] = 0;
      for (let top = top0; top < n; top++) {
        const i = s[top];
        const lki = x[i] / Lx[Lp[i]];
        x[i] = 0;
        for (let p = Lp[i] + 1; p < c[i]; p++) x[Li[p]] -= Lx[p] * lki;
        d -= lki * lki;
        const p = c[i]++;
        Li[p] = k;
        Lx[p] = lki;
      }
      if (!(d > 0)) throw new Error('SparseCholesky: matrix is not positive definite');
      const p = c[k]++;
      Li[p] = k;
      Lx[p] = Math.sqrt(d);
    }
    this.Lp = Lp;
    this.Li = Li;
    this.Lx = Lx;
  }

  get nnzL() {
    return this.Lp[this.n];
  }

  /** Solve A x = b. `out` may alias `b`. */
  solve(b, out) {
    const { n, perm, Lp, Li, Lx } = this;
    const x = this._work;
    for (let k = 0; k < n; k++) x[k] = b[perm[k]];
    for (let j = 0; j < n; j++) {
      x[j] /= Lx[Lp[j]];
      const xj = x[j];
      for (let p = Lp[j] + 1; p < Lp[j + 1]; p++) x[Li[p]] -= Lx[p] * xj;
    }
    for (let j = n - 1; j >= 0; j--) {
      let sum = x[j];
      for (let p = Lp[j] + 1; p < Lp[j + 1]; p++) sum -= Lx[p] * x[Li[p]];
      x[j] = sum / Lx[Lp[j]];
    }
    for (let k = 0; k < n; k++) out[perm[k]] = x[k];
    return out;
  }
}
