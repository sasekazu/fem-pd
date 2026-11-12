// Automatic triangle mesh generation (rectangle and disc).
// Meshes are plain typed arrays (no three.js dependency):
//   n, X (Float64Array 2n, rest = initial positions), tris (Uint32Array 3m, CCW),
//   Dinv (Float64Array 4m: inverse of the rest edge matrix Dm = [e1 e2], row-major),
//   area (Float64Array m), fixedIdx (default pinned vertices), tipIdx (probe vertex),
//   boundary (Uint32Array, vertices on the boundary), outline (Float64Array xy of the rest boundary loop),
//   kind ('rect' | 'disc') plus the shape parameters.

function finalizeMesh(X, tris) {
  const n = X.length / 2;
  const m = tris.length / 3;
  const Dinv = new Float64Array(4 * m);
  const area = new Float64Array(m);
  for (let e = 0; e < m; e++) {
    const i0 = tris[3 * e];
    const i1 = tris[3 * e + 1];
    const i2 = tris[3 * e + 2];
    const e1x = X[2 * i1] - X[2 * i0];
    const e1y = X[2 * i1 + 1] - X[2 * i0 + 1];
    const e2x = X[2 * i2] - X[2 * i0];
    const e2y = X[2 * i2 + 1] - X[2 * i0 + 1];
    const det = e1x * e2y - e2x * e1y;
    area[e] = 0.5 * det;
    // Dm = [[e1x, e2x], [e1y, e2y]]
    Dinv[4 * e] = e2y / det;
    Dinv[4 * e + 1] = -e2x / det;
    Dinv[4 * e + 2] = -e1y / det;
    Dinv[4 * e + 3] = e1x / det;
  }
  // boundary vertices: endpoints of edges that belong to exactly one triangle
  const edgeCount = new Map();
  const bump = (a, b) => {
    const key = a < b ? a * n + b : b * n + a;
    edgeCount.set(key, (edgeCount.get(key) ?? 0) + 1);
  };
  for (let e = 0; e < m; e++) {
    bump(tris[3 * e], tris[3 * e + 1]);
    bump(tris[3 * e + 1], tris[3 * e + 2]);
    bump(tris[3 * e + 2], tris[3 * e]);
  }
  const onBoundary = new Set();
  for (const [key, c] of edgeCount) {
    if (c === 1) {
      onBoundary.add(Math.floor(key / n));
      onBoundary.add(key % n);
    }
  }
  return { n, X, tris, Dinv, area, boundary: Uint32Array.from(onBoundary).sort() };
}

/**
 * Split the rectangle [0,width] x [-height/2, height/2] into nx*ny cells and
 * each cell into two triangles. The diagonal direction alternates in a
 * checkerboard pattern to reduce directional bias.
 * fixedIdx = vertices on the left edge (x = 0); tipIdx = middle of the right edge.
 */
export function createRectMesh(width = 4, height = 1, nx = 40, ny = 8) {
  const cols = nx + 1;
  const n = cols * (ny + 1);
  const X = new Float64Array(2 * n);
  for (let iy = 0; iy <= ny; iy++) {
    for (let ix = 0; ix <= nx; ix++) {
      const i = iy * cols + ix;
      X[2 * i] = (ix * width) / nx;
      X[2 * i + 1] = (iy * height) / ny - height / 2;
    }
  }

  const tris = new Uint32Array(6 * nx * ny);
  let t = 0;
  const push = (a, b, c) => {
    tris[t++] = a;
    tris[t++] = b;
    tris[t++] = c;
  };
  for (let iy = 0; iy < ny; iy++) {
    for (let ix = 0; ix < nx; ix++) {
      const v00 = iy * cols + ix;
      const v10 = v00 + 1;
      const v01 = v00 + cols;
      const v11 = v01 + 1;
      if ((ix + iy) % 2 === 0) {
        push(v00, v10, v11);
        push(v00, v11, v01);
      } else {
        push(v00, v10, v01);
        push(v10, v11, v01);
      }
    }
  }

  const fixedIdx = new Uint32Array(ny + 1);
  for (let iy = 0; iy <= ny; iy++) fixedIdx[iy] = iy * cols;
  const tipIdx = Math.floor(ny / 2) * cols + nx;
  const rightIdx = new Uint32Array(ny + 1);
  for (let iy = 0; iy <= ny; iy++) rightIdx[iy] = iy * cols + nx;
  const outline = Float64Array.from([0, -height / 2, width, -height / 2, width, height / 2, 0, height / 2]);

  return { kind: 'rect', nx, ny, width, height, ...finalizeMesh(X, tris), fixedIdx, leftIdx: fixedIdx, rightIdx, tipIdx, outline };
}

/**
 * Disc of the given radius made of concentric rings (ring k has 6k vertices),
 * triangulated by strips between neighbouring rings. Nearly equilateral
 * triangles everywhere, no preferred direction. Nothing is pinned.
 * tipIdx = bottom-most vertex.
 */
export function createDiscMesh(radius = 0.5, rings = 8, center = [0, 0]) {
  const base = (k) => (k === 0 ? 0 : 1 + 3 * k * (k - 1)); // first vertex index of ring k
  const n = base(rings + 1);
  const X = new Float64Array(2 * n);
  X[0] = center[0];
  X[1] = center[1];
  for (let k = 1; k <= rings; k++) {
    const cnt = 6 * k;
    const r = (radius * k) / rings;
    for (let p = 0; p < cnt; p++) {
      const a = (2 * Math.PI * p) / cnt;
      X[2 * (base(k) + p)] = center[0] + r * Math.cos(a);
      X[2 * (base(k) + p) + 1] = center[1] + r * Math.sin(a);
    }
  }
  const tris = [];
  for (let k = 1; k <= rings; k++) {
    const outer = (p) => base(k) + (((p % (6 * k)) + 6 * k) % (6 * k));
    const inner = (p) => (k === 1 ? 0 : base(k - 1) + (((p % (6 * (k - 1))) + 6 * (k - 1)) % (6 * (k - 1))));
    for (let s = 0; s < 6; s++) {
      for (let j = 0; j < k; j++) {
        tris.push(outer(s * k + j), outer(s * k + j + 1), inner(s * (k - 1) + j));
        if (j < k - 1) tris.push(inner(s * (k - 1) + j), outer(s * k + j + 1), inner(s * (k - 1) + j + 1));
      }
    }
  }
  const T = Uint32Array.from(tris);
  let tipIdx = 0;
  for (let i = 1; i < n; i++) if (X[2 * i + 1] < X[2 * tipIdx + 1]) tipIdx = i;
  const outerN = 6 * rings;
  const outline = new Float64Array(2 * outerN);
  for (let p = 0; p < outerN; p++) {
    outline[2 * p] = X[2 * (base(rings) + p)];
    outline[2 * p + 1] = X[2 * (base(rings) + p) + 1];
  }
  return {
    kind: 'disc',
    radius,
    rings,
    ...finalizeMesh(X, T),
    fixedIdx: new Uint32Array(0),
    tipIdx,
    outline,
  };
}
