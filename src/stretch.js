// Uniaxial stretch / compression test: the left and right edge columns (grips) of a rectangle mesh
// are moved apart (stretch) or together (compress) with a prescribed displacement. Optionally the
// grips can be released, after which the body moves freely (snap-back). Works with any solver
// exposing setPinned / setPinTarget (no three.js dependency).

export class StretchDriver {
  /**
   * @param mesh        rectangle mesh from createRectMesh (needs leftIdx, rightIdx, width, X)
   * @param speed       grip speed per side [m/s]
   * @param maxStretch  stretch ratio at which loading stops (s = current width / initial width, > 1)
   * @param compress    move the grips together instead of apart
   * @param minStretch  compression ratio at which loading stops (0 < s < 1), compress mode only
   */
  constructor(mesh, { speed = 0.5, maxStretch = 10, compress = false, minStretch = 0.3 } = {}) {
    this.mesh = mesh;
    this.speed = speed;
    this.maxStretch = maxStretch;
    this.compress = compress;
    this.minStretch = minStretch;
    this.mode = 'load'; // 'load' | 'hold' | 'unload' | 'release'
    this.d = 0; // displacement magnitude of each grip [m] (outwards when stretching, inwards when compressing)
    this.time = 0;
  }

  get sign() {
    return this.compress ? -1 : 1;
  }

  /** Grip displacement at which loading stops. */
  get dMax() {
    const target = this.compress ? this.minStretch : this.maxStretch;
    return (this.mesh.width * Math.abs(target - 1)) / 2;
  }

  /** Prescribed stretch ratio (valid while the grips are held). */
  get stretch() {
    return 1 + (this.sign * 2 * this.d) / this.mesh.width;
  }

  get released() {
    return this.mode === 'release';
  }

  /** Actual stretch ratio of a solver: distance between the mean grip positions over the rest width. */
  currentStretch(sim) {
    const { leftIdx, rightIdx, width } = this.mesh;
    const mean = (idx) => idx.reduce((s, i) => s + sim.x[2 * i], 0) / idx.length;
    return (mean(rightIdx) - mean(leftIdx)) / width;
  }

  /** Make the given solvers hold both grip columns as prescribed-motion vertices. */
  attach(...sims) {
    const idx = [...this.mesh.leftIdx, ...this.mesh.rightIdx];
    for (const s of sims) s.setPinned(idx);
    this.apply(...sims);
  }

  /** Let go of the grips: the body then moves freely (no boundary conditions). */
  release(...sims) {
    if (this.released) return;
    this.mode = 'release';
    for (const s of sims) s.setPinned([]);
  }

  /** Advance the grip displacement by h and set the pin targets of the solvers. */
  advance(h, ...sims) {
    this.time += h;
    if (this.released) return;
    if (this.mode === 'load') {
      this.d = Math.min(this.d + this.speed * h, this.dMax);
      if (this.d >= this.dMax) this.mode = 'hold';
    } else if (this.mode === 'unload') {
      this.d = Math.max(this.d - this.speed * h, 0);
    }
    this.apply(...sims);
  }

  apply(...sims) {
    const { X, leftIdx, rightIdx } = this.mesh;
    const dx = this.sign * this.d;
    for (const s of sims) {
      for (const i of leftIdx) s.setPinTarget(i, X[2 * i] - dx, X[2 * i + 1]);
      for (const i of rightIdx) s.setPinTarget(i, X[2 * i] + dx, X[2 * i + 1]);
    }
  }

  reset(...sims) {
    const wasReleased = this.released;
    this.d = 0;
    this.time = 0;
    this.mode = 'load';
    for (const s of sims) s.reset();
    if (wasReleased) this.attach(...sims); // grab the grips again
    else this.apply(...sims);
  }
}
