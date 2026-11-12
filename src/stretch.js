// Uniaxial stretch test: the left and right edge columns (grips) of a rectangle mesh are
// moved apart with a prescribed displacement. Works with any solver exposing
// setPinned / setPinTarget (no three.js dependency).

export class StretchDriver {
  /**
   * @param mesh        rectangle mesh from createRectMesh (needs leftIdx, rightIdx, width, X)
   * @param speed       grip speed per side [m/s]
   * @param maxStretch  stretch ratio at which loading stops (s = current width / initial width)
   */
  constructor(mesh, { speed = 0.5, maxStretch = 10 } = {}) {
    this.mesh = mesh;
    this.speed = speed;
    this.maxStretch = maxStretch;
    this.mode = 'load'; // 'load' | 'hold' | 'unload'
    this.d = 0; // displacement of each grip [m]
    this.time = 0;
  }

  get dMax() {
    return (this.mesh.width * (this.maxStretch - 1)) / 2;
  }

  get stretch() {
    return 1 + (2 * this.d) / this.mesh.width;
  }

  /** Make the given solvers hold both grip columns as prescribed-motion vertices. */
  attach(...sims) {
    const idx = [...this.mesh.leftIdx, ...this.mesh.rightIdx];
    for (const s of sims) s.setPinned(idx);
    this.apply(...sims);
  }

  /** Advance the grip displacement by h and set the pin targets of the solvers. */
  advance(h, ...sims) {
    if (this.mode === 'load') {
      this.d = Math.min(this.d + this.speed * h, this.dMax);
      if (this.d >= this.dMax) this.mode = 'hold';
    } else if (this.mode === 'unload') {
      this.d = Math.max(this.d - this.speed * h, 0);
    }
    this.time += h;
    this.apply(...sims);
  }

  apply(...sims) {
    const { X, leftIdx, rightIdx } = this.mesh;
    for (const s of sims) {
      for (const i of leftIdx) s.setPinTarget(i, X[2 * i] - this.d, X[2 * i + 1]);
      for (const i of rightIdx) s.setPinTarget(i, X[2 * i] + this.d, X[2 * i + 1]);
    }
  }

  reset(...sims) {
    this.d = 0;
    this.time = 0;
    this.mode = 'load';
    for (const s of sims) s.reset();
    this.apply(...sims);
  }
}
