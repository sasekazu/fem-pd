// Tiny canvas line plot for (x, y1, y2, ...) streams. Handles non-monotonic x (load/unload loops).

export class LinePlot {
  /**
   * @param parent   DOM element to append the canvas to
   * @param opts     { title, xLabel, yLabel, series: [{name, color}], yClamp: [lo, hi] (optional display clamp),
   *                   zeroLine: boolean, xRange: [lo, hi] (optional fixed x range) }
   */
  constructor(parent, opts) {
    this.opts = opts;
    this.canvas = document.createElement('canvas');
    this.canvas.style.cssText = 'width:100%;height:100%;display:block';
    parent.appendChild(this.canvas);
    this.xs = [];
    this.ys = opts.series.map(() => []);
    this.dirty = true;
  }

  clear() {
    this.xs.length = 0;
    this.ys.forEach((a) => (a.length = 0));
    this.dirty = true;
  }

  push(x, ...ys) {
    if (this.xs.length > 20000) return;
    this.xs.push(x);
    ys.forEach((y, k) => this.ys[k].push(Number.isFinite(y) ? y : NaN));
    this.dirty = true;
  }

  static ticks(lo, hi, count = 5) {
    const span = hi - lo || 1;
    const raw = span / count;
    const mag = 10 ** Math.floor(Math.log10(raw));
    const step = [1, 2, 5, 10].map((m) => m * mag).find((s) => s >= raw) ?? raw;
    const out = [];
    for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9 * span; v += step) out.push(v);
    return out;
  }

  static fmt(v) {
    const a = Math.abs(v);
    if (a >= 1e5 || (a > 0 && a < 1e-2)) return v.toExponential(1);
    return Number(v.toPrecision(3)).toString();
  }

  draw() {
    if (!this.dirty) return;
    this.dirty = false;
    const { canvas, opts } = this;
    const dpr = window.devicePixelRatio || 1;
    const W = Math.max(10, Math.floor(canvas.clientWidth * dpr));
    const H = Math.max(10, Math.floor(canvas.clientHeight * dpr));
    if (canvas.width !== W || canvas.height !== H) {
      canvas.width = W;
      canvas.height = H;
    }
    const g = canvas.getContext('2d');
    g.setTransform(1, 0, 0, 1, 0, 0);
    g.clearRect(0, 0, W, H);
    g.scale(dpr, dpr);
    const w = W / dpr;
    const h = H / dpr;
    const m = { l: 52, r: 10, t: 22, b: 28 };

    // ranges
    let xlo = Infinity;
    let xhi = -Infinity;
    let ylo = Infinity;
    let yhi = -Infinity;
    const clamp = (v) => (opts.yClamp ? Math.min(opts.yClamp[1], Math.max(opts.yClamp[0], v)) : v);
    for (let i = 0; i < this.xs.length; i++) {
      xlo = Math.min(xlo, this.xs[i]);
      xhi = Math.max(xhi, this.xs[i]);
      for (const ys of this.ys) {
        const v = ys[i];
        if (Number.isFinite(v)) {
          ylo = Math.min(ylo, clamp(v));
          yhi = Math.max(yhi, clamp(v));
        }
      }
    }
    if (opts.xRange) [xlo, xhi] = opts.xRange;
    if (!Number.isFinite(xlo) || xhi <= xlo) [xlo, xhi] = [xlo || 0, (xlo || 0) + 1];
    if (!Number.isFinite(ylo)) [ylo, yhi] = [0, 1];
    if (opts.zeroLine) {
      ylo = Math.min(ylo, 0);
      yhi = Math.max(yhi, 0);
    }
    if (yhi - ylo < 1e-12) yhi = ylo + 1;
    const pad = 0.05 * (yhi - ylo);
    ylo -= pad;
    yhi += pad;

    const X = (v) => m.l + ((v - xlo) / (xhi - xlo)) * (w - m.l - m.r);
    const Y = (v) => h - m.b - ((clamp(v) - ylo) / (yhi - ylo)) * (h - m.t - m.b);

    g.font = '11px system-ui, sans-serif';
    g.fillStyle = '#93a0b4';
    g.strokeStyle = '#2a3140';
    g.lineWidth = 1;
    g.textBaseline = 'middle';
    g.textAlign = 'right';
    for (const v of LinePlot.ticks(ylo, yhi, 4)) {
      g.beginPath();
      g.moveTo(m.l, Y(v));
      g.lineTo(w - m.r, Y(v));
      g.stroke();
      g.fillText(LinePlot.fmt(v), m.l - 5, Y(v));
    }
    g.textAlign = 'center';
    g.textBaseline = 'top';
    for (const v of LinePlot.ticks(xlo, xhi, 6)) {
      g.beginPath();
      g.moveTo(X(v), m.t);
      g.lineTo(X(v), h - m.b);
      g.stroke();
      g.fillText(LinePlot.fmt(v), X(v), h - m.b + 4);
    }
    if (opts.zeroLine) {
      g.strokeStyle = '#6b7788';
      g.setLineDash([4, 3]);
      g.beginPath();
      g.moveTo(m.l, Y(0));
      g.lineTo(w - m.r, Y(0));
      g.stroke();
      g.setLineDash([]);
    }
    // series
    opts.series.forEach((s, k) => {
      g.strokeStyle = s.color;
      g.lineWidth = s.width ?? 1.6;
      g.setLineDash(s.dash ?? []);
      g.beginPath();
      let pen = false;
      for (let i = 0; i < this.xs.length; i++) {
        const v = this.ys[k][i];
        if (!Number.isFinite(v)) {
          pen = false;
          continue;
        }
        if (pen) g.lineTo(X(this.xs[i]), Y(v));
        else g.moveTo(X(this.xs[i]), Y(v));
        pen = true;
      }
      g.stroke();
    });
    g.setLineDash([]);
    // title + legend
    g.fillStyle = '#dfe5ee';
    g.textAlign = 'left';
    g.textBaseline = 'top';
    g.font = '12px system-ui, sans-serif';
    g.fillText(`${opts.title}${opts.xLabel ? `  (x: ${opts.xLabel})` : ''}`, m.l, 4);
    let lx = w - m.r;
    g.textAlign = 'right';
    for (const s of [...opts.series].reverse()) {
      g.fillStyle = s.color;
      g.fillText(s.name, lx, 4);
      lx -= g.measureText(s.name).width + 14;
    }
  }
}
