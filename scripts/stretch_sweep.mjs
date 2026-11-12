// Stretch each solver up to a maximum stretch ratio and report when elements first invert.
// usage: node scripts/stretch_sweep.mjs [maxStretch=10] [speed=0.5] [substeps=2] [pdIter=20] [nx=16] [model=corotational|neohookean]
import { createRectMesh } from '../src/mesh.js';
import { CorotationalFEM } from '../src/sim/corotationalFEM.js';
import { HyperelasticFEM } from '../src/sim/hyperelasticFEM.js';
import { ProjectiveDynamics } from '../src/sim/projectiveDynamics.js';
import { StretchDriver } from '../src/stretch.js';

const argv = process.argv.slice(2);
const [maxS = 10, speed = 0.5, sub = 2, pdIt = 20, nx = 16] = argv.slice(0, 5).map(Number);
const model = argv[5] ?? 'corotational';
const FEM = model === 'neohookean' ? HyperelasticFEM : CorotationalFEM;
const E = 1e5;
const h = 1 / 60 / sub;
const rows = [];
for (const nu of [0, 0.2, 0.3, 0.4, 0.45]) {
  const res = {};
  for (const [name, S] of [['FEM', FEM], ['PD', ProjectiveDynamics]]) {
    const mesh = createRectMesh(1, 1, nx, nx);
    const sim = new S(mesh, { E, nu, density: 1000, gravity: 0, damping: 2, model });
    if (name === 'PD') sim.iterations = pdIt;
    const drv = new StretchDriver(mesh, { speed, maxStretch: maxS });
    drv.attach(sim);
    let firstInv = null, maxInv = 0, minDet = Infinity, bad = false, ms = 0, steps = 0, forceAt2 = null;
    while (drv.mode === 'load') {
      drv.advance(h, sim); sim.step(h); ms += sim.stats.stepMs; steps++;
      const st = sim.deformationStats();
      if (!Number.isFinite(st.minDet) || !sim.x.every(Number.isFinite)) { bad = true; break; }
      if (st.inverted > 0 && firstInv === null) firstInv = drv.stretch;
      maxInv = Math.max(maxInv, st.inverted); minDet = Math.min(minDet, st.minDet);
      if (forceAt2 === null && drv.stretch >= 2) forceAt2 = sim.gripForce(mesh.rightIdx, +1);
    }
    const st = sim.deformationStats();
    res[name] = { firstInv, maxInv, minDet, bad, endInv: st.inverted, force: bad ? NaN : sim.gripForce(mesh.rightIdx, +1), forceAt2, ms: ms / steps };
  }
  rows.push({ nu, predicted: nu > 0 ? 1 + (1 - nu) / nu : Infinity, ...res });
}
const f = (v, d = 2) => (v === null ? '  -  ' : Number.isFinite(v) ? v.toFixed(d) : String(v));
console.log(`model=${model} E=${E} mesh ${nx}x${nx} speed=${speed}/side max s=${maxS} h=1/${60 * sub} PD iters=${pdIt}`);
console.log('nu   s*(pred)  | FEM: firstInv maxInv minDet   F@s=2   F@end   ms | PD: firstInv maxInv minDet   F@s=2   F@end   ms');
for (const r of rows) {
  const c = (x) => `${f(x.firstInv).padStart(8)} ${String(x.maxInv).padStart(6)} ${f(x.minDet, 3).padStart(7)} ${f(x.forceAt2, 0).padStart(8)} ${f(x.force, 0).padStart(8)} ${f(x.ms).padStart(5)}${x.bad ? ' NaN!' : ''}`;
  console.log(`${String(r.nu).padEnd(4)} ${f(r.predicted).padStart(8)}  | ${c(r.FEM)} | ${c(r.PD)}`);
}
