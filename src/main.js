import * as THREE from 'three';
import GUI from 'lil-gui';
import { createRectMesh, createDiscMesh } from './mesh.js';
import { CorotationalFEM } from './sim/corotationalFEM.js';
import { HyperelasticFEM } from './sim/hyperelasticFEM.js';
import { ProjectiveDynamics } from './sim/projectiveDynamics.js';
import { DEFAULT_PARAMS } from './sim/base.js';
import { SoftBodyView, fitCamera } from './render.js';
import { StretchDriver } from './stretch.js';
import { LinePlot } from './plot.js';

const canvas = document.getElementById('canvas');
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setScissorTest(true);

const BALL_RADIUS = 0.5;
const PLOT_H = 230; // px reserved for the plots in the stretch scene

// Per-scene presets (applied when the scene is selected).
const SCENES = {
  beam: {
    E: 5e6, nu: 0.3, density: 1000, gravity: 9.8, damping: 1,
    width: 4, height: 1, nx: 40, ny: 8,
    substeps: 1, pdIterations: 10, floor: false, floorY: -1.6, contactK: 1e6, strainMax: 0.15,
    view: { cx: 2.2, cy: -0.9, w: 5.6, h: 4.2 },
  },
  ball: {
    E: 1e6, nu: 0.3, density: 1000, gravity: 9.8, damping: 0,
    substeps: 4, pdIterations: 20, floor: true, floorY: 0, contactK: 1e6, strainMax: 0.15,
    view: { cx: 0, cy: 0.95, w: 3.6, h: 3.0 },
  },
  stretch: {
    E: 1e5, nu: 0.3, density: 1000, gravity: 0, damping: 2,
    width: 1, height: 1, nx: 16, ny: 16,
    substeps: 2, pdIterations: 20, floor: false, floorY: -1.6, contactK: 1e6, strainMax: 2,
    view: null, // fitted to the bodies every frame
  },
};

// ---- shared settings (edited through the GUI) ----
const settings = {
  ...DEFAULT_PARAMS,
  scene: 'beam',
  nx: 40,
  ny: 8,
  rings: 8,
  gap: 1.0, // initial gap between the ball and the floor
  floor: false,
  floorY: -1.6,
  contactK: 1e6,
  lbfgs: 6,
  pullSpeed: 0.5, // stretch test: grip speed per side [m/s]
  maxStretch: 10, // stretch test: final stretch ratio
  dt: 1 / 60,
  substeps: 1,
  pdIterations: 10,
  cgTol: 1e-6,
  cgMaxIter: 300,
  timeScale: 1,
  strainMax: 0.15,
  wireframe: true,
  paused: false,
};
Object.assign(settings, SCENES[settings.scene]);

let mesh, fem, pd, viewFem, viewPd, driver = null;
let accumulator = 0;
let stepRequest = 0;
const smoothing = { fem: null, pd: null };
const track = { fem: null, pd: null };
let y0 = 0; // initial centre-of-mass height (ball scene)
let cameraRect = null; // smoothed shared view rectangle (stretch scene)

function materialParams() {
  const { model, E, nu, density, gravity, damping } = settings;
  return { model, E, nu, density, gravity, damping };
}
const isNeo = () => settings.model === 'neohookean';

/** Panel headings follow the selected material model. */
function updateLabels() {
  const set = (id, text) => (document.getElementById(id).textContent = text);
  if (isNeo()) {
    set('title-fem', 'Stable Neo-Hookean FEM');
    set('sub-fem', 'implicit Euler, Newton, PSD-projected Hessian, CG');
    set('title-pd', 'Stable Neo-Hookean PD + L-BFGS');
    set('sub-pd', 'constant PD matrix (Cholesky) as L-BFGS preconditioner');
  } else {
    set('title-fem', 'Corotational FEM');
    set('sub-fem', 'implicit Euler, warped stiffness, CG');
    set('title-pd', 'Projective Dynamics');
    set('sub-pd', 'local/global, pre-factored Cholesky');
  }
}
const contactParams = () => ({ enabled: settings.floor, floorY: settings.floorY, k: settings.contactK });

function resetTrack() {
  track.fem = { prevVy: 0, apexes: [] };
  track.pd = { prevVy: 0, apexes: [] };
}

// ---- plots (stretch scene) ----
const plotsEl = document.getElementById('plots');
const plotForce = new LinePlot(document.getElementById('plot-force'), {
  title: 'Grip force',
  xLabel: 'stretch ratio s',
  series: [
    { name: 'FEM', color: '#4ea1ff' },
    { name: 'PD', color: '#ff8a3d', dash: [5, 3] },
  ],
});
const plotDet = new LinePlot(document.getElementById('plot-det'), {
  title: 'min det F (≤ 0: inverted element)',
  xLabel: 'stretch ratio s',
  yClamp: [-3, 10],
  zeroLine: true,
  series: [
    { name: 'FEM', color: '#4ea1ff' },
    { name: 'PD', color: '#ff8a3d', dash: [5, 3] },
  ],
});
let lastSampleTime = -1;

function clearPlots() {
  plotForce.clear();
  plotDet.clear();
  lastSampleTime = -1;
}

function samplePlots() {
  if (!driver) return;
  const rf = fem.gripForce(mesh.rightIdx, +1);
  const rp = pd.gripForce(mesh.rightIdx, +1);
  plotForce.push(driver.stretch, rf, rp);
  plotDet.push(driver.stretch, fem.deformationStats().minDet, pd.deformationStats().minDet);
}

function build() {
  viewFem?.dispose();
  viewPd?.dispose();
  const ball = settings.scene === 'ball';
  const stretch = settings.scene === 'stretch';
  mesh = ball
    ? createDiscMesh(BALL_RADIUS, settings.rings, [0, settings.floorY + BALL_RADIUS + settings.gap])
    : createRectMesh(SCENES[settings.scene].width, SCENES[settings.scene].height, settings.nx, settings.ny);
  fem = isNeo() ? new HyperelasticFEM(mesh, materialParams()) : new CorotationalFEM(mesh, materialParams());
  pd = new ProjectiveDynamics(mesh, materialParams());
  fem.setContact(contactParams());
  pd.setContact(contactParams());
  applySolverSettings();
  driver = null;
  if (stretch) {
    driver = new StretchDriver(mesh, { speed: settings.pullSpeed, maxStretch: settings.maxStretch });
    driver.attach(fem, pd);
  }
  const rect = SCENES[settings.scene].view ?? { cx: 0.5, cy: 0, w: 3, h: 2 };
  viewFem = new SoftBodyView(mesh, rect);
  viewPd = new SoftBodyView(mesh, rect);
  viewFem.wireframe = viewPd.wireframe = settings.wireframe;
  y0 = fem.centerOfMass().y;
  accumulator = 0;
  smoothing.fem = smoothing.pd = null;
  cameraRect = null;
  resetTrack();
  clearPlots();
  document.documentElement.style.setProperty('--plot-h', stretch ? `${PLOT_H}px` : '0px');
  resize();
  updateLegend();
  updateLabels();
}

function applySolverSettings() {
  pd.iterations = settings.pdIterations;
  pd.lbfgsM = settings.lbfgs;
  fem.cg.tol = settings.cgTol;
  fem.cg.maxIter = settings.cgMaxIter;
}

function applyStretchSettings() {
  if (!driver) return;
  driver.speed = settings.pullSpeed;
  driver.maxStretch = settings.maxStretch;
  if (driver.mode === 'hold' && driver.d < driver.dMax) driver.mode = 'load';
  if (driver.d > driver.dMax) driver.d = driver.dMax;
}

function pushParams() {
  fem.setParams(materialParams());
  pd.setParams(materialParams());
}

function pushContact() {
  fem.setContact(contactParams());
  pd.setContact(contactParams());
}

function reset() {
  if (driver) driver.reset(fem, pd);
  else {
    fem.reset();
    pd.reset();
  }
  accumulator = 0;
  resetTrack();
  clearPlots();
}

// ---- GUI ----
const gui = new GUI({ title: 'Parameters' });
gui
  .add(settings, 'scene', { 'Cantilever beam': 'beam', 'Bouncing ball': 'ball', 'Stretch test': 'stretch' })
  .name('scene')
  .onChange((name) => {
    Object.assign(settings, SCENES[name]);
    gui.controllersRecursive().forEach((c) => c.updateDisplay());
    updateSceneControls();
    build();
  });

const fMat = gui.addFolder('Material / world');
fMat
  .add(settings, 'model', { Corotational: 'corotational', 'Stable Neo-Hookean': 'neohookean' })
  .name('material model')
  .onChange(build);
fMat.add(settings, 'E', 1e5, 1e8, 1e5).name("Young's modulus E").onChange(pushParams);
fMat.add(settings, 'nu', 0, 0.45, 0.01).name('Poisson ratio ν').onChange(pushParams);
fMat.add(settings, 'density', 100, 5000, 10).name('density ρ').onChange(pushParams);
fMat.add(settings, 'gravity', 0, 30, 0.1).name('gravity g').onChange(pushParams);
fMat.add(settings, 'damping', 0, 20, 0.1).name('damping [1/s]').onChange(pushParams);

const fStretch = gui.addFolder('Stretch test');
fStretch.add(settings, 'pullSpeed', 0.05, 3, 0.05).name('grip speed / side [m/s]').onChange(applyStretchSettings);
fStretch.add(settings, 'maxStretch', 1.5, 30, 0.5).name('max stretch ratio').onChange(applyStretchSettings);
const stretchActions = {
  load() {
    if (driver) driver.mode = driver.d < driver.dMax ? 'load' : 'hold';
  },
  hold() {
    if (driver) driver.mode = 'hold';
  },
  unload() {
    if (driver) driver.mode = 'unload';
  },
};
fStretch.add(stretchActions, 'load').name('▶ load');
fStretch.add(stretchActions, 'hold').name('⏸ hold');
fStretch.add(stretchActions, 'unload').name('◀ unload');

const fContact = gui.addFolder('Floor contact (penalty)');
fContact.add(settings, 'floor').name('floor').onChange(pushContact);
fContact
  .add(settings, 'floorY', -3, 1, 0.05)
  .name('floor height y')
  .onChange(pushContact)
  .onFinishChange(() => {
    if (settings.scene === 'ball') build();
  });
fContact
  .add(settings, 'contactK', { '1e5 (soft)': 1e5, '1e6': 1e6, '1e7': 1e7, '1e8 (hard)': 1e8 })
  .name('stiffness k [1/s²]')
  .onChange(pushContact);
fContact.add(settings, 'lbfgs', 0, 12, 1).name('L-BFGS history (PD)').onChange(applySolverSettings);

const fSim = gui.addFolder('Simulation');
fSim.add(settings, 'dt', { '1/240 s': 1 / 240, '1/120 s': 1 / 120, '1/60 s': 1 / 60, '1/30 s': 1 / 30 }).name('time step dt');
fSim.add(settings, 'substeps', 1, 8, 1).name('substeps');
fSim.add(settings, 'pdIterations', 1, 100, 1).name('PD / L-BFGS iterations').onChange(applySolverSettings);
fSim.add(settings, 'cgMaxIter', 5, 1000, 1).name('CG max iter').onChange(applySolverSettings);
fSim.add(settings, 'cgTol', [1e-3, 1e-4, 1e-6, 1e-8]).name('CG tolerance').onChange(applySolverSettings);
fSim.add(settings, 'timeScale', 0.1, 2, 0.1).name('time scale');

const fMesh = gui.addFolder('Mesh');
const cNx = fMesh.add(settings, 'nx', 4, 160, 1).name('cells along length').onFinishChange(build);
const cNy = fMesh.add(settings, 'ny', 1, 32, 1).name('cells across thickness').onFinishChange(build);
const cRings = fMesh.add(settings, 'rings', 3, 24, 1).name('ball rings').onFinishChange(build);
const cGap = fMesh.add(settings, 'gap', 0.1, 2, 0.05).name('ball drop height').onFinishChange(build);

function updateSceneControls() {
  const s = settings.scene;
  for (const c of [cNx, cNy]) s === 'ball' ? c.hide() : c.show();
  for (const c of [cRings, cGap]) s === 'ball' ? c.show() : c.hide();
  s === 'stretch' ? fStretch.show() : fStretch.hide();
  s === 'stretch' ? fContact.hide() : fContact.show();
}
updateSceneControls();

const fView = gui.addFolder('Display');
fView
  .add(settings, 'wireframe')
  .name('wireframe')
  .onChange((v) => {
    viewFem.wireframe = viewPd.wireframe = v;
  });
fView.add(settings, 'strainMax', 0.01, 5, 0.01).name('strain colour max').onChange(updateLegend);

const actions = {
  pause() {
    settings.paused = !settings.paused;
  },
  step() {
    settings.paused = true;
    stepRequest++;
  },
  reset,
};
gui.add(actions, 'pause').name('Pause / resume');
gui.add(actions, 'step').name('Step once');
gui.add(actions, 'reset').name('Reset');

function updateLegend() {
  document.getElementById('legend-max').textContent = settings.strainMax.toFixed(2);
}
updateLegend();

// ---- stats overlay ----
const ema = (prev, v) => (prev == null ? v : prev * 0.9 + v * 0.1);
const elFem = document.getElementById('stats-fem');
const elPd = document.getElementById('stats-pd');
let lastStatsUpdate = 0;

/** Record rebound apexes of the centre of mass (ball scene). */
function updateTrack(sim, tr) {
  const c = sim.centerOfMass();
  if (tr.prevVy > 0 && c.vy <= 0) tr.apexes.push(c.y);
  tr.prevVy = c.vy;
}

function statsText(sim, ms, tr, labelIt) {
  const scene = settings.scene;
  const row = (k, v) => `${k.padEnd(11)} ${String(v).padStart(9)}\n`;
  let s = row('step time', `${ms.toFixed(2)} ms`) + row(labelIt, sim.stats.iterations);
  if (sim.stats.cgIterations !== undefined) s += row('CG iters', sim.stats.cgIterations);
  if (sim.stats.lineSearch !== undefined && (settings.floor || isNeo())) s += row('backtracks', sim.stats.lineSearch);
  if (scene === 'ball') {
    const yRest = settings.floorY + BALL_RADIUS;
    s += row('centre y', sim.centerOfMass().y.toFixed(4));
    if (tr.apexes.length) {
      const rebound = (100 * (tr.apexes[0] - yRest)) / (y0 - yRest);
      s += row('rebound', `${rebound.toFixed(1)} %`);
    } else s += row('rebound', '-');
  } else if (scene === 'stretch') {
    const st = sim.deformationStats();
    s += row('stretch s', `${driver.stretch.toFixed(2)} x`);
    s += row('grip force', sim.gripForce(mesh.rightIdx, +1).toExponential(3));
    s += row('min det F', st.minDet.toFixed(3));
    s += row('inverted', `${st.inverted} / ${mesh.area.length}`);
  } else {
    const rest = mesh.X[2 * mesh.tipIdx + 1];
    s += row('tip y', (sim.tip[1] - rest).toFixed(4));
  }
  s += row('energy', sim.energies().total.toFixed(1));
  return s;
}

function updateStats(now) {
  if (now - lastStatsUpdate < 100) return;
  lastStatsUpdate = now;
  elFem.textContent = statsText(fem, smoothing.fem ?? 0, track.fem, isNeo() ? 'Newton iters' : 'CG iters');
  elPd.textContent = statsText(pd, smoothing.pd ?? 0, track.pd, isNeo() ? 'L-BFGS iters' : 'PD iters');
}

/** Shared camera rectangle that keeps both deforming bodies in view (stretch scene). */
function fitBodies() {
  let minx = 0;
  let maxx = mesh.width;
  let miny = -mesh.height / 2;
  let maxy = mesh.height / 2;
  for (const sim of [fem, pd]) {
    for (let i = 0; i < sim.n; i++) {
      const px = sim.x[2 * i];
      const py = sim.x[2 * i + 1];
      if (!Number.isFinite(px) || !Number.isFinite(py)) continue;
      minx = Math.min(minx, px);
      maxx = Math.max(maxx, px);
      miny = Math.min(miny, py);
      maxy = Math.max(maxy, py);
    }
  }
  const w = (maxx - minx) * 1.12 + 0.4;
  // extra room at the top for the statistics overlay
  const hBody = (maxy - miny) * 1.12 + 0.4;
  const h = hBody * 1.3;
  const target = { cx: (minx + maxx) / 2, cy: (miny + maxy) / 2 + 0.15 * hBody, w, h };
  if (!cameraRect) cameraRect = target;
  const k = 0.2; // smoothing
  for (const key of ['cx', 'cy', 'w', 'h']) cameraRect[key] += k * (target[key] - cameraRect[key]);
  return cameraRect;
}

// ---- main loop ----
const stage = document.getElementById('stage');
function resize() {
  renderer.setSize(stage.clientWidth, stage.clientHeight, false);
  plotForce.dirty = plotDet.dirty = true;
}
window.addEventListener('resize', resize);
gui.onOpenClose((changed) => {
  if (changed !== gui) return; // ignore folders
  document.documentElement.style.setProperty('--gui-w', changed._closed ? '0px' : '260px');
  resize();
});
if (window.innerWidth < 800) gui.close();
resize();
build();

let last = performance.now();
function frame(now) {
  requestAnimationFrame(frame);
  const real = Math.min((now - last) / 1000, 0.1);
  last = now;

  const h = settings.dt / settings.substeps;
  let steps = 0;
  if (!settings.paused) {
    accumulator += real * settings.timeScale;
    while (accumulator >= h && steps < 32) {
      accumulator -= h;
      steps++;
    }
  } else {
    accumulator = 0;
    steps = stepRequest * settings.substeps;
    stepRequest = 0;
  }
  for (let s = 0; s < steps; s++) {
    if (driver) driver.advance(h, fem, pd);
    fem.step(h);
    pd.step(h);
    smoothing.fem = ema(smoothing.fem, fem.stats.stepMs);
    smoothing.pd = ema(smoothing.pd, pd.stats.stepMs);
    updateTrack(fem, track.fem);
    updateTrack(pd, track.pd);
    if (driver && driver.time - lastSampleTime >= 0.02) {
      lastSampleTime = driver.time;
      samplePlots();
    }
  }

  viewFem.update(fem, settings.strainMax);
  viewPd.update(pd, settings.strainMax);
  viewFem.setFloor(settings.floor && !driver, settings.floorY);
  viewPd.setFloor(settings.floor && !driver, settings.floorY);
  if (driver) viewFem.viewRect = viewPd.viewRect = fitBodies();

  const W = stage.clientWidth;
  const H = stage.clientHeight;
  renderer.setScissor(0, 0, W, H);
  renderer.setViewport(0, 0, W, H);
  renderer.clear();
  const half = Math.floor(W / 2);
  for (const [view, x0, w] of [
    [viewFem, 0, half],
    [viewPd, half, W - half],
  ]) {
    fitCamera(view.camera, w / H, view.viewRect);
    renderer.setViewport(x0, 0, w, H);
    renderer.setScissor(x0, 0, w, H);
    renderer.render(view.scene, view.camera);
  }
  if (driver) {
    plotForce.draw();
    plotDet.draw();
  }
  updateStats(now);
}
requestAnimationFrame(frame);

// handy for debugging from the console
window.sim = { settings, build, get fem() { return fem; }, get pd() { return pd; }, get driver() { return driver; } };
