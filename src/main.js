import * as THREE from 'three';
import GUI from 'lil-gui';
import { createRectMesh, createDiscMesh, createBlockMesh } from './mesh.js';
import { CorotationalFEM } from './sim/corotationalFEM.js';
import { ProjectiveDynamics } from './sim/projectiveDynamics.js';
import { DEFAULT_PARAMS } from './sim/base.js';
import { SoftBodyView, fitCamera } from './render.js';
import { StretchDriver } from './stretch.js';
import { LinePlot } from './plot.js';
import { MouseGrabber, viewportToWorld } from './interaction.js';
import { Polygon, boxVertices, regularVertices, starVertices, lShapeVertices } from './rigid.js';

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
    substeps: 1, pdIterations: 10, floor: false, floorY: -1.6, contactK: 1e6, strainMax: 0.15, mouseK: 1500,
    view: { cx: 2.2, cy: -0.9, w: 5.6, h: 4.2 },
  },
  ball: {
    E: 1e6, nu: 0.3, density: 1000, gravity: 9.8, damping: 0,
    substeps: 4, pdIterations: 20, floor: true, floorY: 0, contactK: 1e6, strainMax: 0.15,
    mouseK: 8000, // heavy body: the mouse spring is mass-scaled per vertex, so it needs to be stiffer to lift it
    view: { cx: 0, cy: 0.95, w: 3.6, h: 3.0 },
  },
  rigid: {
    E: 1e6, nu: 0.3, density: 1000, gravity: 9.8, damping: 0.5,
    width: 2, height: 1, nx: 24, ny: 12,
    // contact k = 1e7: the mass-scaled penalty must be stiff compared with the material (see README)
    substeps: 4, pdIterations: 20, floor: true, floorY: 0, contactK: 1e7, strainMax: 0.3, mouseK: 8000,
    view: { cx: 0, cy: 1.15, w: 4.4, h: 3.2 },
  },
  stretch: {
    E: 1e5, nu: 0.3, density: 1000, gravity: 0, damping: 2,
    width: 1, height: 1, nx: 16, ny: 16,
    substeps: 2, pdIterations: 20, floor: false, floorY: -1.6, contactK: 1e6, strainMax: 2, mouseK: 1500,
    view: null, // fitted to the bodies every frame
  },
};

// Kinematic rigid shapes (body frame, lengths in m).
const RIGID_SHAPES = {
  wedge: () => regularVertices(3, 0.5, -Math.PI / 2), // sharp tip pointing down
  box: () => boxVertices(0.9, 0.5),
  star: () => starVertices(5, 0.55, 0.24, -Math.PI / 2),
  'L-shape (concave)': () => lShapeVertices(0.9, 0.8, 0.45, 0.4),
};
const RIGID_CLEARANCE = 0.25; // gap between the lowest point of the rigid body and the block top at its home pose
const RIGID_MAX_SPEED = 2; // [m/s] kinematic body follows its target at most this fast (avoids tunnelling)
const RIGID_MAX_OMEGA = 4; // [rad/s]

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
  rigidShape: 'wedge',
  rigidPinBottom: false, // pin the bottom edge of the block
  rigidAuto: true, // rigid body presses into the block along a scripted path
  rigidDepth: 0.4, // how far the lowest point goes below the block top [m]
  rigidPeriod: 6, // period of the scripted press [s]
  rigidAngle: 0, // body rotation [deg]
  mouse: true, // drag the body with the mouse (not in the stretch scene)
  mouseK: 1500, // mouse spring stiffness [1/s²] (capped for stability, see interaction.js)
};
Object.assign(settings, SCENES[settings.scene]);

let mesh, fem, pd, viewFem, viewPd, driver = null;
const grabber = new MouseGrabber({ stiffness: settings.mouseK });
let dragView = -1; // 0 = FEM half, 1 = PD half (the half the drag started in)
let accumulator = 0;
let stepRequest = 0;
const smoothing = { fem: null, pd: null };
const track = { fem: null, pd: null };
let y0 = 0; // initial centre-of-mass height (ball scene)
let rigidBody = null; // Polygon collider shared by both solvers (rigid scene)
const rigid = { x: 0, y: 0, angle: 0, tx: 0, ty: 0, tAngle: 0, time: 0, drag: null, lowest: 0, topY: 0 };
let cameraRect = null; // smoothed shared view rectangle (stretch scene)

function materialParams() {
  const { E, nu, density, gravity, damping } = settings;
  return { E, nu, density, gravity, damping };
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
  grabber.release();
  grabber.stiffness = settings.mouseK;
  dragView = -1;
  viewFem?.dispose();
  viewPd?.dispose();
  const ball = settings.scene === 'ball';
  const stretch = settings.scene === 'stretch';
  const hasRigid = settings.scene === 'rigid';
  const { width, height } = SCENES[settings.scene];
  mesh = ball
    ? createDiscMesh(BALL_RADIUS, settings.rings, [0, settings.floorY + BALL_RADIUS + settings.gap])
    : hasRigid
      ? createBlockMesh(width, height, settings.nx, settings.ny, [0, settings.floorY], settings.rigidPinBottom)
      : createRectMesh(width, height, settings.nx, settings.ny);
  fem = new CorotationalFEM(mesh, materialParams());
  pd = new ProjectiveDynamics(mesh, materialParams());
  fem.setContact(contactParams());
  pd.setContact(contactParams());
  rigidBody = hasRigid ? new Polygon(RIGID_SHAPES[settings.rigidShape]()) : null;
  fem.setColliders(rigidBody ? [rigidBody] : []);
  pd.setColliders(rigidBody ? [rigidBody] : []);
  if (rigidBody) {
    rigid.topY = settings.floorY + height;
    resetRigid();
  }
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
  if (rigidBody) {
    viewFem.setRigid([rigidBody]);
    viewPd.setRigid([rigidBody]);
  }
  y0 = fem.centerOfMass().y;
  accumulator = 0;
  smoothing.fem = smoothing.pd = null;
  cameraRect = null;
  resetTrack();
  clearPlots();
  document.documentElement.style.setProperty('--plot-h', stretch ? `${PLOT_H}px` : '0px');
  resize();
  updateLegend();
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

// ---- kinematic rigid body: scripted press and/or mouse drag, speed-limited ----
/** Lowest point of the body (body frame rotated by `angle`) relative to its origin. */
function rigidLowest(angle) {
  const c = Math.cos(angle);
  const s = Math.sin(angle);
  let low = Infinity;
  for (let i = 0; i < rigidBody.n; i++) low = Math.min(low, s * rigidBody.vertices[2 * i] + c * rigidBody.vertices[2 * i + 1]);
  return low;
}

/** y of the body origin when its lowest point is `depth` below the block top (negative depth: above). */
function rigidYAt(depth, angle) {
  return rigid.topY - depth - rigidLowest(angle);
}

function resetRigid() {
  if (!rigidBody) return;
  rigid.time = 0;
  rigid.drag = null;
  rigid.angle = rigid.tAngle = (settings.rigidAngle * Math.PI) / 180;
  rigid.x = rigid.tx = 0;
  rigid.y = rigid.ty = rigidYAt(-RIGID_CLEARANCE, rigid.angle);
  rigidBody.setPose(rigid.x, rigid.y, rigid.angle);
}

/** Advance the target (scripted path unless dragging) and move the pose towards it by at most the speed limits. */
function advanceRigid(h) {
  if (!rigidBody) return;
  rigid.tAngle = rigid.drag ? rigid.tAngle : (settings.rigidAngle * Math.PI) / 180;
  if (settings.rigidAuto && !rigid.drag) {
    rigid.time += h;
    const phase = 0.5 - 0.5 * Math.cos((2 * Math.PI * rigid.time) / settings.rigidPeriod); // 0 -> 1 -> 0
    const depth = -RIGID_CLEARANCE + phase * (settings.rigidDepth + RIGID_CLEARANCE);
    rigid.tx = 0;
    rigid.ty = rigidYAt(depth, rigid.tAngle);
  }
  const dx = rigid.tx - rigid.x;
  const dy = rigid.ty - rigid.y;
  const dist = Math.hypot(dx, dy);
  const step = Math.min(1, (RIGID_MAX_SPEED * h) / (dist || 1));
  rigid.x += dx * step;
  rigid.y += dy * step;
  const da = rigid.tAngle - rigid.angle;
  rigid.angle += Math.max(-RIGID_MAX_OMEGA * h, Math.min(RIGID_MAX_OMEGA * h, da));
  rigidBody.setPose(rigid.x, rigid.y, rigid.angle);
}

function reset() {
  resetRigid();
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
  .add(settings, 'scene', { 'Cantilever beam': 'beam', 'Bouncing ball': 'ball', 'Rigid contact': 'rigid', 'Stretch test': 'stretch' })
  .name('scene')
  .onChange((name) => {
    Object.assign(settings, SCENES[name]);
    gui.controllersRecursive().forEach((c) => c.updateDisplay());
    updateSceneControls();
    build();
  });

const fMat = gui.addFolder('Material / world');
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
    if (settings.scene === 'ball' || settings.scene === 'rigid') build();
  });
fContact
  .add(settings, 'contactK', { '1e5 (soft)': 1e5, '1e6': 1e6, '1e7': 1e7, '1e8 (hard)': 1e8 })
  .name('stiffness k [1/s²]')
  .onChange(pushContact);
fContact.add(settings, 'lbfgs', 0, 12, 1).name('PD L-BFGS history').onChange(applySolverSettings);

const fRigid = gui.addFolder('Rigid body (kinematic)');
fRigid.add(settings, 'rigidShape', Object.keys(RIGID_SHAPES)).name('shape').onChange(build);
fRigid.add(settings, 'rigidPinBottom').name('pin block bottom').onChange(build);
const cAuto = fRigid.add(settings, 'rigidAuto').name('scripted press');
fRigid.add(settings, 'rigidDepth', 0, 0.9, 0.05).name('press depth [m]');
fRigid.add(settings, 'rigidPeriod', 2, 20, 0.5).name('period [s]');
fRigid.add(settings, 'rigidAngle', -180, 180, 1).name('angle [deg]');

const fMouse = gui.addFolder('Mouse');
fMouse
  .add(settings, 'mouse')
  .name('drag with mouse')
  .onChange((v) => {
    if (!v) releaseGrab();
  });
fMouse
  .add(settings, 'mouseK', 100, 30000, 100)
  .name('spring stiffness [1/s²]')
  .onChange((v) => (grabber.stiffness = v));

const fSim = gui.addFolder('Simulation');
fSim.add(settings, 'dt', { '1/240 s': 1 / 240, '1/120 s': 1 / 120, '1/60 s': 1 / 60, '1/30 s': 1 / 30 }).name('time step dt');
fSim.add(settings, 'substeps', 1, 8, 1).name('substeps');
fSim.add(settings, 'pdIterations', 1, 100, 1).name('PD iterations').onChange(applySolverSettings);
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
  s === 'rigid' ? fRigid.show() : fRigid.hide();
  s === 'stretch' ? fStretch.show() : fStretch.hide();
  s === 'stretch' ? fContact.hide() : fContact.show();
  s === 'stretch' ? fMouse.hide() : fMouse.show();
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
  if (sim.stats.lineSearch !== undefined && settings.floor) s += row('backtracks', sim.stats.lineSearch);
  if (scene === 'ball') {
    const yRest = settings.floorY + BALL_RADIUS;
    s += row('centre y', sim.centerOfMass().y.toFixed(4));
    if (tr.apexes.length) {
      const rebound = (100 * (tr.apexes[0] - yRest)) / (y0 - yRest);
      s += row('rebound', `${rebound.toFixed(1)} %`);
    } else s += row('rebound', '-');
  } else if (scene === 'rigid') {
    const ci = settings.floor ? 1 : 0; // collider list: floor (if on) then the rigid body
    const f = sim.contact.colliderForces(sim.x, sim.mass, sim.n)[ci];
    s += row('force on rigid', `${(-f.fy).toFixed(0)} N`);
    s += row('contacts', f.count);
    s += row('max pen.', `${(1000 * sim.contact.maxPenetration(sim.x, sim.n, ci)).toFixed(2)} mm`);
    s += row('min det F', sim.deformationStats().minDet.toFixed(3));
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
  elFem.textContent = statsText(fem, smoothing.fem ?? 0, track.fem, 'CG iters');
  elPd.textContent = statsText(pd, smoothing.pd ?? 0, track.pd, 'PD iters');
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

// ---- mouse interaction: grab a point of the body and drag it with a spring ----
// The grab is picked in the half (FEM / PD) that was clicked, but the same material point and the same
// world-space target are applied to both solvers.
const mouseEnabled = () => settings.mouse && settings.scene !== 'stretch';

function pointerWorld(e, which) {
  const r = canvas.getBoundingClientRect();
  const half = Math.floor(r.width / 2);
  const x0 = which === 0 ? 0 : half;
  const w = which === 0 ? half : r.width - half;
  const view = which === 0 ? viewFem : viewPd;
  return viewportToWorld(view.camera, (e.clientX - r.left - x0) / w, (e.clientY - r.top) / r.height);
}

function releaseGrab() {
  grabber.release();
  rigid.drag = null;
  dragView = -1;
}

canvas.addEventListener('pointerdown', (e) => {
  if (e.button !== 0 || !mouseEnabled()) return;
  const r = canvas.getBoundingClientRect();
  const which = e.clientX - r.left < Math.floor(r.width / 2) ? 0 : 1;
  const [wx, wy] = pointerWorld(e, which);
  if (rigidBody?.contains(wx, wy)) {
    // drag the kinematic rigid body (the scripted press is switched off)
    rigid.drag = { dx: rigid.x - wx, dy: rigid.y - wy };
    rigid.tx = rigid.x;
    rigid.ty = rigid.y;
    dragView = which;
    settings.rigidAuto = false;
    cAuto.updateDisplay();
    canvas.setPointerCapture(e.pointerId);
    e.preventDefault();
    return;
  }
  if (!grabber.grab(which === 0 ? fem : pd, wx, wy)) return;
  dragView = which;
  canvas.setPointerCapture(e.pointerId);
  e.preventDefault();
});
canvas.addEventListener('pointermove', (e) => {
  if (dragView < 0) return;
  const [wx, wy] = pointerWorld(e, dragView);
  if (rigid.drag) {
    rigid.tx = wx + rigid.drag.dx;
    rigid.ty = wy + rigid.drag.dy;
  } else grabber.setTarget(wx, wy);
});
for (const type of ['pointerup', 'pointercancel']) canvas.addEventListener(type, releaseGrab);

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
    advanceRigid(h);
    grabber.applyForces(fem, h);
    grabber.applyForces(pd, h);
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
  viewFem.setGrab(grabber.active ? grabber.point(fem) : null, grabber.target);
  viewPd.setGrab(grabber.active ? grabber.point(pd) : null, grabber.target);
  const cursor = !mouseEnabled() ? 'default' : grabber.active ? 'grabbing' : 'grab';
  if (canvas.style.cursor !== cursor) canvas.style.cursor = cursor;
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
window.sim = { settings, build, grabber, rigid, get rigidBody() { return rigidBody; }, get fem() { return fem; }, get pd() { return pd; }, get driver() { return driver; } };
