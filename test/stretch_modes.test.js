import test from 'node:test';
import assert from 'node:assert/strict';
import { createRectMesh } from '../src/mesh.js';
import { CorotationalFEM } from '../src/sim/corotationalFEM.js';
import { HyperelasticFEM } from '../src/sim/hyperelasticFEM.js';
import { ProjectiveDynamics } from '../src/sim/projectiveDynamics.js';
import { StretchDriver } from '../src/stretch.js';

const close = (a, b, tol, msg) => assert.ok(Math.abs(a - b) <= tol, `${msg ?? ''} ${a} vs ${b} (tol ${tol})`);
const H = 1 / 240;

function make(S, model, nu) {
  const mesh = createRectMesh(1, 1, 12, 12);
  const sim = new S(mesh, { E: 1e5, nu, density: 1000, gravity: 0, damping: 2, model });
  if (S === ProjectiveDynamics) sim.iterations = 20;
  return { mesh, sim };
}

const SOLVERS = [
  ['corotational FEM', CorotationalFEM, 'corotational'],
  ['corotational PD', ProjectiveDynamics, 'corotational'],
  ['Newton FEM', HyperelasticFEM, 'neohookean'],
  ['PD L-BFGS', ProjectiveDynamics, 'neohookean'],
];

function compress(S, model, nu, minStretch) {
  const { mesh, sim } = make(S, model, nu);
  const drv = new StretchDriver(mesh, { speed: 0.5, compress: true, minStretch });
  drv.attach(sim);
  let maxInv = 0;
  while (drv.mode === 'load') {
    drv.advance(H, sim);
    sim.step(H);
    maxInv = Math.max(maxInv, sim.deformationStats().inverted);
  }
  return { mesh, sim, drv, maxInv };
}

function release(S, model, nu, s0) {
  const { mesh, sim } = make(S, model, nu);
  const drv = new StretchDriver(mesh, { speed: 1, maxStretch: s0 });
  drv.attach(sim);
  while (drv.mode === 'load') {
    drv.advance(H, sim);
    sim.step(H);
  }
  drv.release(sim);
  let maxInv = 0;
  for (let i = 0; i < 3 / H; i++) {
    drv.advance(H, sim);
    sim.step(H);
    maxInv = Math.max(maxInv, sim.deformationStats().inverted);
  }
  return { mesh, sim, drv, maxInv };
}

test('compress driver: grips move together and the stretch ratio falls to minStretch', () => {
  const { mesh, sim, drv } = compress(CorotationalFEM, 'corotational', 0.3, 0.4);
  assert.equal(drv.mode, 'hold');
  close(drv.stretch, 0.4, 1e-9, 'prescribed');
  close(drv.currentStretch(sim), 0.4, 1e-9, 'actual');
  for (const i of mesh.rightIdx) close(sim.x[2 * i], mesh.X[2 * i] - 0.3, 1e-12);
  for (const i of mesh.leftIdx) close(sim.x[2 * i], mesh.X[2 * i] + 0.3, 1e-12);
});

test('stretch driver is unchanged by default (grips apart, ratio rises)', () => {
  const mesh = createRectMesh(1, 1, 4, 4);
  const drv = new StretchDriver(mesh, { speed: 0.5, maxStretch: 3 });
  assert.equal(drv.dMax, 1);
  drv.d = 0.5;
  close(drv.stretch, 2, 1e-12);
});

test('release frees the grips and Reset grabs them again', () => {
  const { mesh, sim, drv } = release(CorotationalFEM, 'corotational', 0.3, 2);
  assert.ok(drv.released);
  assert.equal(sim.pinned.reduce((a, b) => a + b, 0), 0);
  drv.reset(sim);
  assert.ok(!drv.released);
  assert.equal(drv.mode, 'load');
  assert.equal(sim.pinned.reduce((a, b) => a + b, 0), mesh.leftIdx.length + mesh.rightIdx.length);
  for (let i = 0; i < sim.x.length; i++) close(sim.x[i], mesh.X[i], 1e-12);
});

for (const [name, S, model] of SOLVERS) {
  test(`${name}: stays finite after compression to 15% and after release from 6x`, () => {
    const c = compress(S, model, 0.3, 0.15);
    for (const v of c.sim.x) assert.ok(Number.isFinite(v));
    const r = release(S, model, 0.3, 6);
    for (const v of r.sim.x) assert.ok(Number.isFinite(v));
    assert.equal(r.sim.deformationStats().inverted, 0, 'settles back without inverted elements');
  });
}

test('compression to 15% (nu = 0.3): Neo-Hookean inverts far fewer elements than corotational', () => {
  const rows = SOLVERS.map(([name, S, model]) => [name, model, compress(S, model, 0.3, 0.15).maxInv]);
  console.log('  inverted elements:', rows.map(([n, , k]) => `${n}=${k}`).join('  '));
  const co = rows.filter((r) => r[1] === 'corotational').map((r) => r[2]);
  const nh = rows.filter((r) => r[1] === 'neohookean').map((r) => r[2]);
  assert.ok(Math.min(...co) > 2 * Math.max(...nh), `corotational ${co} vs neo-hookean ${nh}`);
});

test('release from 3x (nu = 0.45): corotational inverts the body, Neo-Hookean does not', () => {
  const rows = SOLVERS.map(([name, S, model]) => [name, model, release(S, model, 0.45, 3).maxInv]);
  console.log('  inverted elements:', rows.map(([n, , k]) => `${n}=${k}`).join('  '));
  for (const [name, model, k] of rows) {
    if (model === 'corotational') assert.ok(k > 50, `${name}: ${k}`);
    else assert.equal(k, 0, name);
  }
});
