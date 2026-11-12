import test from 'node:test';
import assert from 'node:assert/strict';
import { createRectMesh, createDiscMesh } from '../src/mesh.js';
import { CorotationalFEM } from '../src/sim/corotationalFEM.js';
import { ProjectiveDynamics } from '../src/sim/projectiveDynamics.js';
import { MouseGrabber, viewportToWorld, locateTriangle, effectiveStiffness } from '../src/interaction.js';

const close = (a, b, tol, msg) => assert.ok(Math.abs(a - b) <= tol, `${msg ?? ''} ${a} vs ${b} (tol ${tol})`);

test('viewportToWorld maps the corners and centre of an orthographic frustum', () => {
  const cam = { left: -2, right: 6, top: 3, bottom: -1 };
  assert.deepEqual(viewportToWorld(cam, 0, 0), [-2, 3]);
  assert.deepEqual(viewportToWorld(cam, 1, 1), [6, -1]);
  assert.deepEqual(viewportToWorld(cam, 0.5, 0.5), [2, 1]);
});

test('locateTriangle: weights sum to 1 and reproduce the point; far points miss', () => {
  const mesh = createRectMesh(4, 1, 8, 4);
  const x = Float64Array.from(mesh.X);
  for (const [px, py] of [[0.3, 0.1], [2.01, -0.2], [3.9, 0.45]]) {
    const hit = locateTriangle(mesh, x, px, py);
    assert.ok(hit, `hit at ${px},${py}`);
    close(hit.w[0] + hit.w[1] + hit.w[2], 1, 1e-12);
    let qx = 0;
    let qy = 0;
    hit.w.forEach((w, k) => {
      const i = mesh.tris[3 * hit.tri + k];
      qx += w * x[2 * i];
      qy += w * x[2 * i + 1];
    });
    close(qx, px, 1e-12);
    close(qy, py, 1e-12);
  }
  assert.equal(locateTriangle(mesh, x, 2, 3), null);
  assert.equal(locateTriangle(mesh, x, -1, 0), null);
});

test('effectiveStiffness caps k at 1/h^2', () => {
  assert.equal(effectiveStiffness(1500, 1 / 60), 1500);
  close(effectiveStiffness(1e9, 1 / 60), 3600, 1e-9);
});

const params = { E: 5e6, nu: 0.3, density: 1000, gravity: 9.8, damping: 1 };
const sims = () => {
  const mesh = createRectMesh(4, 1, 16, 4);
  return [new CorotationalFEM(mesh, params), new ProjectiveDynamics(mesh, params)];
};

test('applyForces: mass-weighted spring force toward the target, cleared on release', () => {
  const [fem] = sims();
  const g = new MouseGrabber({ stiffness: 1000, damping: 0 });
  assert.ok(g.grab(fem, 3.5, 0.1));
  g.setTarget(3.5, 0.6); // 0.5 m above the grabbed point
  g.applyForces(fem, 1 / 60);
  let fy = 0;
  let fx = 0;
  let expect = 0;
  for (let k = 0; k < 3; k++) {
    const i = fem.mesh.tris[3 * g.tri + k];
    close(fem.fext[2 * i + 1], fem.mass[i] * g.w[k] * 1000 * 0.5, 1e-6, 'force on vertex');
    expect += fem.mass[i] * g.w[k] * 500;
  }
  for (let i = 0; i < fem.n; i++) {
    fx += fem.fext[2 * i];
    fy += fem.fext[2 * i + 1];
  }
  close(fx, 0, 1e-9);
  close(fy, expect, 1e-6);
  g.release();
  g.applyForces(fem, 1 / 60);
  assert.ok(fem.fext.every((v) => v === 0));
});

test('applyForces: distance clamp and damping', () => {
  const [fem] = sims();
  const g = new MouseGrabber({ stiffness: 1000, damping: 0, maxDist: 1 });
  g.grab(fem, 3.5, 0.1);
  g.setTarget(3.5, 100);
  g.applyForces(fem, 1 / 60);
  const i = fem.mesh.tris[3 * g.tri];
  close(fem.fext[2 * i + 1], fem.mass[i] * g.w[0] * 1000 * 1, 1e-6, 'clamped to maxDist');
  // damping opposes the velocity of the grabbed point
  const g2 = new MouseGrabber({ stiffness: 1000, damping: 1 });
  g2.grab(fem, 3.5, 0.1);
  g2.setTarget(...g2.point(fem)); // zero spring extension
  for (let k = 0; k < 3; k++) fem.v[2 * fem.mesh.tris[3 * g2.tri + k] + 1] = 1;
  g2.applyForces(fem, 1 / 60);
  assert.ok(fem.fext[2 * i + 1] < 0);
});

test('dragging the beam tip up: FEM and PD respond alike, stay finite, and relax when released', () => {
  const [fem, pd] = sims();
  pd.iterations = 20;
  const g = new MouseGrabber({ stiffness: 1500 });
  assert.ok(g.grab(fem, 3.9, 0.0));
  const [gx, gy] = g.point(fem);
  g.setTarget(gx, gy + 1.0);
  const h = 1 / 60;
  const free = sims(); // reference run without the mouse
  for (let s = 0; s < 120; s++) {
    for (const [sim, ref] of [[fem, free[0]], [pd, free[1]]]) {
      g.applyForces(sim, h);
      sim.step(h);
      ref.step(h);
    }
  }
  const tip = (s) => s.x[2 * s.mesh.tipIdx + 1];
  for (const [sim, ref] of [[fem, free[0]], [pd, free[1]]]) {
    assert.ok(Number.isFinite(tip(sim)));
    assert.ok(tip(sim) > tip(ref) + 0.3, `tip raised by the mouse (${tip(sim)} vs ${tip(ref)})`);
  }
  close(tip(fem), tip(pd), 0.1 * Math.abs(tip(fem) - tip(free[0])) + 0.02, 'FEM ~ PD with the same mouse');
  // release: the beam falls back toward the unforced state
  g.release();
  for (let s = 0; s < 1200; s++) {
    for (const sim of [fem, pd, ...free]) {
      g.applyForces(sim, h);
      sim.step(h);
    }
  }
  for (const [sim, ref] of [[fem, free[0]], [pd, free[1]]]) close(tip(sim), tip(ref), 0.05, 'relaxed after release');
});

test('grabbing the ball and lifting it works with the floor contact too', () => {
  const mesh = createDiscMesh(0.5, 6, [0, 0.5]);
  const sim = new ProjectiveDynamics(mesh, { ...params, E: 1e6, damping: 0 });
  sim.iterations = 20;
  sim.setContact({ enabled: true, floorY: 0, k: 1e6 });
  const g = new MouseGrabber({ stiffness: 1500 });
  assert.ok(g.grab(sim, 0, 0.5));
  g.setTarget(0, 1.5);
  for (let s = 0; s < 120; s++) {
    g.applyForces(sim, 1 / 120);
    sim.step(1 / 120);
  }
  assert.ok(Number.isFinite(sim.centerOfMass().y));
  assert.ok(sim.centerOfMass().y > 0.7, `ball lifted (${sim.centerOfMass().y})`);
});
