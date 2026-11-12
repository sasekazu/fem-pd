import * as THREE from 'three';

/** Fit an orthographic camera so that the world rectangle {cx, cy, w, h} is fully visible. */
export function fitCamera(camera, aspect, rect) {
  let halfW = rect.w / 2;
  let halfH = rect.h / 2;
  if (aspect > halfW / halfH) halfW = halfH * aspect;
  else halfH = halfW / aspect;
  camera.left = rect.cx - halfW;
  camera.right = rect.cx + halfW;
  camera.top = rect.cy + halfH;
  camera.bottom = rect.cy - halfH;
  camera.updateProjectionMatrix();
}

const C0 = new THREE.Color('#8cbff2');
const C1 = new THREE.Color('#f2c14e');
const C2 = new THREE.Color('#e8452f');
const tmp = new THREE.Color();

function strainColor(t, out) {
  t = Math.min(1, Math.max(0, t));
  if (t < 0.5) tmp.copy(C0).lerp(C1, t * 2);
  else tmp.copy(C1).lerp(C2, (t - 0.5) * 2);
  out.copy(tmp);
}

/** Draws one soft body (filled triangles coloured by strain + wireframe), an optional floor and kinematic rigid bodies. */
export class SoftBodyView {
  constructor(mesh, viewRect) {
    this.mesh = mesh;
    this.viewRect = viewRect;
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color('#14171c');
    this.camera = new THREE.OrthographicCamera(-1, 1, 1, -1, -10, 10);

    const n = mesh.n;
    this.positions = new Float32Array(3 * n);
    this.colors = new Float32Array(3 * n);
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(this.positions, 3).setUsage(THREE.DynamicDrawUsage));
    geo.setAttribute('color', new THREE.BufferAttribute(this.colors, 3).setUsage(THREE.DynamicDrawUsage));
    geo.setIndex(new THREE.BufferAttribute(mesh.tris, 1));
    this.geometry = geo;

    this.fill = new THREE.Mesh(geo, new THREE.MeshBasicMaterial({ vertexColors: true, side: THREE.DoubleSide }));
    this.wire = new THREE.Mesh(
      geo,
      new THREE.MeshBasicMaterial({ color: 0x1b2330, wireframe: true, transparent: true, opacity: 0.55 })
    );
    this.wire.position.z = 0.01;
    this.scene.add(this.fill, this.wire);

    // rest-shape outline
    const pts = [];
    for (let i = 0; i < mesh.outline.length; i += 2) pts.push(new THREE.Vector3(mesh.outline[i], mesh.outline[i + 1], -0.01));
    this.scene.add(
      new THREE.LineLoop(new THREE.BufferGeometry().setFromPoints(pts), new THREE.LineBasicMaterial({ color: 0x3a4454 }))
    );

    // pinned vertices (positions follow the solver; buffer is (re)allocated in update())
    this.pins = new THREE.Points(new THREE.BufferGeometry(), new THREE.PointsMaterial({ color: 0xffffff, size: 7, sizeAttenuation: false }));
    this._pinN = -1;
    this.scene.add(this.pins);

    // mouse grab indicator: grabbed material point -> pointer target (line + two dots)
    this.grabPos = new Float32Array(6);
    const gg = new THREE.BufferGeometry();
    gg.setAttribute('position', new THREE.BufferAttribute(this.grabPos, 3).setUsage(THREE.DynamicDrawUsage));
    this.grab = new THREE.Group();
    this.grab.add(
      new THREE.Line(gg, new THREE.LineBasicMaterial({ color: 0xffffff })),
      new THREE.Points(gg, new THREE.PointsMaterial({ color: 0xffffff, size: 9, sizeAttenuation: false }))
    );
    for (const o of this.grab.children) o.frustumCulled = false; // positions change every frame
    this.grab.visible = false;
    this.scene.add(this.grab);

    // floor: a dark slab below y = floorY and a bright line on its top edge
    this.floor = new THREE.Group();
    const slab = new THREE.Mesh(new THREE.PlaneGeometry(60, 30), new THREE.MeshBasicMaterial({ color: 0x232a36 }));
    slab.position.set(0, -15, -0.02);
    const line = new THREE.Line(
      new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(-30, 0, 0.03), new THREE.Vector3(30, 0, 0.03)]),
      new THREE.LineBasicMaterial({ color: 0x9fb0c8 })
    );
    this.floor.add(slab, line);
    this.scene.add(this.floor);

    // kinematic rigid bodies (Polygon colliders): one group per body, posed in update()
    this.rigidGroup = new THREE.Group();
    this.scene.add(this.rigidGroup);
    this.rigid = [];
  }

  /** Show the given Polygon / Circle colliders (replaces the previous ones). Their pose is read every update(). */
  setRigid(bodies) {
    for (const r of this.rigid) {
      this.rigidGroup.remove(r.group);
      r.group.traverse((o) => {
        o.geometry?.dispose();
        o.material?.dispose();
      });
    }
    this.rigid = bodies.map((body) => {
      let geometry;
      const pts = [];
      if (body.radius !== undefined) {
        // disc: filled circle plus an outline with a radius mark so that rotation stays visible
        geometry = new THREE.CircleGeometry(body.radius, 64);
        for (let i = 0; i < 64; i++) {
          const a = (2 * Math.PI * i) / 64;
          pts.push(new THREE.Vector3(body.radius * Math.cos(a), body.radius * Math.sin(a), 0.001));
        }
      } else {
        const v = body.vertices;
        const shape = new THREE.Shape();
        shape.moveTo(v[0], v[1]);
        for (let i = 1; i < body.n; i++) shape.lineTo(v[2 * i], v[2 * i + 1]);
        shape.closePath();
        geometry = new THREE.ShapeGeometry(shape);
        for (let i = 0; i < body.n; i++) pts.push(new THREE.Vector3(v[2 * i], v[2 * i + 1], 0.001));
      }
      const fill = new THREE.Mesh(
        geometry,
        new THREE.MeshBasicMaterial({ color: 0x66758f, transparent: true, opacity: 0.82, side: THREE.DoubleSide })
      );
      const outline = new THREE.LineLoop(new THREE.BufferGeometry().setFromPoints(pts), new THREE.LineBasicMaterial({ color: 0xd3dcec }));
      const group = new THREE.Group();
      group.add(fill, outline);
      if (body.radius !== undefined) {
        const mark = new THREE.Line(
          new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(0, 0, 0.002), new THREE.Vector3(body.radius, 0, 0.002)]),
          new THREE.LineBasicMaterial({ color: 0xd3dcec })
        );
        group.add(mark);
      }
      this.rigidGroup.add(group);
      return { body, group };
    });
    this.syncRigid();
  }

  syncRigid() {
    for (const { body, group } of this.rigid) {
      group.position.set(body.x, body.y, 0.015);
      group.rotation.z = body.angle;
    }
  }

  set wireframe(v) {
    this.wire.visible = v;
  }

  setFloor(visible, y) {
    this.floor.visible = visible;
    this.floor.position.y = y;
  }

  /** Show the grab indicator between the grabbed point `p` of this body and the pointer `target` (null hides it). */
  setGrab(p, target) {
    this.grab.visible = !!p;
    if (!p) return;
    this.grabPos.set([p[0], p[1], 0.03, target[0], target[1], 0.03]);
    this.grab.children[0].geometry.attributes.position.needsUpdate = true;
  }

  update(sim, strainMax) {
    this.syncRigid();
    const x = sim.x;
    const strain = sim.updateVertexStrain();
    const c = new THREE.Color();
    for (let i = 0; i < this.mesh.n; i++) {
      this.positions[3 * i] = x[2 * i];
      this.positions[3 * i + 1] = x[2 * i + 1];
      strainColor(strain[i] / strainMax, c);
      this.colors[3 * i] = c.r;
      this.colors[3 * i + 1] = c.g;
      this.colors[3 * i + 2] = c.b;
    }
    this.geometry.attributes.position.needsUpdate = true;
    this.geometry.attributes.color.needsUpdate = true;

    const pinIdx = sim.pinnedList;
    if (this._pinN !== pinIdx.length) {
      this.pins.geometry.dispose();
      this.pins.geometry = new THREE.BufferGeometry();
      this.pins.geometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array(3 * pinIdx.length), 3));
      this._pinN = pinIdx.length;
    }
    const pa = this.pins.geometry.attributes.position;
    pinIdx.forEach((i, k) => pa.setXYZ(k, x[2 * i], x[2 * i + 1], 0.02));
    pa.needsUpdate = true;
  }

  dispose() {
    this.geometry.dispose();
    this.fill.material.dispose();
    this.wire.material.dispose();
    this.grab.children[0].geometry.dispose();
    this.setRigid([]);
  }
}
