import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { LineMaterial } from "three/examples/jsm/lines/LineMaterial.js";
import { LineSegments2 } from "three/examples/jsm/lines/LineSegments2.js";
import { LineSegmentsGeometry } from "three/examples/jsm/lines/LineSegmentsGeometry.js";

export type ViewName = "front" | "top" | "side" | "iso" | "fit";
export type ClipAxis = "off" | "x" | "y" | "z";
export type ViewMode = "original" | "result" | "overlay";

const VIEW_DIRECTIONS: Record<Exclude<ViewName, "fit">, THREE.Vector3> = {
  front: new THREE.Vector3(0, -1, 0),
  top: new THREE.Vector3(0, -0.0001, 1),
  side: new THREE.Vector3(1, 0, 0),
  iso: new THREE.Vector3(1, -1, 0.8),
};
const AXIS_INDEX = { x: 0, y: 1, z: 2 } as const;
const NO_CLIP = 1e9;

/** 3D viewport. Z is up (build direction), units are millimetres. */
export class Viewport {
  readonly scene = new THREE.Scene();
  readonly camera: THREE.PerspectiveCamera;
  readonly renderer: THREE.WebGLRenderer;
  readonly controls: OrbitControls;

  /** Box the view buttons frame: the loaded model's bounds, or a default. */
  private focusBox = new THREE.Box3(new THREE.Vector3(-60, -60, 0), new THREE.Vector3(60, 60, 80));
  private readonly grid: THREE.GridHelper;
  private readonly clipPlane = new THREE.Plane(new THREE.Vector3(-1, 0, 0), NO_CLIP);
  private readonly outsideMaterial = new THREE.MeshStandardMaterial({
    polygonOffset: true, polygonOffsetFactor: 1, polygonOffsetUnits: 1,
    color: 0xaeb9c4, roughness: 0.65, metalness: 0.05, flatShading: true,
    side: THREE.FrontSide, clippingPlanes: [this.clipPlane],
  });
  /** Back faces are only visible through a clipping cut: drawn in a contrasting colour. */
  private readonly insideMaterial = new THREE.MeshBasicMaterial({
    color: 0xc7682a, side: THREE.BackSide, clippingPlanes: [this.clipPlane],
  });
  private readonly resultMaterial = new THREE.MeshStandardMaterial({
    polygonOffset: true, polygonOffsetFactor: 1, polygonOffsetUnits: 1,
    color: 0xe3c88f, roughness: 0.6, metalness: 0.05, flatShading: true,
    side: THREE.FrontSide, clippingPlanes: [this.clipPlane],
  });
  /** The original drawn as a transparent ghost over the result. */
  private readonly ghostMaterial = new THREE.MeshStandardMaterial({
    color: 0x7fb4ff, roughness: 0.8, flatShading: true, transparent: true, opacity: 0.22,
    depthWrite: false, side: THREE.FrontSide, clippingPlanes: [this.clipPlane],
    // Drawn just in front of the result where the two surfaces coincide (no flicker).
    polygonOffset: true, polygonOffsetFactor: -2, polygonOffsetUnits: -2,
  });
  private readonly overhang = { value: 0 };
  private readonly bedZ = { value: 0 };
  private readonly modelGroup = new THREE.Group();
  private readonly resultGroup = new THREE.Group();
  private viewMode: ViewMode = "original";
  private readonly boxHelper = new THREE.Box3Helper(new THREE.Box3(), 0x4cc2a5);
  private readonly labels: HTMLElement[] = [];
  private hasModel = false;
  private needsRender = true;
  private edgeLines: LineSegments2 | null = null;
  private edgeOfSegment: Uint32Array = new Uint32Array(0);
  private readonly edgeMaterial = new LineMaterial({ vertexColors: true, linewidth: 3, worldUnits: false });
  private readonly raycaster = new THREE.Raycaster();

  constructor(private readonly container: HTMLElement) {
    THREE.Object3D.DEFAULT_UP.set(0, 0, 1);
    this.camera = new THREE.PerspectiveCamera(40, 1, 0.1, 10000);
    this.camera.up.set(0, 0, 1);

    this.renderer = new THREE.WebGLRenderer({ antialias: true });
    this.renderer.setPixelRatio(window.devicePixelRatio);
    this.renderer.localClippingEnabled = true;
    this.scene.background = new THREE.Color(0x15181c);
    container.appendChild(this.renderer.domElement);

    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.12;

    this.scene.add(new THREE.HemisphereLight(0xffffff, 0x8f959d, 1.6));
    const key = new THREE.DirectionalLight(0xffffff, 1.4);
    key.position.set(150, -200, 300);
    this.scene.add(key);
    const fill = new THREE.DirectionalLight(0xffffff, 0.5);
    fill.position.set(-200, 150, 100);
    this.scene.add(fill);

    // Print bed: 250 x 250 mm, 10 mm squares, in the XY plane.
    this.grid = new THREE.GridHelper(250, 25, 0x56606d, 0x2b313a);
    this.grid.rotation.x = Math.PI / 2;
    this.scene.add(this.grid);
    this.scene.add(new THREE.AxesHelper(30));

    this.addOverhangHighlight(this.outsideMaterial);
    this.addOverhangHighlight(this.resultMaterial);
    this.boxHelper.visible = false;
    this.scene.add(this.modelGroup, this.resultGroup, this.boxHelper);
    for (let i = 0; i < 3; i++) {
      const label = document.createElement("div");
      label.className = "dimension-label";
      label.hidden = true;
      container.appendChild(label);
      this.labels.push(label);
    }

    new ResizeObserver(() => this.resize()).observe(container);
    this.resize();
    this.setView("iso");
    // Redraw only when the camera moved or something changed: large results would
    // otherwise be re-rendered sixty times a second for nothing.
    this.renderer.setAnimationLoop(() => {
      const moved = this.controls.update();
      if (!moved && !this.needsRender) return;
      this.needsRender = false;
      this.renderer.render(this.scene, this.camera);
      this.updateLabels();
    });
  }

  /** Show a model (replacing the previous one) and frame it. */
  setModel(geometry: THREE.BufferGeometry): void {
    this.needsRender = true;
    this.clearModel();
    this.clearResult();
    // STL normals are often wrong or missing: use flat per-face shading instead.
    geometry.deleteAttribute("normal");
    geometry.computeBoundingBox();
    this.modelGroup.add(
      new THREE.Mesh(geometry, this.outsideMaterial),
      new THREE.Mesh(geometry, this.insideMaterial),
    );
    this.focusBox = geometry.boundingBox!.clone();
    this.boxHelper.box.copy(this.focusBox);
    this.boxHelper.visible = true;
    this.hasModel = true;
    const center = this.focusBox.getCenter(new THREE.Vector3());
    this.grid.position.set(center.x, center.y, this.focusBox.min.z);
    this.bedZ.value = this.focusBox.min.z;
    this.applyViewMode();
    this.setView("iso");
  }

  /** Show a generated result (indexed triangles), replacing the previous one. */
  setResult(positions: Float32Array, indices: Uint32Array): void {
    this.needsRender = true;
    this.clearResult();
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    geometry.setIndex(new THREE.BufferAttribute(indices, 1));
    this.resultGroup.add(
      new THREE.Mesh(geometry, this.resultMaterial),
      new THREE.Mesh(geometry, this.insideMaterial),
    );
    this.applyViewMode();
  }

  clearResult(): void {
    this.needsRender = true;
    for (const child of [...this.resultGroup.children]) {
      (child as THREE.Mesh).geometry.dispose();
      this.resultGroup.remove(child);
    }
    this.applyViewMode();
  }

  get hasResult(): boolean {
    return this.resultGroup.children.length > 0;
  }

  setViewMode(mode: ViewMode): void {
    this.needsRender = true;
    this.viewMode = mode;
    this.applyViewMode();
  }

  /** Without a result the original is always shown. */
  private applyViewMode(): void {
    const mode = this.hasResult ? this.viewMode : "original";
    this.resultGroup.visible = mode !== "original";
    this.modelGroup.visible = mode !== "result";
    const [outside, inside] = this.modelGroup.children as THREE.Mesh[];
    if (outside) {
      outside.material = mode === "overlay" ? this.ghostMaterial : this.outsideMaterial;
      outside.renderOrder = mode === "overlay" ? 1 : 0;
      inside.visible = mode !== "overlay";
    }
  }

  /** Draw the sharp edges of the model. `kept` edges are orange, the others blue. */
  setEdges(positions: Float32Array, edgeOfSegment: Uint32Array, kept: number[]): void {
    this.needsRender = true;
    this.clearEdges();
    if (edgeOfSegment.length === 0) return;
    const geometry = new LineSegmentsGeometry();
    geometry.setPositions(positions);
    this.edgeOfSegment = edgeOfSegment;
    this.edgeLines = new LineSegments2(geometry, this.edgeMaterial);
    this.edgeLines.renderOrder = 2;
    this.scene.add(this.edgeLines);
    this.setKeptEdges(kept);
  }

  setKeptEdges(kept: number[]): void {
    this.needsRender = true;
    if (!this.edgeLines) return;
    const keep = new Set(kept);
    const colors = new Float32Array(this.edgeOfSegment.length * 6);
    this.edgeOfSegment.forEach((edge, i) => {
      const [r, g, b] = keep.has(edge) ? [1.0, 0.55, 0.1] : [0.25, 0.6, 1.0];
      colors.set([r, g, b, r, g, b], i * 6);
    });
    (this.edgeLines.geometry as LineSegmentsGeometry).setColors(colors);
  }

  showEdges(on: boolean): void {
    this.needsRender = true;
    if (this.edgeLines) this.edgeLines.visible = on;
  }

  /** The model edge under the mouse (client coordinates), or null. */
  pickEdge(clientX: number, clientY: number): number | null {
    if (!this.edgeLines?.visible) return null;
    const rect = this.renderer.domElement.getBoundingClientRect();
    const pointer = new THREE.Vector2(
      ((clientX - rect.left) / rect.width) * 2 - 1,
      -((clientY - rect.top) / rect.height) * 2 + 1,
    );
    this.raycaster.setFromCamera(pointer, this.camera);
    this.raycaster.params.Line2 = { threshold: 8 }; // pixels either side of the line
    const hits = this.raycaster.intersectObject(this.edgeLines);
    if (!hits.length || hits[0].faceIndex === undefined) return null;
    return this.edgeOfSegment[hits[0].faceIndex!] ?? null;
  }

  private clearEdges(): void {
    if (this.edgeLines) {
      this.scene.remove(this.edgeLines);
      this.edgeLines.geometry.dispose();
      this.edgeLines = null;
    }
    this.edgeOfSegment = new Uint32Array(0);
  }

  clearModel(): void {
    this.needsRender = true;
    this.clearEdges();
    const first = this.modelGroup.children[0] as THREE.Mesh | undefined;
    first?.geometry.dispose();
    this.modelGroup.clear();
    this.boxHelper.visible = false;
    this.hasModel = false;
  }

  /** Cut the model open. `fraction` (0..1) is the cut position across the bounding box. */
  setClip(axis: ClipAxis, fraction: number, flip: boolean): void {
    this.needsRender = true;
    if (axis === "off") {
      this.clipPlane.constant = NO_CLIP;
      return;
    }
    const i = AXIS_INDEX[axis];
    const min = this.focusBox.min.getComponent(i);
    const max = this.focusBox.max.getComponent(i);
    const position = min + (max - min) * fraction;
    // By default the cut surface faces the standard (isometric) camera: +X, -Y, +Z.
    const keepBelow = (axis === "y") === flip;
    const sign = keepBelow ? -1 : 1;
    this.clipPlane.normal.set(0, 0, 0).setComponent(i, sign);
    this.clipPlane.constant = -sign * position;
  }

  /** Colour faces that overhang more than 45 degrees (build direction is +Z). */
  setOverhangHighlight(on: boolean): void {
    this.needsRender = true;
    this.overhang.value = on ? 1 : 0;
  }

  /**
   * Patch a material's shader: a face whose normal points more than 45 degrees below
   * horizontal is drawn red, except where it lies on the bed.
   */
  private addOverhangHighlight(material: THREE.MeshStandardMaterial): void {
    material.onBeforeCompile = (shader) => {
      shader.uniforms.uOverhang = this.overhang;
      shader.uniforms.uBedZ = this.bedZ;
      shader.vertexShader = shader.vertexShader
        .replace("#include <common>", "#include <common>\nvarying float vWorldZ;")
        .replace(
          "#include <begin_vertex>",
          "#include <begin_vertex>\nvWorldZ = (modelMatrix * vec4(position, 1.0)).z;",
        );
      shader.fragmentShader = shader.fragmentShader
        .replace(
          "#include <common>",
          "#include <common>\nuniform float uOverhang;\nuniform float uBedZ;\nvarying float vWorldZ;",
        )
        .replace(
          "#include <normal_fragment_begin>",
          `#include <normal_fragment_begin>
          if (uOverhang > 0.5) {
            vec3 worldNormal = normalize((vec4(normal, 0.0) * viewMatrix).xyz);
            if (worldNormal.z < -0.7071 && vWorldZ > uBedZ + 0.3) {
              diffuseColor.rgb = vec3(0.92, 0.22, 0.2);
            }
          }`,
        );
    };
  }

  /** Look at the focus box from a named direction; "fit" keeps the current direction. */
  setView(view: ViewName): void {
    this.needsRender = true;
    const center = this.focusBox.getCenter(new THREE.Vector3());
    const radius = this.focusBox.getSize(new THREE.Vector3()).length() / 2;
    const direction =
      view === "fit"
        ? this.camera.position.clone().sub(this.controls.target)
        : VIEW_DIRECTIONS[view].clone();
    direction.normalize();
    const halfFov = THREE.MathUtils.degToRad(this.camera.fov / 2);
    const fitFov = Math.min(halfFov, Math.atan(Math.tan(halfFov) * this.camera.aspect));
    const distance = (radius / Math.sin(fitFov)) * 1.15;
    this.camera.position.copy(center).addScaledVector(direction, distance);
    this.camera.near = Math.max(distance / 1000, 0.01);
    this.camera.far = distance * 100;
    this.camera.updateProjectionMatrix();
    this.controls.target.copy(center);
    this.controls.update();
  }

  /** Dimension labels sit at the middle of three bounding-box edges. */
  private updateLabels(): void {
    if (!this.hasModel) {
      for (const label of this.labels) label.hidden = true;
      return;
    }
    const { min, max } = this.focusBox;
    const size = this.focusBox.getSize(new THREE.Vector3());
    const anchors = [
      new THREE.Vector3((min.x + max.x) / 2, min.y, min.z),
      new THREE.Vector3(max.x, (min.y + max.y) / 2, min.z),
      new THREE.Vector3(min.x, min.y, (min.z + max.z) / 2),
    ];
    const width = this.container.clientWidth;
    const height = this.container.clientHeight;
    anchors.forEach((anchor, i) => {
      const label = this.labels[i];
      const p = anchor.project(this.camera);
      label.hidden = p.z > 1 || Math.abs(p.x) > 1 || Math.abs(p.y) > 1;
      label.textContent = `${"XYZ"[i]} ${size.getComponent(i).toFixed(1)} mm`;
      label.style.left = `${((p.x + 1) / 2) * width}px`;
      label.style.top = `${((1 - p.y) / 2) * height}px`;
    });
  }

  private resize(): void {
    this.needsRender = true;
    const { clientWidth: w, clientHeight: h } = this.container;
    if (w === 0 || h === 0) return;
    this.renderer.setSize(w, h);
    this.edgeMaterial.resolution.set(w, h);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
  }
}
