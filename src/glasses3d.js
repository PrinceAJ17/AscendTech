// =============================================================================
// Step 3: 3D glasses rendered with Three.js
//
// What this module does, every frame:
//   478 landmarks → build the head's 3D position + orientation ("pose")
//     → move a group containing [glasses model + invisible head occluder] to that pose
//     → Three.js draws it on a transparent canvas layered over the video
//
// Why 3D beats the 2D image: turning (yaw) and nodding (pitch) now rotate the
// glasses for real, the arms appear from the side, and they disappear behind
// the head thanks to the occluder.
// =============================================================================

import * as THREE from 'three';
// RoomEnvironment is a ready-made, generated "room" used only as a reflection
// source, so shiny frames and lenses reflect something believable.
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';

// -----------------------------------------------------------------------------
// Real-world dimensions (millimetres). The models are built at real size, and
// one scale factor (pixels per mm, measured from the face each frame) converts
// them to screen pixels.
// -----------------------------------------------------------------------------
const EYE_CORNER_MM = 90;  // typical adult outer eye-corner distance (landmarks 33 ↔ 263)
const LENS_OFFSET_MM = 31.5; // half the typical pupil distance (63 mm): lens centres at ±31.5
// Glasses sit in front of the eyes, not on the eye corners (which are recessed).
// This moves the whole frame forward/up in head space. shortcut: one value for
// all faces and designs; tune by eye or make it per-design if some look off.
const FRAME_OFFSET = new THREE.Vector3(0, 0, 18);
// Head-fit defaults, tuned with the debug sliders by lining the virtual frames
// up against a real pair of glasses worn on camera. Exported so the sliders
// start at the same values.
// shortcut: tuned on one face; re-check on several people and average.
export const OCCLUDER_DEFAULT_Y_MM = -12; // occluder height vs the eye line (negative = lower)
export const PITCH_DEFAULT_DEG = -2;      // see pitchCorrection below

// -----------------------------------------------------------------------------
// Three.js core objects (see the explanation of Scene/Camera/Renderer earlier)
// -----------------------------------------------------------------------------
let renderer, scene, camera;
let head;         // group that follows the head; everything below is its child
let occluder;     // invisible head shape that hides the arms behind the head
let keyLight;     // the main directional light, adjusted by setLighting()
const KEY_LIGHT_INTENSITY = 1.2; // its brightness in a "normal" room
const designs = {}; // id → THREE.Group for each glasses model

// Reusable vectors/matrix. Creating new objects 60 times a second makes the
// garbage collector pause the page now and then (visible stutter), so we
// allocate once and overwrite them every frame.
const pA = new THREE.Vector3();
const pB = new THREE.Vector3();
const pTop = new THREE.Vector3();
const pChin = new THREE.Vector3();
const xAxis = new THREE.Vector3();
const yAxis = new THREE.Vector3();
const zAxis = new THREE.Vector3();
const centre = new THREE.Vector3();
const scaleVec = new THREE.Vector3();
const poseMatrix = new THREE.Matrix4();

// Pitch correction. Our "up" direction comes from chin → forehead, but that
// line isn't exactly the head's true up: the forehead slopes backwards, so the
// computed "out of the face" direction tilts a few degrees. Things near the
// face (the glasses) barely notice, but the occluder sits ~100 mm behind the
// face, so the tilt shifts it visibly up or down. This rotation around the
// eye line (the x axis) cancels that tilt. Tuned with the debug slider.
const pitchCorrection = new THREE.Matrix4().makeRotationX(THREE.MathUtils.degToRad(PITCH_DEFAULT_DEG));

// -----------------------------------------------------------------------------
// Setup
// -----------------------------------------------------------------------------
export function initGlasses3D(canvas) {
  // alpha: true → transparent background, so the <video> underneath shows through.
  // antialias: true → smooth edges on the thin frame lines.
  renderer = new THREE.WebGLRenderer({ canvas, alpha: true, antialias: true });
  renderer.setPixelRatio(1); // canvas size = video size exactly (set in resize)
  // Tone mapping compresses bright highlights the way a camera does, so the
  // reflections don't look blown out next to the real video.
  renderer.toneMapping = THREE.ACESFilmicToneMapping;

  scene = new THREE.Scene();

  // Build the reflection map once. PMREM pre-blurs the room at several
  // roughness levels so rough and glossy materials both reflect correctly.
  const pmrem = new THREE.PMREMGenerator(renderer);
  scene.environment = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;
  pmrem.dispose();

  // One soft light from above-front, like a ceiling light, for a highlight on
  // top of the frame. setLighting() moves, dims and tints it to match the room.
  keyLight = new THREE.DirectionalLight(0xffffff, KEY_LIGHT_INTENSITY);
  keyLight.position.set(0, 1, 1);
  scene.add(keyLight);

  // Orthographic camera: no perspective, so 1 unit = 1 video pixel everywhere.
  // Why not a perspective camera with MediaPipe's head-pose matrix? That needs
  // our camera's field of view to exactly match the one MediaPipe assumes, and
  // any mismatch makes glasses drift off the face near the image edges. The
  // landmarks themselves are in image space, so building the pose from them and
  // using a matching orthographic camera lines up by construction.
  // Trade-off: no perspective shrinking within the glasses (the far arm isn't
  // drawn smaller). At webcam distance (~50 cm) the difference is small.
  camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0.1, 10000);
  camera.position.z = 5000; // far in front, looking down -z at the face

  head = new THREE.Group();
  // We set head.matrix ourselves every frame, so stop Three.js from
  // recomputing it from position/rotation/scale.
  head.matrixAutoUpdate = false;
  head.visible = false; // hidden until a face is found
  scene.add(head);

  head.add(buildOccluder());

  designs.round = buildGlasses(roundOutline(), plasticMaterial(0x1a1a1a), clearLensMaterial());
  designs.rectangle = buildGlasses(rectangleOutline(), plasticMaterial(0x3b2418), clearLensMaterial());
  designs.aviator = buildGlasses(aviatorOutline(), metalMaterial(0xc9a24a), tintedLensMaterial(), { thin: true });
  for (const g of Object.values(designs)) {
    g.position.copy(FRAME_OFFSET);
    g.visible = false;
    head.add(g);
  }
}

// Called when the camera starts: match the drawing area to the video's pixels.
export function resizeGlasses3D(width, height) {
  // false = don't touch the canvas's CSS size; the stylesheet already stretches it.
  renderer.setSize(width, height, false);
  // Camera window centred on the image: x from -W/2 to W/2, y from -H/2 to H/2.
  camera.left = -width / 2;
  camera.right = width / 2;
  camera.top = height / 2;
  camera.bottom = -height / 2;
  camera.updateProjectionMatrix(); // required after changing camera settings
}

export function setDesign(id) {
  for (const [key, g] of Object.entries(designs)) g.visible = key === id;
}

// Match the virtual lighting to the room (estimate from src/lighting.js).
// Pass null to go back to the fixed default lighting.
//   brightness: 1 = normal room → scales both the reflections and the key light
//   tint:       colour multipliers (~0.8-1.2) → colours the key light
//   dirX/dirY:  -1..1, which side of the face is brighter → moves the key light there
// The reflection map stays neutral in colour (re-tinting it would mean
// rebuilding it), so metal frames pick up the tint less than plastic ones.
export function setLighting(est) {
  const e = est ?? { brightness: 1, tint: { r: 1, g: 1, b: 1 }, dirX: 0, dirY: 0 };
  scene.environmentIntensity = e.brightness;
  keyLight.intensity = KEY_LIGHT_INTENSITY * e.brightness;
  keyLight.color.setRGB(e.tint.r, e.tint.g, e.tint.b);
  // A directional light shines from its position toward the origin. x: left/
  // right of the face (×2 to exaggerate a little, since the measured difference
  // is subtle). y: stays mostly overhead, nudged by top/bottom brightness.
  // z: always in front, since a light behind the head wouldn't light the frame front.
  keyLight.position.set(e.dirX * 2, 1 + e.dirY, 1);
}

// Debug sliders for tuning the head fit (see pitchCorrection and buildOccluder).
export function setPitchCorrection(degrees) {
  pitchCorrection.makeRotationX(THREE.MathUtils.degToRad(degrees));
}
export function setOccluderHeight(mm) {
  occluder.position.y = mm;
}

// Debug: draw the occluder as a wireframe so its fit can be checked.
export function setOccluderVisible(show) {
  occluder.material.colorWrite = show;
  occluder.material.wireframe = show;
}

// -----------------------------------------------------------------------------
// Per frame: landmarks → head pose
// -----------------------------------------------------------------------------
// Converts one MediaPipe landmark into Three.js coordinates (pixels):
//   MediaPipe: x right, y DOWN, origin top-left;   z negative = closer to camera
//   Three.js:  x right, y UP,   origin at centre;  z positive = closer to camera
// So we shift to the centre, and flip both y and z. Flipping two axes is the
// same as a 180° rotation, which keeps the coordinate system "right-handed"
// (flipping just one would mirror the model and turn it inside out).
// z is in "roughly the same scale as x", which was normalised by image width,
// so multiplying by width puts it in pixels too.
function toScene(p, w, h, out) {
  return out.set(p.x * w - w / 2, -(p.y * h - h / 2), -p.z * w);
}

export function updatePose(face, w, h) {
  if (!face) {
    head.visible = false;
    return;
  }
  head.visible = true;

  toScene(face[33], w, h, pA);    // outer eye corner, left side of the raw image
  toScene(face[263], w, h, pB);   // outer eye corner, right side
  toScene(face[10], w, h, pTop);  // top of the forehead
  toScene(face[152], w, h, pChin); // bottom of the chin

  // Build three perpendicular unit vectors describing which way the head faces.
  // x: along the eye line (left → right in the image).
  xAxis.subVectors(pB, pA);
  const eyeDist = xAxis.length(); // 3D distance in pixels, measured before normalising
  xAxis.normalize();
  // "Up-ish" direction: chin → forehead. Not exactly perpendicular to x, so:
  yAxis.subVectors(pTop, pChin);
  // z = x × up: the cross product gives a vector perpendicular to both, i.e.
  // straight out of the face (towards the camera when you face it).
  zAxis.crossVectors(xAxis, yAxis).normalize();
  // Recompute y = z × x so all three are exactly perpendicular (a clean rotation).
  yAxis.crossVectors(zAxis, xAxis);

  // Scale: pixels per millimetre. Because eyeDist is a 3D distance (it includes
  // depth), it does NOT shrink when you turn your head, unlike the 2D version
  // where turning made the glasses shrink.
  const pxPerMm = eyeDist / EYE_CORNER_MM;

  centre.addVectors(pA, pB).multiplyScalar(0.5); // midpoint of the eye corners

  // Assemble one 4×4 matrix = rotation (the three axes as columns) × uniform
  // scale, with the translation set to the eye-line centre. Every child of
  // `head` (glasses + occluder) built in millimetres is now placed on the face.
  poseMatrix.makeBasis(xAxis, yAxis, zAxis);
  // Multiplying on the right applies the correction in head space, i.e. it
  // tilts the head's own axes around its own eye line.
  poseMatrix.multiply(pitchCorrection);
  poseMatrix.scale(scaleVec.setScalar(pxPerMm));
  poseMatrix.setPosition(centre);
  head.matrix.copy(poseMatrix);
  // Children use matrixWorld; mark it dirty so Three.js recomputes it.
  head.matrixWorldNeedsUpdate = true;
}

export function renderGlasses3D() {
  renderer.render(scene, camera);
}

// -----------------------------------------------------------------------------
// Occluder: an invisible head that hides whatever is behind it
// -----------------------------------------------------------------------------
// The GPU keeps a depth buffer (distance of the nearest surface per pixel) and
// only draws a pixel if it's closer than what's there. This shape writes depth
// but no colour (colorWrite: false): invisible, yet anything behind it, like
// the arms running back to the ears, fails the depth test and isn't drawn, so
// the real head in the video shows through. renderOrder -1 draws it first.
function buildOccluder() {
  // A sphere stretched into a rough head-shaped ellipsoid.
  const geometry = new THREE.SphereGeometry(1, 32, 24);
  const material = new THREE.MeshBasicMaterial({ color: 0x00ff88, colorWrite: false });
  occluder = new THREE.Mesh(geometry, material);
  // Head-sized radii in mm: 75 wide (half-width), 120 tall (top of skull to
  // chin is ~240 mm), 95 deep.
  occluder.scale.set(75, 120, 95);
  // Centre: the eyes sit roughly halfway down the head, so the centre is near
  // the eye line (slightly below it), and ~100 mm behind the eye corners so its
  // front surface stays just behind the eye-corner plane. Any further forward
  // and it would cover the bridge and lens centres (which sit in front of the
  // face). shortcut: one average head shape; a face-fitted mesh would hug real
  // faces better.
  occluder.position.set(0, OCCLUDER_DEFAULT_Y_MM, -100);
  occluder.renderOrder = -1;
  return occluder;
}

// -----------------------------------------------------------------------------
// Glasses models, built from code (procedural)
// -----------------------------------------------------------------------------
// Why build them in code instead of downloading .glb files? No licensing to
// track, no large downloads, and every dimension is explicit and adjustable.
// The trade-off is simpler shapes than a professional 3D model. Swapping in a
// .glb later only means replacing buildGlasses() with GLTFLoader.
//
// Each design is described by one lens OUTLINE (2D points, in mm, centred on
// the lens). From that we build: the rim (a ring around the outline), the
// lens (the filled outline), a bridge and two arms.

// Lens outlines. THREE.Shape is a 2D path (like an SVG path); getPoints()
// turns its curves into a list of points.
function roundOutline() {
  const s = new THREE.Shape();
  s.absarc(0, 0, 22, 0, Math.PI * 2, false); // full circle, radius 22 mm
  return s.getPoints(48);
}

function rectangleOutline() {
  // Rounded rectangle, 52 × 34 mm, corner radius 7 mm.
  const w = 26, h = 17, r = 7;
  const s = new THREE.Shape();
  s.moveTo(-w + r, -h);
  s.lineTo(w - r, -h);
  s.quadraticCurveTo(w, -h, w, -h + r);
  s.lineTo(w, h - r);
  s.quadraticCurveTo(w, h, w - r, h);
  s.lineTo(-w + r, h);
  s.quadraticCurveTo(-w, h, -w, h - r);
  s.lineTo(-w, -h + r);
  s.quadraticCurveTo(-w, -h, -w + r, -h);
  return s.getPoints(8);
}

function aviatorOutline() {
  // Teardrop for the lens on the +x side: flatter top, deepest part toward the
  // nose (-x is the inner/nose side). The other lens is mirrored automatically.
  const s = new THREE.Shape();
  s.moveTo(-21, 13);
  s.quadraticCurveTo(2, 17, 25, 13);    // top edge, inner → outer
  s.quadraticCurveTo(29, -4, 12, -20);  // outer side, down to the bottom
  s.quadraticCurveTo(-6, -27, -19, -12); // bottom, curving up toward the nose
  s.quadraticCurveTo(-26, 0, -21, 13);  // inner side back to the start
  return s.getPoints(16);
}

// Materials. "Physical"/"Standard" materials react to light and reflections
// realistically (PBR: physically based rendering).
function plasticMaterial(color) {
  // Acetate frames: fairly smooth, with a glossy clear coat on top.
  return new THREE.MeshPhysicalMaterial({ color, roughness: 0.35, clearcoat: 1, clearcoatRoughness: 0.1 });
}
function metalMaterial(color) {
  return new THREE.MeshStandardMaterial({ color, metalness: 1, roughness: 0.25 });
}
// Lenses use plain transparency (opacity) rather than real glass "transmission":
// transmission refracts what's behind it in the 3D scene, but our background is
// the HTML video, which isn't in the scene, so it would refract nothing.
// depthWrite: false stops a lens from hiding the arm or other lens behind it.
function clearLensMaterial() {
  return new THREE.MeshPhysicalMaterial({
    color: 0xddeeff, transparent: true, opacity: 0.12, roughness: 0.05, depthWrite: false,
  });
}
function tintedLensMaterial() {
  return new THREE.MeshPhysicalMaterial({
    color: 0x3c2814, transparent: true, opacity: 0.6, roughness: 0.05, depthWrite: false,
  });
}

// Builds one complete pair of glasses from a lens outline.
// Coordinates (mm): x = across the face, y = up, z = toward the camera.
// The lens plane is z = 0; arms run back toward -z.
function buildGlasses(outline, frameMat, lensMat, { thin = false } = {}) {
  const group = new THREE.Group();
  const rim = thin ? 1.2 : 3; // rim width: metal aviators are much thinner
  const depth = thin ? 1.2 : 4; // how thick the frame is front-to-back

  // Each side: +1 = lens on the +x side, -1 = the mirrored lens on the -x side.
  for (const side of [1, -1]) {
    // Mirror the outline for the -x side. Mirroring reverses the point order
    // (clockwise ↔ anticlockwise), which would flip the faces inside out, so
    // reverse the array to restore it.
    let pts = outline.map((p) => new THREE.Vector2(p.x * side, p.y));
    if (side === -1) pts = pts.reverse();

    // Rim: the outline grown outward by `rim` mm, with the original outline cut
    // out as a hole, then extruded to give it thickness.
    const outer = growOutline(pts, rim);
    const rimShape = new THREE.Shape(outer);
    rimShape.holes.push(new THREE.Path(pts));
    const rimGeo = new THREE.ExtrudeGeometry(rimShape, { depth, bevelEnabled: false });
    rimGeo.translate(0, 0, -depth / 2); // centre the thickness on z = 0
    const rimMesh = new THREE.Mesh(rimGeo, frameMat);
    rimMesh.position.x = side * LENS_OFFSET_MM;
    group.add(rimMesh);

    // Lens: the filled outline, slightly in front so it sits inside the rim.
    const lens = new THREE.Mesh(new THREE.ShapeGeometry(new THREE.Shape(pts)), lensMat);
    lens.position.set(side * LENS_OFFSET_MM, 0, 0.5);
    lens.renderOrder = 1; // transparent things draw after solid ones
    group.add(lens);

    // Arm (temple): starts at the outer edge of the rim, runs back to the ear,
    // then bends down behind it. CatmullRomCurve3 draws a smooth curve through
    // the given points; TubeGeometry wraps a round tube around that curve.
    // Outermost rim point: furthest outline point from the lens centre on the
    // outer side, plus where the lens centre itself sits.
    const edgeX = LENS_OFFSET_MM + Math.max(...outer.map((p) => p.x * side));
    const armTop = 8; // arms attach a bit above the lens centre
    const armCurve = new THREE.CatmullRomCurve3([
      new THREE.Vector3(side * edgeX, armTop, 0),
      new THREE.Vector3(side * (edgeX + 6), armTop, -6),   // hinge, turning backwards
      new THREE.Vector3(side * 68, armTop - 1, -35),
      new THREE.Vector3(side * 70, armTop - 4, -95),       // reaches the ear
      new THREE.Vector3(side * 67, armTop - 22, -115),     // bends down behind it
    ]);
    group.add(new THREE.Mesh(new THREE.TubeGeometry(armCurve, 40, thin ? 0.9 : 1.6, 8), frameMat));
  }

  // Bridge: a small arch over the nose joining the two inner rim edges.
  const innerX = LENS_OFFSET_MM - Math.max(...outline.map((p) => -p.x)); // inner edge of the lens
  const bridgeCurve = new THREE.CatmullRomCurve3([
    new THREE.Vector3(-innerX, 6, 0),
    new THREE.Vector3(0, 10, 1.5),
    new THREE.Vector3(innerX, 6, 0),
  ]);
  group.add(new THREE.Mesh(new THREE.TubeGeometry(bridgeCurve, 20, thin ? 0.9 : 2, 8), frameMat));

  // Aviators have a second, straight "brow bar" above the bridge.
  if (thin) {
    const bar = new THREE.LineCurve3(new THREE.Vector3(-innerX - 6, 15, 0), new THREE.Vector3(innerX + 6, 15, 0));
    group.add(new THREE.Mesh(new THREE.TubeGeometry(bar, 1, 0.9, 8), frameMat));
  }
  return group;
}

// Pushes every point of a closed outline outward by `amount` mm, along the
// direction away from the lens centre (0, 0). Good enough for these convex-ish
// lens shapes; a true offset curve would keep the rim width perfectly even.
function growOutline(pts, amount) {
  return pts.map((p) => {
    const len = p.length();
    return new THREE.Vector2(p.x * (1 + amount / len), p.y * (1 + amount / len));
  });
}
