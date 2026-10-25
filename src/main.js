// =============================================================================
// Glasses Try-On — step 3: webcam + face landmarks + 3D glasses (Three.js)
//
// Pipeline for every camera frame:
//   webcam frame → MediaPipe Face Landmarker → 478 landmarks
//     → 3D mode (default): head pose → Three.js renders 3D glasses  (src/glasses3d.js)
//     → 2D mode (baseline): position/rotation/scale → draw flat image (this file)
//
// The 2D version is kept behind a checkbox so the two can be compared side by
// side, which is useful for the write-up ("what 3D fixed").
//
// Screen layers, bottom to top (all in .stage, all mirrored by CSS):
//   <video> webcam  →  <canvas id="three"> 3D glasses  →  <canvas id="overlay"> 2D glasses + debug dots
// =============================================================================

// FaceLandmarker is MediaPipe's face-tracking "Task". It runs on MediaPipe's
// WebAssembly runtime (the compiled C++ that actually runs the model).
import { FaceLandmarker } from '@mediapipe/tasks-vision';
// "?url" asks Vite for the file's URL instead of its code. Vite then copies the
// runtime files into the build itself, so the deployed app serves them from
// its own server (no CDN), and they always match the installed npm version.
import wasmLoaderPath from '@mediapipe/tasks-vision/vision_wasm_internal.js?url';
import wasmBinaryPath from '@mediapipe/tasks-vision/vision_wasm_internal.wasm?url';
import {
  initGlasses3D, resizeGlasses3D, updatePose, renderGlasses3D, setDesign, setOccluderVisible,
  setPitchCorrection, setOccluderHeight, OCCLUDER_DEFAULT_Y_MM, PITCH_DEFAULT_DEG, setLighting,
  headAngles, updateFaceOccluder,
} from './glasses3d.js';
import { isRecording, startRecording, recordFrame, cancelRecording, setConditions } from './metrics.js';
import { OneEuroFilter } from './oneEuro.js';
import { estimateLighting } from './lighting.js';

// -----------------------------------------------------------------------------
// Where the runtime files come from
// -----------------------------------------------------------------------------
// Both are served by the app itself, so it doesn't depend on anyone else's servers.

// The .wasm runtime: the SIMD build (imported above). SIMD = processing several
// numbers per instruction, which every current browser supports.
// shortcut: no fallback to MediaPipe's non-SIMD build; only browsers from
// before ~2021 (e.g. Safari < 16.4) lack SIMD. Add vision_wasm_nosimd_internal if those matter.
const WASM_FILESET = { wasmLoaderPath, wasmBinaryPath };

// The trained model bundle: BlazeFace detector + face landmark model (+ blendshapes),
// stored in public/models/. Downloaded once from Google's official MediaPipe model storage:
// https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
// "float16" = weights stored as 16-bit floats: half the size of float32, same accuracy in practice.
const MODEL_URL = '/models/face_landmarker.task';

// -----------------------------------------------------------------------------
// Landmarks we care about (index into the 478-point face mesh).
// These are the points the glasses math will use in step 2, drawn big and
// labelled so we can visually confirm each number sits where we expect.
// -----------------------------------------------------------------------------
const KEY_POINTS = {
  168: 'bridge',     // between the eyes, top of nose: where glasses rest
  33: 'eye R out',   // outer corner of the person's right eye (left side of the raw image)
  263: 'eye L out',  // outer corner of the person's left eye
  468: 'iris R',     // iris centres (only exist in the 478-point model, 468-477 are iris points)
  473: 'iris L',
  234: 'face R',     // edges of the face near the ears: for frame width
  454: 'face L',
};

// -----------------------------------------------------------------------------
// Glasses images (step 2: flat 2D overlay)
// -----------------------------------------------------------------------------
// Each design is an SVG in public/glasses/. To place an image on a face we need
// to know where its lens centres are, in the image's own pixel coordinates.
// All three SVGs were drawn on a 280×100 canvas with lens centres at the same
// spots, so they share these numbers. A new design must either use the same
// layout or carry its own lensLeft/lensRight values.
const LENS_LEFT = { x: 78, y: 50 };   // lens centre on the image's left side
const LENS_RIGHT = { x: 202, y: 50 }; // lens centre on the image's right side

const GLASSES = {
  round: { src: '/glasses/round.svg' },
  rectangle: { src: '/glasses/rectangle.svg' },
  aviator: { src: '/glasses/aviator.svg' },
};

// Load every image once at startup and keep the decoded Image on its entry, so
// switching glasses is instant (no download mid-tracking).
for (const g of Object.values(GLASSES)) {
  g.img = new Image();
  g.img.src = g.src;
}

// How far apart the lens centres are compared with the outer eye corners.
// Why not just put the lenses on the irises (468/473)? Because irises move when
// you look sideways, and the glasses would slide around with your gaze. Eye
// corners are fixed to the skull, so we measure those and convert:
//   typical adult pupil distance ≈ 63 mm, outer eye-corner distance ≈ 90 mm
//   → lens centres sit at ~0.7× the eye-corner distance.
// shortcut: one average ratio for everyone; tune by eye, or use per-face iris
// distance measured while looking straight ahead if fit looks off.
const LENS_TO_EYE_CORNER_RATIO = 0.7;

// -----------------------------------------------------------------------------
// Smoothing (step 4)
// -----------------------------------------------------------------------------
// We smooth only the landmarks the glasses actually use, not the final pose.
// Smoothing rotations directly is awkward (angles wrap around 359° → 0°, and
// a blended rotation has to be re-normalised). Smoothing the points and then
// building the pose from them as before avoids both, and the 2D mode gets
// smoothing for free since it uses the same eye corners.
const SMOOTHED_POINTS = [33, 263, 10, 152]; // eye corners, forehead, chin

// Shared by every filter, so the sliders tune all of them at once.
// Units: landmarks are 0-1 fractions of the frame, so speeds are "frame
// widths per second". A head turn is roughly 0.5-1, which is why beta is
// much bigger than the ~0.007 in the paper (that was in pixels per second).
// Defaults tuned live on camera: at minCutoff 1 Hz the glasses visibly dragged
// behind the face (~160 ms of delay when still, since delay ≈ 1/(2π·cutoff)),
// while raw tracking was already fairly steady. 3 Hz (~50 ms) removed the drag
// with only a very slight jitter left. Light smoothing also protects the
// <100 ms end-to-end latency budget.
//
// Smoothing is OFF by default (the checkbox in index.html starts unticked).
// Measurements showed raw frame-to-frame jitter of only ~0.24 px in normal
// light and ~0.36 px in dim light (invisible), while smoothing made the glasses
// lag ~3-4 px behind the face during movement (visible). These values are kept
// for when it's switched on, e.g. on a noisier camera.
// shortcut: measured on one webcam and one face; re-check on a cheaper webcam.
const SMOOTHING = { minCutoff: 3, beta: 30, dCutoff: 1.0 };

// One filter per number: 4 points × (x, y, z) = 12 filters.
const filters = {};
for (const i of SMOOTHED_POINTS) {
  filters[i] = { x: new OneEuroFilter(SMOOTHING), y: new OneEuroFilter(SMOOTHING), z: new OneEuroFilter(SMOOTHING) };
}

// Smoothed copies of those points live here, at the same indices as in the
// raw landmark array, so updatePose/drawGlasses can read face[33] etc. from
// either one without knowing whether it's smoothed. Reused every frame.
const smoothFace = [];
for (const i of SMOOTHED_POINTS) smoothFace[i] = { x: 0, y: 0, z: 0 };

function resetFilters() {
  for (const f of Object.values(filters)) { f.x.reset(); f.y.reset(); f.z.reset(); }
}

// Returns the landmarks to place the glasses with: smoothed, or raw when the
// smoothing checkbox is off (for comparing the two live).
function smoothLandmarks(face, now) {
  if (!face || !smoothing.checked) {
    // No face (or smoothing off): forget history so the next smoothed frame
    // starts from the real position instead of gliding from an old one.
    resetFilters();
    return face;
  }
  for (const i of SMOOTHED_POINTS) {
    smoothFace[i].x = filters[i].x.filter(face[i].x, now);
    smoothFace[i].y = filters[i].y.filter(face[i].y, now);
    smoothFace[i].z = filters[i].z.filter(face[i].z, now);
  }
  return smoothFace;
}

// -----------------------------------------------------------------------------
// Jitter measurement
// -----------------------------------------------------------------------------
// How much the glasses' centre (midpoint of the eye corners) wobbles, in video
// pixels: the standard deviation of its position over the last second. Only
// meaningful while you hold your head still, because real movement also counts
// as "spread". Comparing raw vs smoothed gives the before/after number for the
// write-up ("jitter reduced from X px to Y px").
function eyeCentre(face, w, h) {
  return { x: (face[33].x + face[263].x) / 2 * w, y: (face[33].y + face[263].y) / 2 * h };
}

// 2D standard deviation: sqrt(variance in x + variance in y). Roughly "the
// typical distance of a sample from the average position".
function spread(samples) {
  if (samples.length < 2) return null;
  const mx = samples.reduce((s, p) => s + p.x, 0) / samples.length;
  const my = samples.reduce((s, p) => s + p.y, 0) / samples.length;
  const v = samples.reduce((s, p) => s + (p.x - mx) ** 2 + (p.y - my) ** 2, 0) / samples.length;
  return Math.sqrt(v);
}

// -----------------------------------------------------------------------------
// Grab the HTML elements once, up front
// -----------------------------------------------------------------------------
const video = document.getElementById('video');
const canvas = document.getElementById('overlay');
const ctx = canvas.getContext('2d'); // 2D drawing API for the debug dots
const placeholder = document.getElementById('placeholder');
const cameraBtn = document.getElementById('cameraBtn');
const fpsPill = document.getElementById('fpsPill');
const statusEl = document.getElementById('status');
const showLandmarks = document.getElementById('showLandmarks');
const use3D = document.getElementById('use3D');
const showOccluder = document.getElementById('showOccluder');
const statRender = document.getElementById('statRender');
const smoothing = document.getElementById('smoothing');
const matchLighting = document.getElementById('matchLighting');
const statLight = document.getElementById('statLight');
const statLatency = document.getElementById('statLatency');
const recordBtn = document.getElementById('recordBtn');
const scenarioInput = document.getElementById('scenario');
const durationSelect = document.getElementById('duration');
const recordStatus = document.getElementById('recordStatus');
const summaryOut = document.getElementById('summary');
const cameraCaps = document.getElementById('cameraCaps');
const statJitterRaw = document.getElementById('statJitterRaw');
const statJitterSmooth = document.getElementById('statJitterSmooth');
const statRes = document.getElementById('statRes');
const statFps = document.getElementById('statFps');
const statInference = document.getElementById('statInference');
const statFace = document.getElementById('statFace');
const statDelegate = document.getElementById('statDelegate');
const statGlasses = document.getElementById('statGlasses');

// -----------------------------------------------------------------------------
// App state
// -----------------------------------------------------------------------------
let stream = null;      // the MediaStream from the webcam (null = camera off)
let cameraFps = null;   // frame rate the camera is configured for (from getSettings)
let cameraInfo = null;  // what the camera supports + its current exposure settings (for the CSV)
let frameHandle = null; // id of the pending per-frame callback, so we can cancel it
let landmarker = null;  // the FaceLandmarker, created once and reused
let selectedGlasses = 'round'; // key into GLASSES, changed by the picker buttons

// Set up the 3D renderer once at startup. WebGL can be missing (very old
// machines, or disabled GPU acceleration); in that case fall back to the 2D
// overlay instead of breaking the app.
try {
  initGlasses3D(document.getElementById('three'));
  setDesign(selectedGlasses);
} catch (err) {
  console.warn('3D unavailable, using 2D overlay:', err);
  use3D.checked = false;
  use3D.disabled = true;
}

function setStatus(text, isError = false) {
  statusEl.textContent = text;
  statusEl.classList.toggle('error', isError);
}

// -----------------------------------------------------------------------------
// Model loading
// -----------------------------------------------------------------------------

// Creates the FaceLandmarker on the requested hardware ("GPU" or "CPU").
async function createLandmarker(delegate) {
  return FaceLandmarker.createFromOptions(WASM_FILESET, {
    baseOptions: { modelAssetPath: MODEL_URL, delegate },
    // VIDEO mode = frame-to-frame tracking: after the first detection it reuses
    // the previous frame's face position instead of re-running the detector,
    // which is what makes it fast. Requires increasing timestamps (see onFrame).
    runningMode: 'VIDEO',
    numFaces: 1,
    // The 4x4 head-pose matrix (rotation + position). Not used yet; step 3
    // feeds it to Three.js to place the 3D glasses.
    outputFacialTransformationMatrixes: true,
  });
}

// Try the GPU first (fast). If the browser/machine can't do it, fall back to
// the CPU (slower but works everywhere). This is the hardware fallback from
// the plan, so the app degrades instead of breaking on weaker machines.
async function loadLandmarker() {
  try {
    landmarker = await createLandmarker('GPU');
    statDelegate.textContent = 'GPU';
  } catch (err) {
    console.warn('GPU delegate failed, falling back to CPU:', err);
    landmarker = await createLandmarker('CPU');
    statDelegate.textContent = 'CPU';
  }
}

// -----------------------------------------------------------------------------
// Camera start / stop
// -----------------------------------------------------------------------------

// Turn getUserMedia's error names into messages a user can act on.
function cameraErrorMessage(err) {
  switch (err.name) {
    case 'NotAllowedError': return 'Camera permission denied. Allow it in the browser address bar.';
    case 'NotFoundError': return 'No camera found.';
    case 'NotReadableError': return 'Camera is in use by another app.';
    default: return `Camera error: ${err.message}`;
  }
}

async function startCamera() {
  // getUserMedia only exists on secure pages (https or localhost).
  if (!navigator.mediaDevices?.getUserMedia) {
    setStatus('This browser does not support camera access (needs HTTPS or localhost).', true);
    return;
  }
  // Disable the button while starting so a double-click can't open the camera twice.
  cameraBtn.disabled = true;
  try {
    // Load the model only the first time; it stays in memory afterwards.
    if (!landmarker) {
      setStatus('Loading face model…');
      await loadLandmarker();
    }

    setStatus('Requesting camera…');
    // Ask for 720p at 30 FPS from the front camera. "ideal" means "if possible":
    // the browser picks the closest the camera supports instead of failing.
    // The frame rate is requested explicitly because measurements showed the
    // camera defaulting to ~20 FPS, below the brief's 24 FPS target, while our
    // processing has room for far more. In a dim room the camera may still
    // drop below 30 to expose each frame longer; that's the camera's choice.
    stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 1280 }, height: { ideal: 720 }, frameRate: { ideal: 30 }, facingMode: 'user' },
      audio: false,
    });
  } catch (err) {
    // Camera errors have a .name (NotAllowedError etc.); model errors don't.
    setStatus(err.name ? cameraErrorMessage(err) : `Failed to load face model: ${err.message}`, true);
    cameraBtn.disabled = false;
    return;
  }

  // Feed the webcam stream into the <video> element and start playback.
  video.srcObject = stream;
  await video.play();

  // Make the canvas's drawing resolution equal the video's pixel size. Then a
  // landmark at (x, y) in 0-1 coordinates is simply (x * width, y * height) on
  // the canvas. The stage is full-screen and all three layers use the same
  // `object-fit: cover` crop (see style.css), so they stay aligned on screen.
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  // Same for the 3D canvas, so 1 Three.js unit = 1 video pixel.
  if (!use3D.disabled) resizeGlasses3D(video.videoWidth, video.videoHeight);

  // What the camera agreed to (it may differ from what we asked for). This is
  // the frame rate it's configured for; the live FPS stat shows what it
  // actually delivers, which can be lower in dim light.
  cameraFps = stream.getVideoTracks()[0].getSettings().frameRate ?? null;
  showCameraInfo(stream.getVideoTracks()[0]);
  await resetCameraSettings(stream.getVideoTracks()[0]);
  statRes.textContent = `${video.videoWidth}×${video.videoHeight}${cameraFps ? ` @ ${Math.round(cameraFps)}` : ''}`;
  placeholder.hidden = true;
  cameraBtn.textContent = 'Stop camera';
  cameraBtn.disabled = false;
  setStatus('');
  // CSS shows the live-only UI (frame tray, FPS pill, stop button) from this.
  document.body.dataset.camera = 'on';
  recordBtn.disabled = false; // measuring needs a running camera
  startFrameLoop();
}

// Diagnostic: what can this camera do, and what is it doing right now?
// We asked for 30 FPS and the camera agreed, yet it delivers 20 (10 in dim
// light), which points to auto-exposure holding each frame's shutter open
// longer. To know whether the app could override that, we need to see:
//   capabilities: the ranges the camera SUPPORTS, e.g.
//     frameRate {min, max}           → can it do 30 at all?
//     exposureMode ['manual', 'continuous'] → can exposure be set by us?
//     exposureTime {min, max, step}  → how short a shutter it allows
//   settings: what it's CURRENTLY using (e.g. exposureMode 'continuous' = auto)
// Not every browser/camera reports exposure controls; missing keys mean
// "not exposed to web pages", not necessarily "the camera can't".
function showCameraInfo(track) {
  const caps = track.getCapabilities?.() ?? {};
  const settings = track.getSettings();
  // Keep only the keys relevant to frame rate and exposure, so it's readable.
  const KEYS = ['frameRate', 'width', 'height', 'exposureMode', 'exposureTime', 'exposureCompensation',
    'brightness', 'whiteBalanceMode', 'resizeMode'];
  const pick = (o) => Object.fromEntries(KEYS.filter((k) => k in o).map((k) => [k, o[k]]));
  cameraInfo = { label: track.label, capabilities: pick(caps), settings: pick(settings) };
  cameraCaps.textContent = JSON.stringify(cameraInfo, null, 2);
  console.log('Camera info (full capabilities):', caps, 'settings:', settings);
}

// -----------------------------------------------------------------------------
// Camera settings reset
// -----------------------------------------------------------------------------
// Webcam settings like brightness and exposure mode are stored in the
// camera/driver itself, not in the page. They survive a page refresh, a camera
// restart, even closing the browser. During testing, a brightness of 64 left
// over from one session washed out the image in the next. So every start
// resets the camera to a known baseline: auto exposure, and brightness and
// exposure compensation at the middle of their ranges (0 and 64 on the ASUS
// webcam: the values it reported before anything was changed).
//
// Why there are no manual exposure controls any more: they were tried to get
// past the camera's 20 FPS cap and rejected. In auto mode this webcam also
// raises sensor gain (amplification); in manual mode the gain drops and the
// browser offers no gain control, so even the longest exposure that allows
// 30 FPS (31.25 ms) gave an image too dark to find a face. "Brightness" only
// added grey haze, and FPS didn't reliably rise anyway. See ThoughtProcess.md.
async function resetCameraSettings(track) {
  const caps = track.getCapabilities?.() ?? {};
  const baseline = {};
  if (caps.exposureMode?.includes('continuous')) baseline.exposureMode = 'continuous';
  if (caps.brightness) baseline.brightness = midpoint(caps.brightness);
  if (caps.exposureCompensation) baseline.exposureCompensation = midpoint(caps.exposureCompensation);
  if (Object.keys(baseline).length) {
    // applyConstraints is how a web page changes settings on a running camera;
    // "advanced" means "try these exact values" (the camera may still adjust them).
    try {
      await track.applyConstraints({ advanced: [baseline] });
    } catch (err) {
      console.warn('Could not reset camera settings:', err);
    }
  }
  // Refresh the capabilities panel and CSV info with the post-reset settings.
  showCameraInfo(track);
}

// Middle of a camera range, snapped to its step: the neutral setting.
function midpoint({ min, max, step = 1 }) {
  return min + Math.round((max - min) / 2 / step) * step;
}

function stopCamera() {
  // Cancel the pending frame callback, otherwise it would fire again on the
  // next camera start and we'd end up with two loops running at once.
  if (frameHandle !== null) video.cancelVideoFrameCallback(frameHandle);
  frameHandle = null;
  // Stopping the tracks turns the camera (and its light) off.
  stream.getTracks().forEach((t) => t.stop());
  stream = null;
  video.srcObject = null;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (!use3D.disabled) { updatePose(null); renderGlasses3D(); } // clear the 3D layer too
  placeholder.hidden = false;
  cameraBtn.textContent = 'Start camera';
  statRes.textContent = statFps.textContent = statInference.textContent = statFace.textContent = '–';
  statRender.textContent = statJitterRaw.textContent = statJitterSmooth.textContent = statLight.textContent = '–';
  resetFilters(); // so the next start doesn't glide from where the face was
  statLatency.textContent = '–';
  // A recording can't continue without frames; drop it rather than mixing two sessions.
  if (isRecording()) {
    cancelRecording();
    recordStatus.textContent = 'Recording cancelled (camera stopped)';
  }
  recordBtn.disabled = true;
  fpsPill.textContent = '–';
  document.body.dataset.camera = 'off';
  setFaceState(false);
  setStatus('');
  cameraBtn.focus(); // the stop button just disappeared; put focus on the start button instead
}

// -----------------------------------------------------------------------------
// Per-frame loop
// -----------------------------------------------------------------------------
// requestVideoFrameCallback (rVFC) fires once for every NEW frame the camera
// delivers, unlike requestAnimationFrame which fires on every screen refresh.
// That means we never run the model twice on the same frame, and we never
// queue up old frames: we always process the latest one, so latency can't grow.
//
// shortcut: no fallback when rVFC is missing (very old browsers);
// add a requestAnimationFrame + video.currentTime check if those must be supported.
function startFrameLoop() {
  if (!('requestVideoFrameCallback' in video)) {
    setStatus('This browser lacks requestVideoFrameCallback; use a recent Chrome, Edge, Firefox or Safari.', true);
    return;
  }

  // Counters for the stats panel, reset every second.
  let frames = 0;
  let inferenceTotal = 0;
  let renderTotal = 0;
  let latencyTotal = 0;
  let latencyCount = 0;   // frames that had a capture timestamp
  let rawCentres = [];    // eye-centre positions this second, raw
  let smoothCentres = []; // ... and smoothed
  let windowStart = performance.now();

  // `now` is a high-resolution timestamp (ms) for this frame. It always
  // increases, which is exactly what VIDEO mode needs to track between frames.
  // `metadata` is extra info the browser gives about the frame, notably:
  //   captureTime:     when the camera captured it (same clock as performance.now();
  //                    not every browser provides it)
  //   presentedFrames: running count of frames the camera has delivered, so
  //                    gaps between calls reveal frames we skipped
  const onFrame = (now, metadata) => {
    // --- 1. Run the model and time it -------------------------------------
    const t0 = performance.now();
    const result = landmarker.detectForVideo(video, now);
    const inferenceMs = performance.now() - t0;
    inferenceTotal += inferenceMs;
    frames++;

    // result.faceLandmarks is an array with one entry per face found (we asked
    // for max 1). Each entry is 478 points of {x, y, z}:
    //   x, y: position as a fraction of image width/height (0 to 1)
    //   z:    depth relative to the head's centre, roughly in the same units as x
    const face = result.faceLandmarks[0]; // undefined when no face is visible
    statFace.textContent = face ? 'yes' : 'no';
    setFaceState(!!face);

    // Smoothed landmarks for placing the glasses (raw `face` is still used for
    // the debug dots, so you can see the raw jitter vs the steady glasses).
    const pose = smoothLandmarks(face, now);

    // Record raw and smoothed centres for the jitter stat. When smoothing is
    // off, both are the raw position.
    if (face) {
      rawCentres.push(eyeCentre(face, canvas.width, canvas.height));
      smoothCentres.push(eyeCentre(pose, canvas.width, canvas.height));
    }

    // --- 2. Draw ------------------------------------------------------------
    // Timed separately from inference so we can see what rendering costs.
    // (This is CPU time to issue the drawing; the GPU finishes asynchronously.)
    const t1 = performance.now();

    // 3D layer: move the glasses to the head pose and draw. When 3D is off,
    // updatePose(null) hides them so the 3D layer is empty.
    if (!use3D.disabled) {
      // Room lighting: estimateLighting only does work ~7 times a second and
      // returns null in between, so most frames this costs nothing.
      if (face && use3D.checked && matchLighting.checked) {
        const light = estimateLighting(video, face, now);
        if (light) {
          setLighting(light);
          statLight.textContent = describeLight(light);
        }
      }
      updatePose(use3D.checked ? pose : null, canvas.width, canvas.height);
      // Raw `face`: the face occluder needs all 478 points (updatePose(null) already hid it otherwise).
      if (face && use3D.checked) updateFaceOccluder(face, canvas.width, canvas.height);
      renderGlasses3D();
    }

    // 2D layer: wipe last frame, then the 2D glasses (only in 2D mode), then
    // the debug dots on top so they stay visible for checking alignment.
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (face && !use3D.checked) drawGlasses(pose);
    if (face && showLandmarks.checked) drawLandmarks(face);
    const tEnd = performance.now();
    const renderMs = tEnd - t1;
    renderTotal += renderMs;

    // Capture → render latency: camera capture moment to "we finished drawing".
    // The screen still has to show it (up to one refresh, ~16 ms at 60 Hz), so
    // true end-to-end is slightly higher; a slow-motion phone video measures that.
    const latencyMs = metadata?.captureTime ? tEnd - metadata.captureTime : null;
    if (latencyMs !== null) {
      latencyTotal += latencyMs;
      latencyCount++;
    }

    // --- 3. Log this frame if a measurement recording is running -------------
    if (isRecording()) {
      const rawC = face ? eyeCentre(face, canvas.width, canvas.height) : null;
      const smoothC = face ? eyeCentre(pose, canvas.width, canvas.height) : null;
      // Angles from the RAW landmarks: we want what the tracker saw, unsmoothed.
      const angles = face ? headAngles(face, canvas.width, canvas.height) : {};
      const progress = recordFrame({
        now,
        presentedFrames: metadata?.presentedFrames ?? null,
        face: !!face,
        inferenceMs,
        renderMs,
        latencyMs,
        yaw: angles.yaw ?? null,
        pitch: angles.pitch ?? null,
        roll: angles.roll ?? null,
        // Face width in pixels (cheek edge to cheek edge): a stand-in for
        // distance from the camera in the near/far test.
        faceWidthPx: face ? Math.hypot((face[454].x - face[234].x) * canvas.width, (face[454].y - face[234].y) * canvas.height) : null,
        rawCx: rawC?.x ?? null,
        rawCy: rawC?.y ?? null,
        smoothCx: smoothC?.x ?? null,
        smoothCy: smoothC?.y ?? null,
      });
      if (isRecording()) recordStatus.textContent = `Recording… ${Math.round(progress * 100)}%`;
    }

    // --- 4. Update stats once per second ------------------------------------
    // Averaging over a window gives a readable number instead of a flickering one.
    const elapsed = now - windowStart;
    if (elapsed >= 1000) {
      statFps.textContent = (frames * 1000 / elapsed).toFixed(1);
      fpsPill.textContent = Math.round(frames * 1000 / elapsed); // compact copy for the top bar
      statInference.textContent = `${(inferenceTotal / frames).toFixed(1)} ms`;
      statRender.textContent = `${(renderTotal / frames).toFixed(1)} ms`;
      statLatency.textContent = latencyCount ? `${(latencyTotal / latencyCount).toFixed(1)} ms` : 'n/a';
      latencyTotal = 0;
      latencyCount = 0;
      const jr = spread(rawCentres);
      const js = spread(smoothCentres);
      statJitterRaw.textContent = jr === null ? '–' : `${jr.toFixed(2)} px`;
      statJitterSmooth.textContent = js === null ? '–' : `${js.toFixed(2)} px`;
      rawCentres = [];
      smoothCentres = [];
      frames = 0;
      inferenceTotal = 0;
      renderTotal = 0;
      windowStart = now;
    }

    // Ask to be called again on the next camera frame.
    frameHandle = video.requestVideoFrameCallback(onFrame);
  };
  frameHandle = video.requestVideoFrameCallback(onFrame);
}

// Short human-readable summary of the lighting estimate for the stats panel,
// e.g. "0.85× warm, from R". Thresholds are just for the label.
function describeLight({ brightness, tint, dirX }) {
  const tone = tint.r - tint.b > 0.06 ? 'warm' : tint.b - tint.r > 0.06 ? 'cool' : 'neutral';
  const side = dirX > 0.05 ? ', from R' : dirX < -0.05 ? ', from L' : '';
  return `${brightness.toFixed(2)}× ${tone}${side}`;
}

// -----------------------------------------------------------------------------
// 2D glasses placement
// -----------------------------------------------------------------------------
// We need three things to put a flat image on a face:
//   1. WHERE  → translation: the point between the eyes
//   2. WHICH WAY → rotation (roll): the angle of the line through the eyes
//   3. HOW BIG → scale: eye distance on the face ÷ lens distance in the image
// All three come from just two landmarks: the outer eye corners (33 and 263).
//
// Known limit of 2D: it can follow moving, tilting (roll) and getting closer/
// further, but NOT turning the head (yaw) or nodding (pitch). A flat image
// can only rotate within the screen plane; when you turn, it just shrinks
// evenly instead of the far lens getting smaller. Step 3 (3D) fixes this.
function drawGlasses(face) {
  const glasses = GLASSES[selectedGlasses];
  // complete = finished loading; naturalWidth 0 = it failed to load
  if (!glasses.img.complete || glasses.img.naturalWidth === 0) return;

  const w = canvas.width;
  const h = canvas.height;

  // Eye corners in canvas pixels (landmarks come as 0-1 fractions of the frame).
  // In the raw (un-mirrored) camera image, the person's right eye (33) is on
  // the LEFT side, so a = left of the image, b = right: same order as the
  // image's LENS_LEFT / LENS_RIGHT, which means no flipping is needed.
  const a = { x: face[33].x * w, y: face[33].y * h };
  const b = { x: face[263].x * w, y: face[263].y * h };

  // 1. WHERE: the midpoint between the eye corners. This sits on the eye line,
  // so lens centres line up with the eyes vertically. (Landmark 168 sits a bit
  // above the eye line, closer to the brow, so we keep it for reference only.)
  const centre = { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 };

  // 2. WHICH WAY: angle of the eye line. atan2(dy, dx) returns the angle of the
  // vector a→b in radians; 0 = level, positive = tilted clockwise on canvas.
  const dx = b.x - a.x;
  const dy = b.y - a.y;
  const angle = Math.atan2(dy, dx);

  // 3. HOW BIG: Pythagoras gives the eye-corner distance in pixels. Convert it to
  // where the lens centres should be (×0.7, see LENS_TO_EYE_CORNER_RATIO), then
  // divide by the lens distance in the image to get the scale factor.
  const eyeCornerDist = Math.hypot(dx, dy);
  const targetLensDist = eyeCornerDist * LENS_TO_EYE_CORNER_RATIO;
  const imageLensDist = LENS_RIGHT.x - LENS_LEFT.x;
  const scale = targetLensDist / imageLensDist;

  // The point in the image that should land on `centre`: halfway between lenses.
  const imgCentre = { x: (LENS_LEFT.x + LENS_RIGHT.x) / 2, y: (LENS_LEFT.y + LENS_RIGHT.y) / 2 };

  // Canvas transforms apply in reverse order of how they're written, so read
  // these bottom-up for what happens to each image pixel:
  //   drawImage at -imgCentre → image's lens midpoint now sits at the origin
  //   scale   → resize around that point
  //   rotate  → tilt around that point
  //   translate → move that point onto the face's eye-line centre
  // save/restore keeps these transforms from leaking into later drawing.
  ctx.save();
  ctx.translate(centre.x, centre.y);
  ctx.rotate(angle);
  ctx.scale(scale, scale);
  ctx.drawImage(glasses.img, -imgCentre.x, -imgCentre.y);
  ctx.restore();
}

// -----------------------------------------------------------------------------
// Debug drawing
// -----------------------------------------------------------------------------
function drawLandmarks(face) {
  const w = canvas.width;
  const h = canvas.height;

  // All 478 points as small grey dots: shows the mesh is tracking the face.
  ctx.fillStyle = 'rgba(200, 200, 200, 0.6)';
  for (const p of face) {
    ctx.fillRect(p.x * w - 1, p.y * h - 1, 2, 2); // 0-1 coords → pixels
  }

  // Key points: big orange dots with their index, so we can confirm each one.
  ctx.font = '14px system-ui, sans-serif';
  for (const [index, name] of Object.entries(KEY_POINTS)) {
    const p = face[index];
    const x = p.x * w;
    const y = p.y * h;

    ctx.fillStyle = '#f5a623';
    ctx.beginPath();
    ctx.arc(x, y, 5, 0, Math.PI * 2);
    ctx.fill();

    // The canvas is mirrored by CSS (selfie view), which would make text read
    // backwards. Flip the drawing horizontally just for the label, so after
    // CSS flips it again it reads normally. -x puts it back at the same spot.
    ctx.save();
    ctx.scale(-1, 1);
    ctx.fillStyle = '#fff';
    ctx.fillText(`${index} ${name}`, -x + 8, y - 8);
    ctx.restore();
  }
}

// -----------------------------------------------------------------------------
// UI events
// -----------------------------------------------------------------------------
cameraBtn.addEventListener('click', () => (stream ? stopCamera() : startCamera()));
// The top-bar stop button only exists while the camera runs.
document.getElementById('stopBtn').addEventListener('click', stopCamera);

// Glasses picker. Each button's data-id matches a key in GLASSES. The next
// frame drawn picks up the new selection automatically.
const glassesList = document.getElementById('glassesList');
const glassesButtons = [...glassesList.querySelectorAll('button')];

function selectGlasses(btn) {
  // aria-pressed tells screen readers (and our CSS) which button is active.
  for (const b of glassesButtons) b.setAttribute('aria-pressed', b === btn);
  // The tray's sliding highlight moves to this button's slot (CSS reads data-active).
  glassesList.dataset.active = glassesButtons.indexOf(btn);
  selectedGlasses = btn.dataset.id;
  statGlasses.textContent = selectedGlasses;
  if (!use3D.disabled) setDesign(selectedGlasses); // show the matching 3D model
}

// One listener on the container handles all buttons ("event delegation").
glassesList.addEventListener('click', (e) => {
  const btn = e.target.closest('button');
  if (btn) selectGlasses(btn);
});

// Face status in the top bar (green dot / "looking for your face" hint). Only
// write to the DOM when it changes: this runs every frame, and touching a
// body attribute makes the browser re-check styles for the whole page.
let faceShown = false;
function setFaceState(found) {
  if (found === faceShown) return;
  faceShown = found;
  document.body.dataset.face = found ? 'yes' : 'no';
}

// Developer drawer: hidden by default so the try-on looks like a product, one
// click (or the D key) away for the engineering tools. `inert` removes it from
// keyboard focus and screen readers while it's closed.
const devDrawer = document.getElementById('devDrawer');
const devBtn = document.getElementById('devBtn');
function setDrawer(open) {
  devDrawer.toggleAttribute('data-open', open);
  devDrawer.inert = !open;
  devBtn.setAttribute('aria-expanded', open);
  if (!open && devDrawer.contains(document.activeElement)) devBtn.focus();
}
devBtn.addEventListener('click', () => setDrawer(!devDrawer.hasAttribute('data-open')));
document.getElementById('devClose').addEventListener('click', () => setDrawer(false));

// Keyboard shortcuts: 1-3 pick glasses, D toggles the drawer, Escape closes it.
// Ignored while typing in a field, so the Scenario box still accepts "d" and digits.
document.addEventListener('keydown', (e) => {
  if (e.target.closest('input, select, textarea') || e.ctrlKey || e.metaKey || e.altKey) return;
  if (e.key === 'Escape') setDrawer(false);
  else if (e.key === 'd' || e.key === 'D') setDrawer(!devDrawer.hasAttribute('data-open'));
  else if (['1', '2', '3'].includes(e.key)) selectGlasses(glassesButtons[Number(e.key) - 1]);
});

// Measurement recording. Settings that affect performance are saved with the
// results, so every CSV states exactly what was being tested.
recordBtn.addEventListener('click', () => {
  if (isRecording()) return;
  setConditions({
    resolution: `${video.videoWidth}x${video.videoHeight}`,
    cameraFrameRateSetting: cameraFps, // configured rate; summary.cameraFps is what it really delivered
    camera: cameraInfo, // supported ranges + current exposure settings (see showCameraInfo)
    delegate: statDelegate.textContent,
    mode: use3D.checked ? '3D' : '2D',
    glasses: selectedGlasses,
    smoothing: smoothing.checked ? { ...SMOOTHING } : 'off',
    matchLighting: matchLighting.checked,
    showLandmarks: showLandmarks.checked,
  });
  recordBtn.disabled = true;
  summaryOut.hidden = true;
  recordStatus.textContent = 'Recording… 0%';
  startRecording(Number(durationSelect.value), scenarioInput.value.trim(), (summary) => {
    recordBtn.disabled = false;
    recordStatus.textContent = 'Done: CSV downloaded';
    // Show the summary on the page too, ready to copy into the README.
    summaryOut.textContent = JSON.stringify(summary, null, 2);
    summaryOut.hidden = false;
  });
});

// Turning room-matching off restores the fixed default lighting, so the two
// can be compared side by side.
matchLighting.addEventListener('change', () => {
  if (use3D.disabled || matchLighting.checked) return;
  setLighting(null);
  statLight.textContent = 'fixed';
});

// Debug: show the invisible head occluder as a green wireframe to check its fit.
showOccluder.addEventListener('change', () => !use3D.disabled && setOccluderVisible(showOccluder.checked));

// Debug sliders for tuning the head fit live. Once good values are found they
// get baked in as defaults in glasses3d.js. 'input' fires continuously while
// dragging, so the effect is visible immediately.
const pitch = document.getElementById('pitch');
const pitchOut = document.getElementById('pitchOut');
pitch.value = PITCH_DEFAULT_DEG; // start the slider at the code default
pitchOut.textContent = `${PITCH_DEFAULT_DEG}°`;
pitch.addEventListener('input', () => {
  setPitchCorrection(Number(pitch.value));
  pitchOut.textContent = `${pitch.value}°`;
});

// Smoothing sliders write straight into the shared SMOOTHING object that every
// filter reads, so changes apply on the very next frame.
const minCutoff = document.getElementById('minCutoff');
const minCutoffOut = document.getElementById('minCutoffOut');
const beta = document.getElementById('beta');
const betaOut = document.getElementById('betaOut');
minCutoff.value = SMOOTHING.minCutoff;
beta.value = SMOOTHING.beta;
const showSmoothing = () => {
  minCutoffOut.textContent = `${SMOOTHING.minCutoff} Hz`;
  betaOut.textContent = `${SMOOTHING.beta}`;
};
showSmoothing();
minCutoff.addEventListener('input', () => { SMOOTHING.minCutoff = Number(minCutoff.value); showSmoothing(); });
beta.addEventListener('input', () => { SMOOTHING.beta = Number(beta.value); showSmoothing(); });

const occHeight = document.getElementById('occHeight');
const occHeightOut = document.getElementById('occHeightOut');
occHeight.value = OCCLUDER_DEFAULT_Y_MM; // start the slider at the code default
occHeightOut.textContent = `${OCCLUDER_DEFAULT_Y_MM} mm`;
occHeight.addEventListener('input', () => {
  if (!use3D.disabled) setOccluderHeight(Number(occHeight.value));
  occHeightOut.textContent = `${occHeight.value} mm`;
});
