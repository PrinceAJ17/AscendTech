// =============================================================================
// Glasses Try-On — step 1: webcam + MediaPipe face landmarks
//
// Pipeline for every camera frame:
//   webcam frame → MediaPipe Face Landmarker → 478 landmarks → draw on canvas
//
// Later steps will replace "draw dots" with "place and render glasses".
// =============================================================================

// FilesetResolver loads MediaPipe's WebAssembly runtime (the compiled C++ that
// actually runs the model). FaceLandmarker is the face-tracking "Task" itself.
import { FilesetResolver, FaceLandmarker } from '@mediapipe/tasks-vision';

// -----------------------------------------------------------------------------
// Where the runtime files come from
// -----------------------------------------------------------------------------
// shortcut: both files load from CDNs, so the app needs internet on first load;
// copy them into public/ before the final demo so it doesn't depend on others' servers.

// The .wasm runtime. The version MUST match the npm package we installed
// (see package.json), otherwise the JS and the wasm won't talk to each other.
const WASM_URL = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.1.1/wasm';

// The trained model bundle: BlazeFace detector + face landmark model (+ blendshapes).
// "float16" = weights stored as 16-bit floats: half the size of float32, same accuracy in practice.
const MODEL_URL =
  'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task';

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
// Grab the HTML elements once, up front
// -----------------------------------------------------------------------------
const video = document.getElementById('video');
const canvas = document.getElementById('overlay');
const ctx = canvas.getContext('2d'); // 2D drawing API for the debug dots
const stage = document.querySelector('.stage');
const placeholder = document.getElementById('placeholder');
const cameraBtn = document.getElementById('cameraBtn');
const statusEl = document.getElementById('status');
const showLandmarks = document.getElementById('showLandmarks');
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
let frameHandle = null; // id of the pending per-frame callback, so we can cancel it
let landmarker = null;  // the FaceLandmarker, created once and reused

function setStatus(text, isError = false) {
  statusEl.textContent = text;
  statusEl.classList.toggle('error', isError);
}

// -----------------------------------------------------------------------------
// Model loading
// -----------------------------------------------------------------------------

// Creates the FaceLandmarker on the requested hardware ("GPU" or "CPU").
async function createLandmarker(delegate) {
  const fileset = await FilesetResolver.forVisionTasks(WASM_URL);
  return FaceLandmarker.createFromOptions(fileset, {
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
    // Ask for 720p from the front camera. "ideal" means "if possible": the
    // browser picks the closest the camera supports instead of failing.
    stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: 'user' },
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

  // Match the stage to the camera's real aspect ratio (e.g. 16:9) so nothing is cropped.
  stage.style.aspectRatio = `${video.videoWidth} / ${video.videoHeight}`;
  // Make the canvas's drawing resolution equal the video's pixel size. Then a
  // landmark at (x, y) in 0-1 coordinates is simply (x * width, y * height) on
  // the canvas, and CSS stretches both layers identically on screen.
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;

  statRes.textContent = `${video.videoWidth}×${video.videoHeight}`;
  placeholder.hidden = true;
  cameraBtn.textContent = 'Stop camera';
  cameraBtn.disabled = false;
  setStatus('Camera running');
  startFrameLoop();
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
  placeholder.hidden = false;
  cameraBtn.textContent = 'Start camera';
  statRes.textContent = statFps.textContent = statInference.textContent = statFace.textContent = '–';
  setStatus('');
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
  let windowStart = performance.now();

  // `now` is a high-resolution timestamp (ms) for this frame. It always
  // increases, which is exactly what VIDEO mode needs to track between frames.
  const onFrame = (now) => {
    // --- 1. Run the model and time it -------------------------------------
    const t0 = performance.now();
    const result = landmarker.detectForVideo(video, now);
    inferenceTotal += performance.now() - t0;
    frames++;

    // result.faceLandmarks is an array with one entry per face found (we asked
    // for max 1). Each entry is 478 points of {x, y, z}:
    //   x, y: position as a fraction of image width/height (0 to 1)
    //   z:    depth relative to the head's centre, roughly in the same units as x
    const face = result.faceLandmarks[0]; // undefined when no face is visible
    statFace.textContent = face ? 'yes' : 'no';

    // --- 2. Draw ------------------------------------------------------------
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (face && showLandmarks.checked) drawLandmarks(face);

    // --- 3. Update stats once per second ------------------------------------
    // Averaging over a window gives a readable number instead of a flickering one.
    const elapsed = now - windowStart;
    if (elapsed >= 1000) {
      statFps.textContent = (frames * 1000 / elapsed).toFixed(1);
      statInference.textContent = `${(inferenceTotal / frames).toFixed(1)} ms`;
      frames = 0;
      inferenceTotal = 0;
      windowStart = now;
    }

    // Ask to be called again on the next camera frame.
    frameHandle = video.requestVideoFrameCallback(onFrame);
  };
  frameHandle = video.requestVideoFrameCallback(onFrame);
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

// Glasses picker: only highlights the choice for now; step 2 will load the
// matching glasses image/model. One listener on the container handles all buttons.
document.getElementById('glassesList').addEventListener('click', (e) => {
  const btn = e.target.closest('button');
  if (!btn) return;
  for (const b of e.currentTarget.children) b.setAttribute('aria-pressed', b === btn);
  statGlasses.textContent = btn.dataset.id;
});
