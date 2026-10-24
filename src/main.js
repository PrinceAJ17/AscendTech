const video = document.getElementById('video');
const stage = document.querySelector('.stage');
const placeholder = document.getElementById('placeholder');
const cameraBtn = document.getElementById('cameraBtn');
const statusEl = document.getElementById('status');
const statRes = document.getElementById('statRes');
const statFps = document.getElementById('statFps');
const statGlasses = document.getElementById('statGlasses');

let stream = null;
let frameHandle = null;

function setStatus(text, isError = false) {
  statusEl.textContent = text;
  statusEl.classList.toggle('error', isError);
}

// Map getUserMedia errors to messages a user can act on
function cameraErrorMessage(err) {
  switch (err.name) {
    case 'NotAllowedError': return 'Camera permission denied. Allow it in the browser address bar.';
    case 'NotFoundError': return 'No camera found.';
    case 'NotReadableError': return 'Camera is in use by another app.';
    default: return `Camera error: ${err.message}`;
  }
}

async function startCamera() {
  if (!navigator.mediaDevices?.getUserMedia) {
    setStatus('This browser does not support camera access (needs HTTPS or localhost).', true);
    return;
  }
  setStatus('Requesting camera…');
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: 'user' },
      audio: false,
    });
  } catch (err) {
    setStatus(cameraErrorMessage(err), true);
    return;
  }
  video.srcObject = stream;
  await video.play();

  const { width, height } = stream.getVideoTracks()[0].getSettings();
  // Match the stage to the camera's real aspect ratio so nothing gets cropped
  stage.style.aspectRatio = `${video.videoWidth} / ${video.videoHeight}`;
  statRes.textContent = `${width}×${height}`;
  placeholder.hidden = true;
  cameraBtn.textContent = 'Stop camera';
  setStatus('Camera running');
  countFrames();
}

function stopCamera() {
  if (frameHandle !== null) video.cancelVideoFrameCallback(frameHandle);
  stream.getTracks().forEach((t) => t.stop());
  stream = null;
  video.srcObject = null;
  placeholder.hidden = false;
  cameraBtn.textContent = 'Start camera';
  statRes.textContent = statFps.textContent = '–';
  setStatus('');
}

// Counts frames the camera actually delivers (not screen refreshes), averaged over 1 s
function countFrames() {
  if (!('requestVideoFrameCallback' in video)) {
    statFps.textContent = 'n/a';
    return;
  }
  let frames = 0;
  let windowStart = performance.now();
  const onFrame = (now) => {
    frames++;
    if (now - windowStart >= 1000) {
      statFps.textContent = (frames * 1000 / (now - windowStart)).toFixed(1);
      frames = 0;
      windowStart = now;
    }
    frameHandle = video.requestVideoFrameCallback(onFrame);
  };
  frameHandle = video.requestVideoFrameCallback(onFrame);
}

cameraBtn.addEventListener('click', () => (stream ? stopCamera() : startCamera()));

document.getElementById('glassesList').addEventListener('click', (e) => {
  const btn = e.target.closest('button');
  if (!btn) return;
  for (const b of e.currentTarget.children) b.setAttribute('aria-pressed', b === btn);
  statGlasses.textContent = btn.dataset.id;
});
