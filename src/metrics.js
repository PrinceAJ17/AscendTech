// =============================================================================
// Step 7: performance measurement and stress-test recording
//
// The live stats panel is good for watching, but the brief asks us to REPORT
// performance with test conditions. So this module records every frame for a
// fixed time (e.g. 30 s), then:
//   1. computes a summary: FPS, inference/render/latency (mean, median, p95),
//      face-loss rate, jitter, and the head angles reached
//   2. records the test conditions (browser, GPU, resolution, settings)
//   3. downloads everything as a CSV file, one row per frame
//
// A "scenario" label (e.g. "dim light", "head turns") is saved with each
// recording, so each stress test from the plan produces its own labelled file.
// =============================================================================

let rows = [];          // one object per frame while recording
let recording = false;
let startTime = 0;      // timestamp of the first frame (ms)
let durationMs = 0;
let scenario = '';
let onDone = null;      // called with the summary when the recording ends

export function isRecording() {
  return recording;
}

// Begin a new recording. `done(summary, conditions)` fires when time is up.
export function startRecording(seconds, label, done) {
  rows = [];
  recording = true;
  startTime = 0; // set on the first frame, so the clock starts with real data
  durationMs = seconds * 1000;
  scenario = label || 'unlabelled';
  onDone = done;
}

// Abandon the current recording without saving (e.g. the camera was stopped).
export function cancelRecording() {
  recording = false;
  rows = [];
}

// Called once per processed frame by the frame loop with that frame's numbers.
// Returns how far through the recording we are (0-1), for a progress display.
export function recordFrame(row) {
  if (!recording) return 0;
  if (!startTime) startTime = row.now;
  const t = row.now - startTime;
  rows.push({ t, ...row });
  if (t >= durationMs) finish();
  return Math.min(1, t / durationMs);
}

function finish() {
  recording = false;
  const summary = summarize(rows);
  const conditions = testConditions();
  downloadCsv(rows, summary, conditions);
  onDone?.(summary, conditions);
}

// -----------------------------------------------------------------------------
// Summary statistics
// -----------------------------------------------------------------------------
// Why mean AND median AND p95?
//   mean:   the average, but one 200 ms hiccup drags it up
//   median: the typical frame (half are faster, half slower)
//   p95:    "95% of frames were at least this fast": shows the bad moments
//           that users actually notice as stutter
function stats(values) {
  const v = values.filter((x) => x !== null && Number.isFinite(x)).sort((a, b) => a - b);
  if (!v.length) return null;
  const pick = (p) => v[Math.min(v.length - 1, Math.ceil(p * v.length) - 1)];
  return {
    mean: round(v.reduce((s, x) => s + x, 0) / v.length),
    median: round(pick(0.5)),
    p95: round(pick(0.95)),
    max: round(v[v.length - 1]),
  };
}

// 2D standard deviation of points (same idea as the live jitter stat).
function spread(points) {
  if (points.length < 2) return null;
  const mx = points.reduce((s, p) => s + p.x, 0) / points.length;
  const my = points.reduce((s, p) => s + p.y, 0) / points.length;
  return round(Math.sqrt(points.reduce((s, p) => s + (p.x - mx) ** 2 + (p.y - my) ** 2, 0) / points.length));
}

export function summarize(data) {
  if (data.length < 2) return null;
  const seconds = (data[data.length - 1].t - data[0].t) / 1000;
  const withFace = data.filter((r) => r.face);

  // Processed FPS: frames we actually ran the model on, per second.
  // Camera FPS: frames the camera delivered, from the browser's running count
  // (presentedFrames). If processed < camera, frames were skipped because our
  // processing couldn't keep up.
  const presented = data[data.length - 1].presentedFrames - data[0].presentedFrames;

  // Per-second FPS buckets, so we can report the WORST second, not just the average.
  const perSecond = {};
  for (const r of data) perSecond[Math.floor(r.t / 1000)] = (perSecond[Math.floor(r.t / 1000)] || 0) + 1;
  const fullSeconds = Object.entries(perSecond).filter(([s]) => Number(s) < Math.floor(seconds)).map(([, n]) => n);

  return {
    scenario,
    durationS: round(seconds),
    frames: data.length,
    processedFps: round((data.length - 1) / seconds),
    cameraFps: round(presented / seconds),
    minFpsInAnySecond: fullSeconds.length ? Math.min(...fullSeconds) : null,
    inferenceMs: stats(data.map((r) => r.inferenceMs)),
    renderMs: stats(data.map((r) => r.renderMs)),
    // Capture → render: from the moment the camera captured the frame to when
    // we finished drawing it. Excludes the final wait for the screen refresh
    // (up to ~16 ms more), so true end-to-end is a bit higher.
    latencyMs: stats(data.map((r) => r.latencyMs)),
    faceLossPercent: round((1 - withFace.length / data.length) * 100),
    // Jitter only means something in a "hold still" scenario: any real movement counts as spread too.
    jitterRawPx: spread(withFace.map((r) => ({ x: r.rawCx, y: r.rawCy }))),
    jitterSmoothedPx: spread(withFace.map((r) => ({ x: r.smoothCx, y: r.smoothCy }))),
    // Largest head angles reached while the face was still tracked: e.g. for
    // a "turn until it fails" test this is the tracking limit.
    maxAbsYawDeg: maxAbs(withFace.map((r) => r.yaw)),
    maxAbsPitchDeg: maxAbs(withFace.map((r) => r.pitch)),
    maxAbsRollDeg: maxAbs(withFace.map((r) => r.roll)),
    faceWidthPx: stats(withFace.map((r) => r.faceWidthPx)),
  };
}

function maxAbs(values) {
  const v = values.filter((x) => x !== null && Number.isFinite(x)).map(Math.abs);
  return v.length ? round(Math.max(...v)) : null;
}

function round(x) {
  return Math.round(x * 100) / 100;
}

// -----------------------------------------------------------------------------
// Test conditions: the "on what machine, in what setup" part of the report
// -----------------------------------------------------------------------------
// Settings from the page (mode, resolution, delegate...) are added by main.js;
// this collects what the browser can tell us about the machine.
let extraConditions = {};
export function setConditions(c) {
  extraConditions = c;
}

function testConditions() {
  return {
    date: new Date().toISOString(),
    userAgent: navigator.userAgent,
    cpuThreads: navigator.hardwareConcurrency,
    gpu: gpuName(),
    ...extraConditions,
  };
}

// The GPU model, via a WebGL debug extension. Some browsers hide or blur it
// for privacy, in which case we just report "unknown".
function gpuName() {
  try {
    const gl = document.createElement('canvas').getContext('webgl');
    const ext = gl.getExtension('WEBGL_debug_renderer_info');
    return ext ? gl.getParameter(ext.UNMASKED_RENDERER_WEBGL) : 'unknown';
  } catch {
    return 'unknown';
  }
}

// -----------------------------------------------------------------------------
// CSV export
// -----------------------------------------------------------------------------
// Layout: a few "#" comment lines with the conditions and summary (so one
// file is self-explanatory), then a header row and one row per frame.
// Spreadsheets show the "#" lines as plain rows above the table.
const COLUMNS = ['t', 'presentedFrames', 'face', 'inferenceMs', 'renderMs', 'latencyMs',
  'yaw', 'pitch', 'roll', 'faceWidthPx', 'rawCx', 'rawCy', 'smoothCx', 'smoothCy'];

function downloadCsv(data, summary, conditions) {
  const lines = [
    `# conditions: ${JSON.stringify(conditions)}`,
    `# summary: ${JSON.stringify(summary)}`,
    COLUMNS.join(','),
    ...data.map((r) => COLUMNS.map((c) => fmt(r[c])).join(',')),
  ];
  // Build the file in memory (a Blob), point a temporary link at it and click
  // it: that's how a web page hands the user a file to save.
  const blob = new Blob([lines.join('\n')], { type: 'text/csv' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = `tryon-${scenario.replace(/[^a-z0-9]+/gi, '-').toLowerCase()}-${Date.now()}.csv`;
  a.click();
  // Free the memory once the download has had time to start.
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}

function fmt(v) {
  if (v === null || v === undefined) return '';
  if (typeof v === 'boolean') return v ? 1 : 0;
  return typeof v === 'number' ? round(v) : v;
}
