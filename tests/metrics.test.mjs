// Self-check for the measurement summary: run with `npm test`.
// Builds 10 s of fake frames with known numbers and checks the summary reports them.
import assert from 'node:assert';
import { summarize } from '../src/metrics.js';

const rows = [];
let presented = 0;
for (let i = 0; i < 300; i++) {
  presented += i % 10 === 0 ? 2 : 1; // camera delivered an extra frame every 10th: we skipped it
  rows.push({
    t: i * (10000 / 299),     // 300 frames spread over exactly 10 s
    presentedFrames: presented,
    face: i % 20 !== 0,       // face lost on 1 in 20 frames = 5%
    inferenceMs: i < 285 ? 10 : 40, // 95% at 10 ms, the slowest 5% at 40 ms
    renderMs: 2,
    latencyMs: 50,
    yaw: i === 101 ? -48 : 5, // one big head turn (on a frame where the face IS tracked)
    pitch: 0,
    roll: 0,
    faceWidthPx: 200,
    rawCx: 100, rawCy: 100, smoothCx: 100, smoothCy: 100,
  });
}

const s = summarize(rows);
assert.strictEqual(s.frames, 300);
assert.strictEqual(s.durationS, 10);
assert.strictEqual(s.processedFps, 29.9);              // 299 frame intervals / 10 s
assert.ok(s.cameraFps > s.processedFps);               // skipped frames show up
assert.strictEqual(s.inferenceMs.median, 10);
assert.strictEqual(s.inferenceMs.p95, 10);             // exactly 95% are 10 ms
assert.strictEqual(s.inferenceMs.max, 40);
assert.strictEqual(s.faceLossPercent, 5);
assert.strictEqual(s.maxAbsYawDeg, 48);
assert.strictEqual(s.jitterRawPx, 0);                  // perfectly still points
console.log('metrics summary ok:', JSON.stringify({ fps: s.processedFps, camFps: s.cameraFps, minFps: s.minFpsInAnySecond }));
