// Self-check for the One Euro filter: run with `npm test`.
// Feeds synthetic signals and checks it (1) cuts jitter when still, (2) keeps lag low when moving, (3) resets cleanly.
import assert from 'node:assert';
import { OneEuroFilter } from '../src/oneEuro.js';
const P = { minCutoff: 1, beta: 10, dCutoff: 1 };
let seed = 1; const rand = () => ((seed = (seed * 16807) % 2147483647) / 2147483647 - 0.5);
const sd = (a) => { const m = a.reduce((s, x) => s + x, 0) / a.length; return Math.sqrt(a.reduce((s, x) => s + (x - m) ** 2, 0) / a.length); };
// 1. still head + noise (0.003 of frame width ≈ 2 px at 640) at 30 FPS
let f = new OneEuroFilter(P), raw = [], out = [];
for (let i = 0; i < 300; i++) { const v = 0.5 + rand() * 0.006; raw.push(v); out.push(f.filter(v, i * 33.3)); }
const r = sd(raw.slice(60)), s = sd(out.slice(60));
console.log('still: raw sd', r.toFixed(5), 'smoothed sd', s.toFixed(5), 'reduction', (r / s).toFixed(1) + 'x');
assert(s < r / 2);
// 2. fast motion: 0.8 frame widths/s ramp, measure lag
f = new OneEuroFilter(P); let last;
for (let i = 0; i < 30; i++) last = { v: i * 0.8 / 30, o: f.filter(i * 0.8 / 30, i * 33.3) };
const lagMs = (last.v - last.o) / 0.8 * 1000;
console.log('moving: lag', lagMs.toFixed(1), 'ms');
assert(lagMs < 50);
// 3. reset: first value after reset returned as-is
f.reset(); assert.strictEqual(f.filter(0.123, 5000), 0.123);
console.log('reset ok');
