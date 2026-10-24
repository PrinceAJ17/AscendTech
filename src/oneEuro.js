// =============================================================================
// One Euro filter (Casiez, Roussel & Vogel, CHI 2012)
//
// A smoothing filter whose strength adapts to speed:
//   - signal barely changing (head still) → heavy smoothing → no jitter
//   - signal changing fast (head moving)  → light smoothing → no lag
//
// It smooths ONE number over time. We create one filter per number we want
// smooth (x, y and z of each key landmark).
// =============================================================================

// How much of the new value to take each step, for a given cutoff frequency.
// Think of the cutoff (in Hz) as "changes slower than this pass through,
// faster ones (jitter) get damped". tau is the matching time constant.
//   high cutoff → small tau → alpha near 1 → follow the new value closely
//   low cutoff  → large tau → alpha near 0 → mostly keep the old value
// Using the real time step dt (not "one frame") keeps behaviour the same at
// 15 FPS and 30 FPS.
function alpha(cutoff, dt) {
  const tau = 1 / (2 * Math.PI * cutoff);
  return 1 / (1 + tau / dt);
}

// Exponential smoothing: blend the previous smoothed value toward the new one.
function lowpass(previous, value, a) {
  return a * value + (1 - a) * previous;
}

export class OneEuroFilter {
  // `params` is a shared object { minCutoff, beta, dCutoff }. It's passed by
  // reference, so the tuning sliders can change it and every filter sees the
  // new values immediately.
  //   minCutoff (Hz): smoothing when still. Lower = less jitter, more lag.
  //   beta:           how fast smoothing relaxes with speed. Higher = less lag when moving.
  //   dCutoff (Hz):   smoothing of the speed estimate itself (rarely needs changing).
  constructor(params) {
    this.params = params;
    this.reset();
  }

  // Forget history. Called when the face is lost, so when it comes back the
  // glasses snap onto it instead of gliding over from the old position.
  reset() {
    this.x = null;  // last smoothed value
    this.dx = 0;    // last smoothed speed
    this.t = null;  // time of the last sample (seconds)
  }

  // value: the new raw measurement. tMs: its timestamp in milliseconds.
  filter(value, tMs) {
    const t = tMs / 1000;
    // First sample after a reset: nothing to smooth against yet.
    if (this.x === null) {
      this.x = value;
      this.t = t;
      return value;
    }
    const dt = t - this.t;
    if (dt <= 0) return this.x; // same or out-of-order timestamp: keep the last value
    this.t = t;

    // 1. Estimate how fast the value is changing (units per second), and smooth
    // that estimate too, otherwise jitter would look like "fast movement".
    const speed = (value - this.x) / dt;
    this.dx = lowpass(this.dx, speed, alpha(this.params.dCutoff, dt));

    // 2. The key idea: raise the cutoff with speed. Still → minCutoff (heavy
    // smoothing). Fast → minCutoff + beta·speed (light smoothing).
    const cutoff = this.params.minCutoff + this.params.beta * Math.abs(this.dx);

    // 3. Smooth the value with that cutoff.
    this.x = lowpass(this.x, value, alpha(cutoff, dt));
    return this.x;
  }
}
