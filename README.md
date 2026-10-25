# Glasses Try-On

A browser app that lets you try on eyeglasses live on your webcam. It tracks your face in real time with MediaPipe and renders 3D glasses with Three.js, entirely on your device.

Built for the AscendTech AI Engineer Intern technical assessment, using **Option A (build from scratch)** with open-source frameworks.

- **Live demo:** https://glasses-tryon-nine.vercel.app (allow camera access when asked; works best in Chrome or Edge)
- **Demo video:** _TODO: [add link_](https://youtu.be/SF9xctutORM)
- **Full reasoning, decisions and test notes:** [ThoughtProcess.md](ThoughtProcess.md)
- **AI tools and prompts used:** [AI_PROMPTS.md](AI_PROMPTS.md)

## Features

- Live webcam try-on with **three glasses designs** (round, rectangle, aviator), switchable instantly.
- **3D glasses** that turn, nod and tilt with your head. The arms disappear behind your head thanks to an invisible head "occluder".
- **Room-light matching:** the glasses' brightness, light colour and light direction are estimated from the webcam image.
- A **2D mode** (flat image overlay), kept as a baseline for comparison.
- **Full-screen mirror UI** with floating frosted-glass controls: a frame tray with a sliding highlight, a live FPS / face-status pill, and a "looking for your face" hint. Keys **1–3** switch frames.
- A **developer drawer** (button top right, or the **D** key; hidden by default): live stats, display toggles (3D/2D, landmarks, occluder wireframe), fit and smoothing sliders, the measurement recorder, and camera capabilities.
- **Built-in measurement:** live FPS / inference / render / latency / jitter stats, plus a recorder that saves 10–60 s of per-frame data with the test conditions as a CSV.

## How to run

Requires Node.js 18+ and a browser with WebGL2 (recent Chrome, Edge, Firefox or Safari).

```bash
npm install
npm run dev      # then open http://localhost:5173 and click "Start camera"
npm test         # self-checks for the smoothing filter and the measurement summary
npm run build    # production build in dist/
```

The camera only works on `localhost` or HTTPS (a browser security rule).

## Technical approach

```
webcam frame
  → MediaPipe Face Landmarker (478 3D face landmarks, on the GPU)
  → head pose from 4 landmarks (eye corners, forehead, chin)
  → [optional One Euro smoothing of those landmarks]
  → Three.js: move glasses + invisible head occluder to the pose, match room lighting
  → transparent 3D canvas drawn over the mirrored video
```

1. **Frame loop.** `requestVideoFrameCallback` runs once per *new* camera frame. The model never runs twice on the same frame, and old frames never queue up, so latency can't build up.
2. **Face tracking.** MediaPipe Face Landmarker in `VIDEO` mode. After the first detection it tracks from the previous frame's position instead of re-running the detector. It tries the GPU first and falls back to the CPU.
3. **Head pose.**
   - The outer eye corners (33, 263) give the left-right axis.
   - Chin → forehead (152 → 10) gives "up".
   - A cross product gives "out of the face".
   - Scale is pixels per millimetre, from the *3D* eye-corner distance, so the glasses don't shrink when you turn.
   - A small pitch correction (−2°) cancels the forehead's backward slope.
4. **Rendering.**
   - The glasses are built in code at real size (mm) from lens outlines: rims, lenses, bridge, arms.
   - An orthographic camera maps 1 unit to 1 video pixel, so the glasses line up with the landmarks by construction. (MediaPipe's own head-pose matrix would have needed an exact field-of-view match with its assumed camera.)
   - Two invisible occluders write depth but no colour, so arms behind the head are hidden: a head-shaped ellipsoid for the back of the head, and MediaPipe's live 468-point face mesh for the front (so the far arm doesn't show through the far lens when you turn).
   - Rendering happens in two passes: first the occluders and arms, then (after clearing depth) the rims, lenses and bridge, which sit in front of the face on real glasses. The face mesh can then hug the face without the nose clipping a rim.
   - Lenses use transparency; frames use physically based materials with an environment map for reflections.
5. **Lighting.** About 7 times a second the video is shrunk to 64×48 pixels.
   - Brightness and colour come from the whole frame. Measuring the face would make glasses darker on darker skin.
   - Light direction comes from the left/right halves of the face, where skin tone cancels out.
6. **Smoothing.** A One Euro filter is available but **off by default**. Measured raw jitter was already ~0.2–0.4 px, and smoothing added visible lag.

## APIs, models and frameworks

| What | Version | Role | Licence |
|---|---|---|---|
| [MediaPipe Tasks Vision](https://developers.google.com/mediapipe) (`@mediapipe/tasks-vision`) | 1.1.1 | Face Landmarker runtime (WebAssembly + WebGL) | Apache-2.0 |
| Face Landmarker model (`face_landmarker.task`, float16) | v1 | BlazeFace detector + 478-point face mesh, served from `public/models/` | Apache-2.0 |
| [Three.js](https://threejs.org) | 0.186 | 3D rendering, PBR materials, environment map | MIT |
| [Vite](https://vite.dev) | 6 | Dev server and build | MIT |

There are no paid APIs and no server: everything runs in the browser, so **usage cost is zero** and no video leaves the device. The glasses models are generated in code, so there are no asset licences to track.

## Performance

Measured with the in-app recorder (30 s runs unless noted; full data and analysis in [ThoughtProcess.md](ThoughtProcess.md#test-results-step-7)).

**Test conditions:**
- Windows 11, Chrome 155
- NVIDIA RTX 5090 Laptop GPU, 24 CPU threads. A high-end machine, so these are best-case numbers.
- ASUS FHD webcam at 1280×720, MediaPipe on the GPU, 3D mode, room-light matching on.

| Metric | Target | Result |
|---|---|---|
| FPS | ≥ 24 | **19.9** in normal indoor light, **10** in dim light: limited by the **webcam** (see below) |
| Inference (MediaPipe) | | median **11.5 ms**, p95 14.2 ms |
| Render | | median **0.2 ms**; p95 10 ms (from the lighting estimate, see below) |
| Capture → render latency | < 100 ms | median **17 ms**, p95 27 ms |
| Frame-to-frame jitter (holding still, raw) | | **0.2–0.4 px** (normal and dim light) |
| Head-turn range before tracking is lost | | ~63–77° yaw (normal light) |
| 4× CPU throttle (simulated weaker machine) | | inference 37 ms, 18.3 FPS, latency median 50 ms / p95 84 ms |

- **FPS is camera-limited, not code-limited.**
  - Our processing takes ~12–15 ms per frame, which is capacity for 30+ FPS.
  - The webcam's auto-exposure holds each frame for ~50 ms indoors (~100 ms in dim light). Requesting 30 FPS had no effect.
  - Manual exposure was tried and rejected: the browser can't control sensor gain, so short exposures were too dark to track a face.
- **Latency** is measured from the camera's capture timestamp to the end of our drawing. It excludes the final screen refresh (up to ~16 ms) and camera exposure; true end-to-end (e.g. by slow-motion video) has not been measured yet.
- **The lighting estimate costs ~9 ms on about 1 in 4 frames,** because reading pixels back from the video waits for the GPU. That was confirmed by an A/B run with it off. It still fits a 30 FPS frame budget.

## Known limitations

- **Frame rate:** below the 24 FPS target on this webcam (20 FPS indoors, 10 in dim light) because of its auto-exposure. See above.
- **Dim light:** tracking at a distance fails (74–80% face loss). It holds up close.
- **Extreme head angles:** tracking drops beyond roughly 60–75° of yaw, less consistently in dim light.
- **One average head:**
  - Glasses size assumes an average eye-corner distance (90 mm). There's no per-user measurement in millimetres.
  - The back-of-head occluder is one average ellipsoid; only the face itself is fitted (live face mesh). Hair and ears aren't modelled, so an arm can disappear slightly early or late near the ear.
  - The pitch correction and occluder height were calibrated against real glasses on one face, and should be re-checked on more people.
- **Lighting estimate:** an approximation. A bright window behind the user makes the glasses look too bright, strongly coloured rooms can still shift the tint slightly (capped at ±20%), and there are no cast shadows.
- **No perspective inside the glasses:** a trade-off of the orthographic camera; the far arm isn't drawn smaller. Barely visible at webcam distance.
- **Procedural glasses models** are simpler than professional 3D models. Swapping in `.glb` files would only replace one function.
- **Browser requirements:** WebGL2 and WebAssembly SIMD (no fallback for very old browsers). Tested only in Chrome on Windows.
- **Camera settings reset:** on start, the camera is reset to auto exposure and mid brightness, which would undo a user's own camera adjustments. Kept as a safeguard (see ThoughtProcess.md).

## Project structure

```
index.html            page layout: video stage, controls, stats, measure panel
src/main.js           camera, MediaPipe, frame loop, 2D overlay, smoothing, stats, UI wiring
src/glasses3d.js      Three.js scene: head pose, glasses models, occluder, lighting
src/lighting.js       room brightness / tint / direction estimate from the video
src/oneEuro.js        One Euro smoothing filter
src/metrics.js        recording, summary statistics, CSV export
src/style.css         styling
public/glasses/       SVG images for the 2D mode
public/models/        MediaPipe face model
tests/                self-checks (npm test)
ThoughtProcess.md     reasoning, decisions, tuning notes and full test results
```

## AI tools

Developed with **Claude (Anthropic), via Claude Code in the Claude desktop app**. It was used to critique my initial approach, explain concepts, write and comment the code, and analyse the test recordings. All prompts are in [AI_PROMPTS.md](AI_PROMPTS.md). Design decisions, tuning values and test runs were made and checked by me on camera.
