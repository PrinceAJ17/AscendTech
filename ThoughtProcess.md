This will be a document/text file covering my thought process throughout the whole assignment.



We first ask ourselves what exactly is the question and what are we trying to achieve here?

***Develop an AI-powered application that allows users to virtually try on different eyeglasses using a live webcam. The application should provide a realistic experience with accurate glasses positioning, smooth facial tracking, and minimal processing delay.***



## Correcting my first draft (please note that I used claude to critique/validate my first approach and the following plan has been made by claude to explain to me personally where to go about with this project)

My first pass got a few things wrong, so before planning I'm noting them:

- **I misread the options.** Option A doesn't mean training my own models. It explicitly allows MediaPipe/OpenCV, so "from scratch" means building the try-on pipeline (landmarks → head pose → glasses rendering) myself. Option B means plugging in a ready-made try-on SDK. My "pretrained YOLO + place glasses myself" idea was really Option A anyway.
- **Wrong tools for the job.** ResNet50 classification and eigenfaces describe *who* a face is, not *where* its features are. Glasses placement needs facial landmarks, not embeddings.
- **YOLO isn't suited here.** COCO YOLOv8 has no face class, and YOLOv8-pose only gives ~5 face points, which is too few to fit glasses in 3D.
- **Jitter isn't mainly a CPU problem.** It comes from per-frame predictions with no temporal smoothing.
- **I had the difficulty backwards.** Detection is a solved problem. Fitting and rendering the glasses realistically is the real work.



## Plan: Option A

### 1. Platform: a browser app

The whole thing runs client-side in the browser (JavaScript/TypeScript, Vite).

*Why:* no server round-trip per frame, which is the single biggest win for the <100 ms latency target. It also gives me a free "working demo link" (static hosting on Vercel/GitHub Pages) and works on any laptop with a webcam. A Python + OpenCV desktop app would be easier to prototype but harder to share and to render realistically.

### 2. Face tracking: MediaPipe Face Landmarker (`@mediapipe/tasks-vision`)

- Running mode `VIDEO`, GPU delegate, `outputFacialTransformationMatrixes: true`.
- Gives 478 3D landmarks (including irises) plus a 4×4 head-pose matrix per frame.

*Why:* it's built for exactly this (real-time, on-device, runs at 30+ FPS on ordinary laptops) and is explicitly allowed in the brief. The transformation matrix gives me full 3D head pose (position, roll, pitch, yaw) without writing my own `solvePnP`. That pose is what lets glasses stay aligned when the head turns.

*Alternative considered:* OpenCV `solvePnP` on a handful of landmarks. It works, but it's more code to get the same result. I'll mention it in the write-up as the fallback / "how it works underneath".

*Changed during implementation:*
- The transformation matrix is still requested but not used. The head pose is built from landmarks instead (see section 3).
- Step 1 loaded MediaPipe's runtime and model from external servers (jsDelivr and Google) for speed of setup. Step 8 moved both into the app itself, so the deployed demo doesn't depend on anyone else's servers. The tracking behaviour is identical.

### 3. Placement: from landmarks to glasses transform

- **Anchor:** nose-bridge landmark (between the eyes). This is where real glasses rest.
- **Scale:** distance between the outer eye corners or temples, so frames fit different face widths.
- **Rotation:** taken from the transformation matrix (roll/pitch/yaw).
- Each glasses model gets a small per-model offset (how far forward it sits, bridge height), because frames differ.

*Why:* this mirrors how glasses physically sit on a face, and it's simple math I can explain line by line in the interview.

*Changed during implementation (step 3): head pose from landmarks, not MediaPipe's matrix.* The original plan used MediaPipe's head-pose matrix with a perspective camera. The risk: if our camera's field of view doesn't exactly match the one MediaPipe assumes, the glasses drift off the face near the image edges. Instead, the pose is built from four landmarks:

- Eye corners 33 and 263 give the left-right direction.
- Forehead 10 and chin 152 give "up".
- A cross product of the two gives "out of the face".

It's rendered with an orthographic camera where 1 unit = 1 video pixel, so the glasses line up with the landmarks by construction, the same way the 2D version already fits. Trade-off: no perspective inside the glasses (the far arm isn't drawn slightly smaller), which is barely visible at webcam distance.

*Other placement details settled during implementation:*
- **Anchor:** the midpoint of the outer eye corners rather than landmark 168. Landmark 168 sits slightly above the eye line, toward the brow; the eye-corner midpoint lines the lenses up with the eyes.
- **Scale:** the *3D* eye-corner distance (it includes depth), so the glasses no longer shrink when the head turns, as the 2D version did. It assumes an average 90 mm eye-corner distance, with lens centres at ±31.5 mm (a 63 mm pupil distance).
- **Frame offset:** glasses sit 18 mm in front of the eye-corner plane.
- **Pitch correction −2° and occluder height −12 mm:** calibrated by lining the virtual frames up against my real glasses on camera. Calibrated on one face only, so these should be re-checked on more people.

### 4. Smoothing: One Euro filter on the pose

Apply a One Euro filter to the position/rotation (or to the key landmarks) before rendering.

*Why:* raw per-frame predictions jitter. A plain moving average removes jitter but adds lag. The One Euro filter smooths heavily when the head is still and lightly when it moves fast, so it fixes jitter without hurting responsiveness. It's a standard choice for AR tracking.

*Note from testing (steps 4–5): smoothing may be doing more harm than good.*

- **What happened:**
  - Raw MediaPipe tracking turned out to be fairly steady on my setup.
  - At min cutoff 1 Hz the glasses visibly dragged across my face when I moved.
  - At 3 Hz the drag was smaller, with only very slight jitter, so 3 Hz / beta 30 is the current default.
  - Even with beta pushed to 50, there's still a noticeable drag.
- **Likely cause:** the filter's speed estimate is itself smoothed (`dCutoff` = 1 Hz), so at the start of every movement it still thinks the head is "slow" and smooths heavily for a moment. Beta can't help until the speed estimate catches up. Any smoothing also adds to the <100 ms latency budget.
- **To decide later (not changed yet):**
  - Measure raw vs smoothed jitter (the stats panel) in normal and dim light.
  - If raw jitter is small even in dim light, turn smoothing **off by default** and keep the toggle.
  - Otherwise, try a higher `dCutoff` (so it reacts faster to the start of movement) or a higher min cutoff.
  - Either way, the write-up should explain the choice with the measured numbers.
- **Resolved (step 7):** measured raw jitter was ~0.2–0.4 px even in dim light, so smoothing is now **off by default**, with the toggle kept. See Test results, finding 2.

### 5. Rendering: Three.js with 3D glasses models

- Webcam `<video>` as the background, mirrored (selfie view). The 3D scene is mirrored the same way so they stay consistent.
- Glasses as glTF/GLB models (at least 3 designs, e.g. round, rectangular, aviator), from CC-licensed sources such as Sketchfab or modelled in Blender. Licences go in the README.
- **Occluder:** an invisible head-shaped mesh that writes depth only (`colorWrite: false`), attached to the same head pose. It hides the temple arms behind the head and ears, which is the biggest single realism win.
- **Lenses:** `MeshPhysicalMaterial` with transparency/transmission and slight reflections.
- **Lighting:** environment map (`RoomEnvironment` + PMREM) for believable reflections on the frames. Stretch goal: estimate average brightness from the video frame and scale light intensity to match, for the "different lighting conditions" target.
- Camera field of view matched to MediaPipe's assumed perspective camera so the overlay lines up.

*Why 3D over a 2D PNG overlay:* a PNG is fine for a front view but goes flat and wrong as soon as the head turns. 3D plus an occluder handles yaw/pitch naturally. I'll build the PNG version first as a quick baseline, then replace it.

*Changed during implementation:*
- **Glasses models are built in code, not downloaded.** Each design is a lens outline from which the rims, lenses, bridge and arms are generated at real size in mm. That means no licences to track, no large downloads, and every dimension is explicit. The cost is simpler shapes than a professional model; swapping in `.glb` files later would only replace one function.
- **The 2D version used SVG images instead of PNGs.** It's kept behind a "3D glasses" toggle as a baseline for comparison.
- **Lenses use plain transparency, not transmission.** Transmission refracts the 3D scene behind it, but the background here is the HTML video, which isn't in the scene, so it would refract nothing.
- **Occluder:** an ellipsoid head (75 × 120 × 95 mm half-sizes), centred ~100 mm behind the eye corners so it hides the arms without covering the bridge.
  - **Problem found in testing:** when I turned my head, the far arm showed through the far lens. The ellipsoid has to stay behind the eye-corner plane, so nothing hid what's behind the *front* of the face (cheeks, temples, nose).
  - **Fix: a second occluder made from MediaPipe's live face mesh.** All 468 landmarks are already the front of the face. MediaPipe lists the mesh as edges, three per triangle in order, so the 852 triangles come straight from the list. The mesh is pushed 4 mm back into the head so landmark noise doesn't hide the near arm at the temple.
  - **Side effect, and its fix:** a mesh close enough to hide the arm also clipped the far rim near the nose. So the glasses now render in **two passes**: occluders + arms first, then the depth buffer is cleared and the rims, lenses and bridge are drawn on top. Real frames always sit in front of the face, so only the arms ever need hiding.
  - **Checked** on a still frame of a turned head from my demo video. The long far-arm segment through the lens is gone, and the rims stay intact for all three designs. A short stub of the far hinge can still show through the lens at strong turns, which real glasses also do.
- **Room-light matching (the stretch goal) was built in step 5.** About 7 times a second the video is shrunk to 64×48 pixels:
  - Brightness and colour (assuming the scene averages to grey) come from the **whole frame**. Measuring the face would make glasses darker on darker skin, which is wrong and unfair.
  - Light direction comes from the left/right halves of the **face**, where skin tone cancels out.
  - The result dims, tints and moves the virtual light.
- **The camera field-of-view point no longer applies,** because of the orthographic camera (section 3).

### 6. UI

- Live video with the glasses on, a row of thumbnails to switch frames, and a toggleable stats overlay (FPS, inference ms).
- A clear message when no face is detected or camera permission is denied.

*Why:* the brief asks for "simple and user-friendly", so no extra features beyond switching glasses.

*Changed during implementation (UI redesign after step 8):* the camera now fills the screen like a mirror, and every control floats over it as frosted glass.
- **Frame tray:** at the bottom, with pictures of each frame and a highlight that slides to the selection. Keys 1–3 switch frames.
- **Top bar:** an FPS pill whose dot turns green when a face is tracked, plus a stop button.
- **Face hint:** "Looking for your face…" appears whenever no face is found.
- **Start screen:** a card with a privacy note ("video never leaves your device").
- **Developer tools are hidden by default** in a drawer that slides in from the right (button or the D key). It holds live stats, display toggles, fit and smoothing sliders, the measurement recorder and camera capabilities. The demo looks like a product, while the engineering stays one click away for the interview.
- **Polish:** press feedback on every button, motion that stays interruptible, and support for reduced motion, reduced transparency and high contrast. Touch targets are at least 44 px, and controls stay clear of phone notches.
- **Trade-off:** the full-screen view crops the camera image to fill the window instead of letterboxing it. The glasses stay aligned because all layers are cropped identically.

### 7. Performance measurement

- **FPS:** frames rendered per second, averaged over a rolling window.
- **Inference time:** `performance.now()` around `detectForVideo`.
- **Frame-to-render latency:** timestamp from `requestVideoFrameCallback` (capture time where the browser exposes it) to the end of the render.
- **True end-to-end check:** film the screen next to a moving object or stopwatch with a phone at high frame rate, and count the frames of lag.
- Report results with the test conditions: laptop model, browser, resolution, lighting.

*Why:* the brief says targets aren't pass/fail but results and conditions must be reported, so honest, reproducible numbers matter more than hitting them perfectly.

*Implemented (step 7):*
- **Live stats** in the developer drawer: FPS, inference, render, capture→render latency, jitter, lighting estimate.
- **A Measure recorder:** you label a scenario and record 10/30/60 s. It downloads a CSV with one row per frame, plus the test conditions (browser, GPU, resolution, settings, camera capabilities) and a summary (mean/median/p95, face-loss %, jitter, head angles reached).
- The slow-motion phone check is still to do.

### 8. Testing alignment and realism

Test head turns (yaw ±45°), nods, tilts, moving close to/far from the camera, glasses-wearers, and bright, dim and backlit rooms. Record where it breaks. Those go straight into "known limitations".

### 9. Build order

1. Webcam feed + Face Landmarker drawing landmarks (prove tracking works, measure FPS).
2. 2D PNG glasses overlay from landmarks (baseline).
3. Three.js scene with one glTF model driven by the head pose.
4. One Euro smoothing.
5. Occluder, lens materials, environment lighting.
6. Three+ models and the switcher UI.
7. Stats overlay, measurements, testing.
8. Deploy, record demo video, write README and `AI_PROMPTS.md`.

*Why this order:* each step gives something working and demoable, so if time runs out I still have a submission.

*Progress:* steps 1–7 are done. Steps 3, 5 and 6 merged in practice: one 3D step built the models, the occluder, the materials and the three designs together. Step 8 is in progress: the README and `AI_PROMPTS.md` are written and the MediaPipe files are self-hosted. Still to do: deploying, and recording the video.

### 10. Submission checklist

- [x] GitHub repo with source
- [ ] Live demo link (to deploy: Vercel or GitHub Pages)
- [ ] 1–2 min demo video
- [x] README: approach, frameworks/models used, performance results + test conditions, known limitations
- [x] `AI_PROMPTS.md` with prompts used and credit for AI tools (Claude)

### Expected limitations (to verify)

*These were written before testing. What the tests showed is in brackets; full numbers are in Test results below.*

- Accuracy drops at extreme head angles (MediaPipe loses landmarks past roughly profile view). *[Confirmed: tracking drops at ~63–77° yaw, or ~42–74° in dim light.]*
- Low light means noisier landmarks and more jitter. *[Not confirmed: jitter stayed below half a pixel. Dim light's real effects are the camera dropping to 10 FPS and tracking failing at a distance.]*
- No real per-user face scale: glasses size is relative to face width, not measured in millimetres. *[Still true: it assumes an average 90 mm eye-corner distance.]*
- Lighting matching is approximate (no true light estimation). *[Still true: backlighting makes the glasses too bright, and there are no cast shadows.]*
- Performance depends on GPU/WebGL support in the browser. *[Partly: with 4× CPU throttling, inference tripled to 37 ms and latency stayed mostly under 100 ms. The actual FPS limit was the webcam's auto-exposure, which I hadn't expected.]*



## Test results (step 7)

13 recordings made with the in-app Measure panel (CSVs, one row per frame). Unless stated otherwise: 3D mode, round glasses, Match room lighting on, smoothing 3 Hz / beta 30.

**Test machine:** Chrome 155 on Windows 11, NVIDIA RTX 5090 Laptop GPU, 24 CPU threads, webcam at 1280×720, MediaPipe on GPU. This is a very powerful machine, so these are best-case numbers.

### Against the brief's targets

| Target | Result | Verdict |
|---|---|---|
| FPS ≥ 24 | 19.9 FPS in normal light, 10 FPS in dim light | Missed, but limited by the **camera**, not the code |
| Latency < 100 ms | Capture→render median 17–19 ms, p95 ~27 ms | Met (excludes display wait and camera exposure) |
| Alignment during head movement | Tracking holds to ~60–75° yaw | Good range |
| Different lighting | Works up close in dim light; fails at a distance in dim light | Partial |

### Findings

1. **The camera is the FPS bottleneck.**
   - Processed FPS equalled camera FPS in every normal run, so no frames were skipped; the camera only delivered ~20 per second.
   - Per frame, our work is ~12 ms inference + 0.2–2.6 ms render, which is capacity for ~60 FPS.
   - In dim light the webcam lengthens its exposure and drops to 10 FPS, so the glasses only update 10 times a second. That partly explains the drag felt in dim light.
   - Fix to try: explicitly request 30 FPS from the camera (we currently only request a resolution).
   - **Applied:** the app now requests `frameRate: { ideal: 30 }`. The camera's configured rate is shown next to the resolution and saved in each CSV's conditions (`cameraFrameRateSetting`), so a new recording shows whether the camera actually agreed.
   - **Re-measured: no effect.** The camera *agreed* to 30 FPS but still delivered exactly 19.9 FPS in normal light (a frame every 50.0 ms) and 10.0 FPS in dim light (every 95.9 ms). Inference, latency and jitter were unchanged.

   **Current limitation: the webcam's auto-exposure caps the frame rate.**

   - **What exposure is:** each frame is a tiny photo, and the exposure time is how long the sensor collects light for it. Longer exposure gives a brighter image but fewer frames per second, because the next frame can't start until the current one finishes (31 ms → max ~30 FPS, 50 ms → 20 FPS, 100 ms → 10 FPS).
   - **Why requesting 30 FPS didn't work:** the camera runs in auto exposure ("continuous"). In my lighting it chooses longer exposures for a brighter picture, which silently overrides the requested frame rate. The exact 50 ms / ~100 ms frame gaps match that: a processing limit would give uneven gaps, not clean steps.
   - **What the camera reports** (ASUS FHD webcam, read via `getCapabilities()` in the new Camera capabilities debug panel):
     - Supports up to 30 FPS, and is configured for 30.
     - **Supports manual exposure** (`exposureMode: continuous, manual`): shutter time 3.9 ms to 500 ms, in 3.9 ms steps.
     - Brightness control from −64 to 64.
     - Exposure compensation exists, but only applies in auto mode.
     - No gain/ISO control is exposed.
   - **A contradiction I can't fully explain:** in auto mode it *reports* a 15.6 ms exposure while delivering 20 FPS. Most likely the reported value is a stored setting rather than what auto-exposure is doing frame by frame, plus a driver "low-light priority" behaviour that web pages can't switch off. This is inferred from the numbers, not confirmed.
   - **Possible fix: manual exposure.** Fix the shutter at 31.25 ms or less so the camera has time for 30 frames a second.
     - *Cost:* a darker image, since there's no longer exposure to brighten a dim room. Raising `brightness` lifts it digitally but adds noise.
     - *Possible bonus:* shorter exposures mean less motion blur, so sharper frames during head turns, and the glasses update 30 times a second instead of 20 (which may reduce the felt drag).
     - *Open question:* whether the darker image hurts tracking (face loss, jitter) more than the extra frames help. Especially important in dim light, where tracking at a distance already fails.
     - *Risk:* some Windows webcam drivers ignore manual exposure, or only apply it after a camera restart.
   - **If manual exposure doesn't work or hurts tracking:** accept ~20 FPS as a documented hardware limit. Our pipeline needs only ~12–15 ms per frame (capacity for 30+ FPS), so the shortfall is the camera's, not the code's.

   **Result: manual exposure tested and rejected.** I added a manual-exposure checkbox, an exposure slider (capped at 31.25 ms) and a camera-brightness slider, then tested them on screen recordings.

   - **First trial:** the image looked washed out even in "auto". Cause: webcam settings are stored in the camera/driver, not the page, so a brightness of 64 from an earlier session survived page refreshes and camera restarts. My code also copied that leftover value into the slider and never sent "auto" when unticked.
     - *Fix (kept):* every camera start now resets the camera to a baseline: auto exposure, brightness 0, exposure compensation 64 (the middle of each range, which matches the original values).
     - A second bug also turned up: the "Camera is off" text stayed drawn over the live video, because a CSS rule overrode the `hidden` attribute. Fixed.
   - **Second trial (after the reset fix):**
     - **Auto exposure:** natural image, face tracked, glasses fitted well, 19.9 FPS.
     - **Manual at 31.25 ms:** almost black. Only the ceiling lamp and lit wall were visible, the face couldn't be found, and FPS was still ~19.8. (An earlier trial without a face in view did reach ~30 FPS, so the rise isn't reliable either.)
     - **Raising brightness (20 → 61):** turned the whole image flat grey; the face was still undetectable.
     - **Shorter exposures** were even darker.
   - **Why it fails:**
     - If auto exposure really used ~50 ms per frame, switching to 31 ms should only darken the image by about a third, yet it was many times darker. So in auto mode the camera must also be raising **sensor gain** (amplification).
     - In manual mode the driver drops the gain, and **the browser offers no gain control for this camera** (it isn't in `getCapabilities()`).
     - "Brightness" is not gain: it adds a fixed offset to every pixel, lifting blacks to grey without recovering detail that was never captured.
     - The frame rate in manual mode also depends on driver logic the browser can't see.
   - **Decision:** removed the manual exposure and brightness controls. Kept the startup reset (clears leftover settings from testing) and the Camera capabilities panel (documents what the camera supports).
   - **Keeping the startup reset: considered and decided.** With the manual controls gone, the app no longer changes camera settings itself, so the reset's original job (undoing our own sliders) is done. I kept it anyway, as a safeguard for edge cases where something else leaves the camera in an odd state (a leftover from earlier testing, or another camera app).
     - *Trade-off:* it forces brightness and exposure compensation to the middle of their ranges on every start, so it would quietly undo any deliberate adjustment a user made in Windows camera settings.
     - *If that becomes a problem:* delete `resetCameraSettings()`, its call in `startCamera()`, and the `midpoint()` helper. Don't just comment out the function: the call would then fail and the camera wouldn't start.
   - **Conclusion for the write-up:** the pipeline needs ~12–15 ms per frame (capacity for 30+ FPS), but on this webcam the browser can't get 30 FPS *with a usable image*. Auto exposure caps delivery at 20 FPS indoors and 10 FPS in dim light, and is documented as a hardware limit.

2. **Smoothing isn't needed on this setup.** The summary's 30 s "jitter" number (~1.8 px either way) is dominated by slow body sway, so I recomputed true frame-to-frame jitter from the per-frame rows:

   | Hold-still run | Raw | Smoothed |
   |---|---|---|
   | Normal light | 0.24 px | 0.12 px |
   | Dim light | 0.36 px | 0.21 px |

   - Smoothing does halve jitter, but raw jitter is already a quarter of a pixel, which is invisible.
   - It costs ~3–4 px (p95) of lag behind the face during movement, which *is* visible.
   - Low light did **not** noticeably increase jitter, contrary to my expected limitation. Its real effect is the FPS drop and a shorter tracking range.
   - By the decision rule in section 4 (raw jitter under ~1 px even in dim light), smoothing should be **off by default**, keeping the toggle.
   - **Applied:** smoothing is now off by default. The tuned values (3 Hz / beta 30) stay in place for when it's switched on, e.g. with a noisier camera.

3. **Tracking limits** (for "known limitations"):
   - **Head turns:** tracking drops at ~63–77° yaw in normal light, and ~42–74° in dim light (less consistent).
   - **Distance, normal light:** tracked from a face ~300 px wide down to ~88 px wide, with 0% loss.
   - **Distance, dim light:** 74–80% face loss. It only held when close (face over ~226 px wide in the smoothing-off run).
   - **Pitch at rest** reads −2° to −5°: the known forehead-slope offset, not an error.

4. **Weaker hardware (Chrome DevTools 4× CPU throttle):**
   - Inference 12 → 37 ms. Processed FPS 18.3 vs the camera's 19.9, so it started skipping frames.
   - Latency median 50 ms, p95 84 ms, worst 106 ms: still mostly under 100 ms, with little margin.
   - Inference slows even in GPU mode because MediaPipe does much of its pre/post-processing on the CPU.

5. **Unexpected cost: lighting estimation.**
   - Render time is ~0.2 ms on most frames but spikes to ~10 ms on ~28% of frames. That matches the lighting estimate's ~7 updates per second: reading pixels back from the video stalls the GPU.
   - **Confirmed** with a second 30 s hold-still run (normal light, same smoothing and resolution) with Match room lighting off:

     | | Lighting on | Lighting off |
     |---|---|---|
     | Render median | 0.2 ms | 0.1 ms |
     | Render p95 | 10 ms | 0.2 ms |
     | Render worst | 14.4 ms | 0.3 ms |
     | Frames with a >5 ms spike | 28% | 0% |
     | Capture→render latency (mean / p95) | 18.8 / 27.1 ms | 15.4 / 19.1 ms |
     | Slowest 5% of frames, total work | 22 ms | 14.4 ms |
     | FPS | 19.9 | 19.9 (camera-limited either way) |

   - The spikes arrive about every 187 ms: the estimate is set to every 150 ms, but it can only run when a frame arrives (every 50 ms at 20 FPS), so it lands on roughly every 4th frame and costs ~9 ms each time. Overall it adds ~3.4 ms to average latency and ~8 ms to the slowest 5% of frames.
   - **Correction:** I first thought this would matter at 30 FPS. It doesn't. A 30 FPS frame has a 33 ms budget and even the spiky frames need only ~22 ms in total. It would only become a problem around 60 FPS (16.7 ms budget) or on much weaker hardware.
   - **Decision:** keep the feature. If the time were ever needed back, the cheapest fix is updating less often (~2 times a second instead of ~7), since room lighting rarely changes faster than that.

### Not measured yet

- True end-to-end latency (slow-motion phone video of the screen).
- Genuinely weaker hardware, and other people's faces (also needed to re-check the pitch/occluder defaults).
