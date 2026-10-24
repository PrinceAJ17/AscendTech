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

### 3. Placement: from landmarks to glasses transform

- **Anchor:** nose-bridge landmark (between the eyes). This is where real glasses rest.
- **Scale:** distance between the outer eye corners or temples, so frames fit different face widths.
- **Rotation:** taken from the transformation matrix (roll/pitch/yaw).
- Each glasses model gets a small per-model offset (how far forward it sits, bridge height), because frames differ.

*Why:* this mirrors how glasses physically sit on a face, and it's simple math I can explain line by line in the interview.

### 4. Smoothing: One Euro filter on the pose

Apply a One Euro filter to the position/rotation (or to the key landmarks) before rendering.

*Why:* raw per-frame predictions jitter. A plain moving average removes jitter but adds lag. The One Euro filter smooths heavily when the head is still and lightly when it moves fast, so it fixes jitter without hurting responsiveness. It's a standard choice for AR tracking.

### 5. Rendering: Three.js with 3D glasses models

- Webcam `<video>` as the background, mirrored (selfie view). The 3D scene is mirrored the same way so they stay consistent.
- Glasses as glTF/GLB models (at least 3 designs, e.g. round, rectangular, aviator), from CC-licensed sources such as Sketchfab or modelled in Blender. Licences go in the README.
- **Occluder:** an invisible head-shaped mesh that writes depth only (`colorWrite: false`), attached to the same head pose. It hides the temple arms behind the head and ears, which is the biggest single realism win.
- **Lenses:** `MeshPhysicalMaterial` with transparency/transmission and slight reflections.
- **Lighting:** environment map (`RoomEnvironment` + PMREM) for believable reflections on the frames. Stretch goal: estimate average brightness from the video frame and scale light intensity to match, for the "different lighting conditions" target.
- Camera field of view matched to MediaPipe's assumed perspective camera so the overlay lines up.

*Why 3D over a 2D PNG overlay:* a PNG is fine for a front view but goes flat and wrong as soon as the head turns. 3D plus an occluder handles yaw/pitch naturally. I'll build the PNG version first as a quick baseline, then replace it.

### 6. UI

- Live video with the glasses on, a row of thumbnails to switch frames, and a toggleable stats overlay (FPS, inference ms).
- A clear message when no face is detected or camera permission is denied.

*Why:* the brief asks for "simple and user-friendly", so no extra features beyond switching glasses.

### 7. Performance measurement

- **FPS:** frames rendered per second, averaged over a rolling window.
- **Inference time:** `performance.now()` around `detectForVideo`.
- **Frame-to-render latency:** timestamp from `requestVideoFrameCallback` (capture time where the browser exposes it) to the end of the render.
- **True end-to-end check:** film the screen next to a moving object or stopwatch with a phone at high frame rate, and count the frames of lag.
- Report results with the test conditions: laptop model, browser, resolution, lighting.

*Why:* the brief says targets aren't pass/fail but results and conditions must be reported, so honest, reproducible numbers matter more than hitting them perfectly.

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

### 10. Submission checklist

- [ ] GitHub repo with source
- [ ] Live demo link
- [ ] 1–2 min demo video
- [ ] README: approach, frameworks/models used, performance results + test conditions, known limitations
- [ ] `AI_PROMPTS.md` with prompts used and credit for AI tools (Claude)

### Expected limitations (to verify)

- Accuracy drops at extreme head angles (MediaPipe loses landmarks past roughly profile view).
- Low light means noisier landmarks and more jitter.
- No real per-user face scale: glasses size is relative to face width, not measured in millimetres.
- Lighting matching is approximate (no true light estimation).
- Performance depends on GPU/WebGL support in the browser.
