// =============================================================================
// Step 5: estimate the room's lighting from the webcam image
//
// The brief asks for frames that "look natural in different lighting
// conditions". Fixed virtual lighting looks fine in one room and wrong in the
// next: too bright in a dim room, too white under warm lamps, lit from the
// wrong side. So a few times a second we look at the video and estimate:
//   1. brightness: how bright the scene is overall
//   2. tint:       the colour of the light (warm lamp vs cool daylight)
//   3. direction:  which side of the face is brighter (where the light comes from)
// glasses3d.js then adjusts the virtual lights to match.
//
// This is a cheap approximation, not true light estimation: a rough match in
// brightness, colour and side is what the eye notices most.
// =============================================================================

// Shrink the video to a tiny image before reading pixels. Reading pixels back
// from the GPU is slow-ish and costs time per pixel; 64×48 = 3,072 pixels is
// plenty for averages and costs well under a millisecond.
const W = 64;
const H = 48;
const small = document.createElement('canvas');
small.width = W;
small.height = H;
// willReadFrequently tells the browser to keep this canvas in CPU memory,
// which makes repeated getImageData calls much cheaper.
const sctx = small.getContext('2d', { willReadFrequently: true });

// Lighting changes slowly, so updating ~7 times a second is enough and keeps
// the per-frame cost near zero.
const UPDATE_INTERVAL_MS = 150;
let lastUpdate = -Infinity;

// Smoothed estimate, starting neutral. Smoothing stops the glasses' lighting
// from flickering with tiny frame-to-frame changes (or the webcam's own
// auto-exposure adjusting).
const current = { brightness: 1, tint: { r: 1, g: 1, b: 1 }, dirX: 0, dirY: 0 };
const SMOOTH = 0.3; // share of each new estimate mixed in (0 = never change, 1 = no smoothing)

// Perceived brightness of an RGB colour (Rec. 709 weights: the eye is most
// sensitive to green, least to blue).
function luma(r, g, b) {
  return 0.2126 * r + 0.7152 * g + 0.0722 * b;
}

// Average luma over a rectangle of the small image (coordinates in 0-1).
function regionLuma(data, x0, y0, x1, y1) {
  const xa = Math.max(0, Math.floor(x0 * W));
  const xb = Math.min(W, Math.ceil(x1 * W));
  const ya = Math.max(0, Math.floor(y0 * H));
  const yb = Math.min(H, Math.ceil(y1 * H));
  let sum = 0;
  let n = 0;
  for (let y = ya; y < yb; y++) {
    for (let x = xa; x < xb; x++) {
      const i = (y * W + x) * 4; // 4 bytes per pixel: R, G, B, A
      sum += luma(data[i], data[i + 1], data[i + 2]);
      n++;
    }
  }
  return n ? sum / n : 0;
}

// Returns the latest smoothed estimate, or null if it isn't time to update yet.
//   video: the <video> element. face: raw landmarks (needed for direction).
//   now: frame timestamp in ms.
export function estimateLighting(video, face, now) {
  if (now - lastUpdate < UPDATE_INTERVAL_MS) return null;
  lastUpdate = now;

  sctx.drawImage(video, 0, 0, W, H);
  const data = sctx.getImageData(0, 0, W, H).data;

  // --- 1 & 2. Brightness and tint from the WHOLE frame -----------------------
  // Why the whole frame and not just the face? Skin tone. Darker skin reflects
  // less light, so measuring the face would make glasses darker on darker-
  // skinned users in the same room. That's wrong and unfair. The whole frame
  // (walls, background) says more about the room's light.
  // Tint uses the "grey world" assumption: on average a scene is neutral grey,
  // so if the average comes out orange, the light itself is orange.
  let r = 0, g = 0, b = 0;
  for (let i = 0; i < data.length; i += 4) {
    r += data[i];
    g += data[i + 1];
    b += data[i + 2];
  }
  const n = data.length / 4;
  r /= n; g /= n; b /= n;
  const mean = (r + g + b) / 3 || 1;

  // Webcams auto-adjust exposure so a normal room averages roughly mid-grey
  // (~115 of 255). Brightness 1 = "normal". It only drops well below 1 when
  // the camera can't compensate any more (a genuinely dim room), which is
  // exactly when the glasses should look dimmer too.
  const brightness = clamp(luma(r, g, b) / 115, 0.25, 1.5);
  // Clamp the tint so a strongly coloured background (a red wall) doesn't
  // turn the glasses red. Real lighting tints are subtle.
  const tint = { r: clamp(r / mean, 0.8, 1.2), g: clamp(g / mean, 0.8, 1.2), b: clamp(b / mean, 0.8, 1.2) };

  // --- 3. Direction from the FACE ---------------------------------------------
  // Compare the brightness of the face's left vs right half, and top vs
  // bottom. Because both halves are the same skin, skin tone cancels out in
  // the ratio: only the lighting difference remains.
  // Face box from four landmarks: face edges (234, 454), forehead (10), chin (152).
  let dirX = 0;
  let dirY = 0;
  if (face) {
    const left = Math.min(face[234].x, face[454].x);
    const right = Math.max(face[234].x, face[454].x);
    const top = face[10].y;
    const bottom = face[152].y;
    const midX = (left + right) / 2;
    const midY = (top + bottom) / 2;
    const L = regionLuma(data, left, top, midX, bottom);
    const R = regionLuma(data, midX, top, right, bottom);
    const T = regionLuma(data, left, top, right, midY);
    const B = regionLuma(data, left, midY, right, bottom);
    // (R - L) / (R + L): -1 = all light from the image's left, +1 = all from the right.
    // These are raw image coordinates, the same ones the 3D scene uses.
    dirX = (R + L) > 0 ? (R - L) / (R + L) : 0;
    // Top vs bottom is naturally uneven (brow and eye sockets are shadowed),
    // so this is weaker; it's still useful to tell overhead from low light.
    dirY = (T + B) > 0 ? (T - B) / (T + B) : 0;
  }

  // Blend the new estimate into the running one (exponential smoothing).
  current.brightness += (brightness - current.brightness) * SMOOTH;
  current.tint.r += (tint.r - current.tint.r) * SMOOTH;
  current.tint.g += (tint.g - current.tint.g) * SMOOTH;
  current.tint.b += (tint.b - current.tint.b) * SMOOTH;
  current.dirX += (dirX - current.dirX) * SMOOTH;
  current.dirY += (dirY - current.dirY) * SMOOTH;
  return current;
}

function clamp(v, lo, hi) {
  return Math.min(hi, Math.max(lo, v));
}
