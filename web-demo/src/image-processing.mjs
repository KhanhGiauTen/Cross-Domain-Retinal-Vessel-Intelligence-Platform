export const SIZE = 512;
const MEAN = [0.485, 0.456, 0.406];
const STD = [0.229, 0.224, 0.225];

export function normalizePixels(rgba) {
  if (rgba.length !== SIZE * SIZE * 4)
    throw new Error("Expected a 512px RGBA image");
  const pixels = SIZE * SIZE;
  const result = new Float32Array(pixels * 3);
  for (let i = 0; i < pixels; i++) {
    for (let c = 0; c < 3; c++)
      result[c * pixels + i] = (rgba[i * 4 + c] / 255 - MEAN[c]) / STD[c];
  }
  return result;
}

export function thresholdStats(probability, threshold) {
  if (!Number.isFinite(threshold) || threshold < 0 || threshold > 1)
    throw new Error("Invalid threshold");
  let count = 0;
  for (const p of probability) {
    if (!Number.isFinite(p) || p < 0 || p > 1)
      throw new Error("Invalid model output");
    if (p >= threshold) count++;
  }
  return {
    vesselPixels: count,
    totalPixels: probability.length,
    density: probability.length ? (count / probability.length) * 100 : 0,
  };
}

export function renderPixels(
  source,
  probability,
  threshold,
  mode,
  alpha = 0.5,
) {
  const result = new Uint8ClampedArray(source);
  if (!probability || mode === "input") return result;
  if (source.length !== probability.length * 4)
    throw new Error("Image and model output sizes differ");
  for (let i = 0; i < probability.length; i++) {
    const index = i * 4;
    if (mode === "probability" || mode === "mask") {
      const value =
        mode === "mask"
          ? probability[i] >= threshold
            ? 255
            : 0
          : Math.round(probability[i] * 255);
      result[index] = result[index + 1] = result[index + 2] = value;
    } else if (probability[i] >= threshold) {
      for (let c = 0; c < 3; c++)
        result[index + c] =
          source[index + c] * (1 - alpha) + [34, 221, 173][c] * alpha;
    }
    result[index + 3] = 255;
  }
  return result;
}

// The same fundus-color criteria and square padding used by the Streamlit demo.
export function retinaCrop(rgba, width, height) {
  let left = width,
    right = 0,
    top = height,
    bottom = 0,
    count = 0;
  for (let y = 0; y < height; y++)
    for (let x = 0; x < width; x++) {
      const i = (y * width + x) * 4;
      const [r, g, b] = [rgba[i], rgba[i + 1], rgba[i + 2]];
      const max = Math.max(r, g, b),
        min = Math.min(r, g, b);
      if (
        max > 35 &&
        max < 245 &&
        max - min > 18 &&
        r >= g - 20 &&
        r > b + 10
      ) {
        left = Math.min(left, x);
        right = Math.max(right, x + 1);
        top = Math.min(top, y);
        bottom = Math.max(bottom, y + 1);
        count++;
      }
    }
  if (count / (width * height) < 0.015) return [0, 0, width, height];
  const side = Math.min(
    Math.round(Math.max(right - left, bottom - top) * 1.08),
    width,
    height,
  );
  const x = Math.max(
    0,
    Math.min(width - side, Math.round((left + right - side) / 2)),
  );
  const y = Math.max(
    0,
    Math.min(height - side, Math.round((top + bottom - side) / 2)),
  );
  return [x, y, side, side];
}
