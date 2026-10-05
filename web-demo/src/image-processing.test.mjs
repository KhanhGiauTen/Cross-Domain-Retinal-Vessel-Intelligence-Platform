import test from "node:test";
import assert from "node:assert/strict";
import {
  normalizePixels,
  thresholdStats,
  renderPixels,
  retinaCrop,
  SIZE,
} from "./image-processing.mjs";

test("RGB normalization uses planar ImageNet channels", () => {
  const rgba = new Uint8ClampedArray(SIZE * SIZE * 4).fill(255);
  const tensor = normalizePixels(rgba);
  assert.equal(tensor.length, 3 * SIZE * SIZE);
  assert.ok(Math.abs(tensor[0] - (1 - 0.485) / 0.229) < 1e-5);
  assert.ok(Math.abs(tensor[SIZE * SIZE] - (1 - 0.456) / 0.224) < 1e-5);
  assert.throws(() => normalizePixels(new Uint8Array(4)));
});
test("threshold is inclusive and monotonic; invalid values fail", () => {
  const p = Float32Array.from([0, 0.5, 1]);
  assert.equal(thresholdStats(p, 0.5).vesselPixels, 2);
  assert.equal(thresholdStats(p, 0.9).vesselPixels, 1);
  assert.equal(thresholdStats([], 0.5).density, 0);
  assert.throws(() => thresholdStats(p, NaN));
  assert.throws(() => thresholdStats([Infinity], 0.5));
});
test("mask, probability, overlay and source views are independent", () => {
  const source = Uint8ClampedArray.from([200, 100, 50, 255, 20, 30, 40, 255]);
  const p = Float32Array.from([0.9, 0.1]);
  assert.deepEqual(
    [...renderPixels(source, p, 0.5, "mask")],
    [255, 255, 255, 255, 0, 0, 0, 255],
  );
  assert.deepEqual([...renderPixels(source, p, 0.5, "input")], [...source]);
  assert.equal(renderPixels(source, p, 0.5, "probability")[0], 229);
  assert.notEqual(renderPixels(source, p, 0.5, "overlay")[0], source[0]);
  assert.equal(source[0], 200);
  assert.throws(() => renderPixels(source, [0.5], 0.5, "mask"));
});
test("black images preserve extent; fundus crop stays inside original", () => {
  assert.deepEqual(
    retinaCrop(new Uint8Array(40 * 30 * 4), 40, 30),
    [0, 0, 40, 30],
  );
  const rgba = new Uint8Array(40 * 30 * 4);
  for (let y = 5; y < 25; y++)
    for (let x = 10; x < 30; x++)
      rgba.set([180, 90, 40, 255], (y * 40 + x) * 4);
  const [x, y, w, h] = retinaCrop(rgba, 40, 30);
  assert.equal(w, h);
  assert.ok(x >= 0 && y >= 0 && x + w <= 40 && y + h <= 30);
});
