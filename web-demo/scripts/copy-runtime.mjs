import { copyFile, mkdir } from "node:fs/promises";
const target = new URL("../public/runtime/", import.meta.url);
await mkdir(target, { recursive: true });
for (const name of [
  "ort-wasm-simd-threaded.mjs",
  "ort-wasm-simd-threaded.wasm",
]) {
  await copyFile(
    new URL(`../node_modules/onnxruntime-web/dist/${name}`, import.meta.url),
    new URL(name, target),
  );
}
