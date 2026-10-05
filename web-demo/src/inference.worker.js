import * as ort from "onnxruntime-web/wasm";

ort.env.wasm.numThreads = 1;
ort.env.wasm.proxy = false;
ort.env.wasm.wasmPaths = new URL("/runtime/", self.location.origin).href;
let session;

self.onmessage = async ({ data }) => {
  try {
    if (!session) {
      self.postMessage({
        type: "status",
        text: "Loading checkpoint and inference runtime...",
      });
      session = await ort.InferenceSession.create("/model/segformer-b0.onnx", {
        executionProviders: ["wasm"],
      });
    }
    self.postMessage({
      type: "status",
      text: "Running SegFormer-B0 on this device...",
    });
    const start = performance.now();
    const tensor = new ort.Tensor("float32", data.tensor, [1, 3, 512, 512]);
    const output = await session.run({ image: tensor });
    const probability = new Float32Array(output.probability.data);
    self.postMessage(
      {
        type: "result",
        probability,
        seconds: (performance.now() - start) / 1000,
      },
      [probability.buffer],
    );
  } catch (error) {
    session = undefined;
    self.postMessage({
      type: "error",
      text: `Inference failed: ${error.message}. Retry or use a recent desktop browser.`,
    });
  }
};
