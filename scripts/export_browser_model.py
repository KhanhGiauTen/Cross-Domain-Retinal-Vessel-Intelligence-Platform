"""Export the tracked SegFormer checkpoint and verify browser-model parity."""

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("browser_segformer", ROOT / "src/models/segformer.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
SegFormerB0 = module.SegFormerB0


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    torch.set_num_threads(4)
    torch.manual_seed(17)
    checkpoint = ROOT / "src/models/best_segformer_b0.pth"
    output = ROOT / "web-demo/public/model"
    output.mkdir(parents=True, exist_ok=True)
    model = SegFormerB0(pretrained=False).eval()
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    for key in ("state_dict", "model_state_dict", "model"):
        if isinstance(state.get(key), dict):
            state = state[key]
            break
    candidates = [state, {f"model.{k}": v for k, v in state.items()},
                  {k.removeprefix("module."): v for k, v in state.items()}]
    for candidate in candidates:
        try:
            model.load_state_dict(candidate, strict=True)
            break
        except RuntimeError:
            continue
    else:
        raise RuntimeError("Checkpoint does not match the source model")

    class ProbabilityModel(torch.nn.Module):
        def __init__(self, source_model):
            super().__init__()
            self.segformer = source_model

        def forward(self, image):
            return torch.sigmoid(self.segformer(image))

    wrapped = ProbabilityModel(model).eval()
    dummy = torch.randn(1, 3, 512, 512)
    path = output / "segformer-b0.onnx"
    torch.onnx.export(wrapped, dummy, str(path), input_names=["image"],
                      output_names=["probability"], opset_version=17, dynamo=False)
    onnx.checker.check_model(onnx.load(path))
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])

    source = ROOT / "docs/images/segformer_prediction_example.png"
    figure = Image.open(source).convert("RGB")
    if figure.size != (1174, 407):
        raise ValueError("Report layout changed; recheck the published image crop")
    sample = figure.crop((10, 32, 375, 397))
    samples = ROOT / "web-demo/public/samples"
    samples.mkdir(parents=True, exist_ok=True)
    sample.save(samples / "published-example.png")
    image = np.asarray(sample.resize((512, 512), Image.Resampling.BILINEAR), dtype=np.float32) / 255
    normalized = ((image - np.array([.485, .456, .406], dtype=np.float32)) /
                  np.array([.229, .224, .225], dtype=np.float32)).transpose(2, 0, 1)[None]
    errors = []
    for tensor in (dummy.numpy(), normalized):
        with torch.no_grad():
            expected = wrapped(torch.from_numpy(tensor)).numpy()
        actual = session.run(None, {"image": tensor})[0]
        error = float(np.max(np.abs(expected - actual)))
        np.testing.assert_allclose(actual, expected, atol=2e-4, rtol=2e-4)
        errors.append(error)
    manifest = {
        "model": "SegFormer-B0", "checkpoint": checkpoint.name,
        "checkpointSha256": sha256(checkpoint), "onnxSha256": sha256(path),
        "bytes": path.stat().st_size, "inputShape": [1, 3, 512, 512],
        "outputShape": [1, 1, 512, 512], "opset": 17,
        "mean": [.485, .456, .406], "std": [.229, .224, .225],
        "maxParityError": max(errors), "sampleSource": "docs/images/segformer_prediction_example.png",
        "sampleCrop": [10, 32, 375, 397],
        "scope": "Actual SegFormer inference; no clinical validation; no DeepLab checkpoint included",
        "torchVersion": torch.__version__, "transformersVersion": __import__("transformers").__version__,
        "onnxRuntimeVersion": ort.__version__,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
