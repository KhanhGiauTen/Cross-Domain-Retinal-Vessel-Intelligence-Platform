# Public Browser Inference

Live demo: https://cross-domain-vessel-lab.vercel.app

Vercel project `cross-domain-vessel-lab`, owner `khanhgiauten`, Git root `web-demo`, Node 24.x. The canonical GitHub repository was renamed to `Cross-Domain-Retinal-Vessel-Intelligence-Platform`; local origin now follows that name.

## Actual Model, Not a Snapshot

The tracked `best_segformer_b0.pth` is exported through the original `SegFormerB0(pretrained=False)` implementation into an ONNX probability-output model. Strict state-dict loading, ONNX structural validation, and PyTorch/ONNX Runtime parity checks run in `scripts/export_browser_model.py`. The maximum observed absolute error on a random tensor and the published example is 0.000004262. The manifest records checkpoint/model SHA256, input/output shapes, preprocessing, versions, and sample provenance.

```powershell
python -m pip install onnx onnxruntime
python scripts/export_browser_model.py
cd web-demo
npm ci
npm test
npm run build
npm run dev -- --port 3002
```

Export additionally requires the project's Torch, Transformers, Pillow and NumPy dependencies. Fixed input `[1,3,512,512]`, ImageNet RGB normalization, bilinear resize, sigmoid probabilities. The browser's canvas resampler can differ slightly from Pillow; the parity measurement compares identical normalized tensors, not all image-decoder pipelines. Auto-crop follows the Streamlit fundus-color criterion and square padding; the black-background fallback preserves image extent. No Streamlit morphology or small-component cleanup is claimed in this port.

## Access And Privacy

- A same-origin ONNX Runtime Web WASM worker runs inference on the recruiter's device. First use downloads the checkpoint and runtime; speed depends on device/browser.
- PNG/JPEG inputs are decoded locally, limited to 10 MB and 16 megapixels. Images and filenames are not sent to an API or persisted.
- The bundled example is the image panel from an already published report figure, not a newly published patient record or raw dataset archive.
- Threshold and overlay-opacity controls update the result without rerunning the model. Mask, current view and summary use native download links with local data URLs.
- Stop terminates the worker; retry creates a fresh worker. Timeout and load/decode/runtime errors are visible.
- Only SegFormer-B0 is runnable. DeepLab's checkpoint is absent, so research comparison figures remain in the report.
- Research output only, not clinical diagnosis. Reported test metrics remain historical, not new validation of browser inputs.

## Hosting

Pushes trigger the Vercel Git integration. Runtime binaries are copied from the exact pinned npm package at build time; the generated ONNX model, sample and manifest are committed. No Python backend, GPU server, credentials or paid subscription is needed.

Verification covers model parity, four image-processing tests, production build, dependency audit, actual browser inference, threshold changes, input recovery, generated export payloads, and desktop/mobile rendering. The embedded browser does not report download events for local blob/data URLs; PNG/JSON contents are validated separately without uploading image results to a server.
