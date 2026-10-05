import React, { useEffect, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import {
  ArrowLeft,
  ArrowUpRight,
  Download,
  FileImage,
  FlaskConical,
  LoaderCircle,
  Play,
  RotateCcw,
  ShieldCheck,
  Square,
  Upload,
} from "lucide-react";
import {
  SIZE,
  normalizePixels,
  renderPixels,
  retinaCrop,
  thresholdStats,
} from "./image-processing.mjs";
import "./styles.css";

const SAMPLE = "/samples/published-example.png";

function App() {
  const [image, setImage] = useState(null);
  const [probability, setProbability] = useState(null);
  const [threshold, setThreshold] = useState(0.15);
  const [opacity, setOpacity] = useState(0.5);
  const [view, setView] = useState("overlay");
  const [crop, setCrop] = useState(true);
  const [busy, setBusy] = useState(false);
  const [status, setStatus] = useState("Loading published example...");
  const [error, setError] = useState("");
  const [seconds, setSeconds] = useState(null);
  const [manifest, setManifest] = useState(null);
  const [exports, setExports] = useState(null);
  const inputCanvas = useRef(null),
    outputCanvas = useRef(null),
    fileInput = useRef(null);
  const worker = useRef(null),
    timeout = useRef(null),
    request = useRef(0);
  const stats = probability ? thresholdStats(probability, threshold) : null;

  function stop() {
    worker.current?.terminate();
    worker.current = null;
    clearTimeout(timeout.current);
    setBusy(false);
    setStatus("Inference stopped.");
  }

  async function loadImage(blob, name, autoCrop = crop) {
    const id = ++request.current;
    setError("");
    setStatus("Preparing image...");
    setImage(null);
    setProbability(null);
    setSeconds(null);
    setExports(null);
    let bitmap;
    try {
      if (blob.size > 10 * 1024 * 1024)
        throw new Error("Choose a PNG or JPEG smaller than 10 MB.");
      bitmap = await createImageBitmap(blob);
      if (
        bitmap.width * bitmap.height > 16000000 ||
        bitmap.width > 8192 ||
        bitmap.height > 8192
      )
        throw new Error("Image exceeds the 16-megapixel limit.");
      const scratch = document.createElement("canvas");
      scratch.width = bitmap.width;
      scratch.height = bitmap.height;
      const context = scratch.getContext("2d", { willReadFrequently: true });
      context.drawImage(bitmap, 0, 0);
      const box = autoCrop
        ? retinaCrop(
            context.getImageData(0, 0, bitmap.width, bitmap.height).data,
            bitmap.width,
            bitmap.height,
          )
        : [0, 0, bitmap.width, bitmap.height];
      const resized = document.createElement("canvas");
      resized.width = resized.height = SIZE;
      const target = resized.getContext("2d", { willReadFrequently: true });
      target.drawImage(bitmap, ...box, 0, 0, SIZE, SIZE);
      if (id !== request.current) return;
      setImage({
        pixels: target.getImageData(0, 0, SIZE, SIZE).data,
        name,
        blob,
        original: [bitmap.width, bitmap.height],
        box,
      });
      setProbability(null);
      setSeconds(null);
      setStatus("Ready to run SegFormer-B0.");
    } catch (err) {
      if (id === request.current) {
        setError(err.message);
        setStatus("Image could not be loaded.");
      }
    } finally {
      bitmap?.close();
    }
  }

  async function loadSample() {
    const id = ++request.current;
    try {
      const response = await fetch(SAMPLE);
      if (!response.ok) throw new Error("Published sample is unavailable.");
      const blob = await response.blob();
      if (id === request.current)
        await loadImage(blob, "Published report example");
    } catch (err) {
      if (id === request.current) setError(err.message);
    }
  }

  useEffect(() => {
    loadSample();
    fetch("/model/manifest.json")
      .then((r) => (r.ok ? r.json() : null))
      .then(setManifest)
      .catch(() => {});
    return () => {
      worker.current?.terminate();
      clearTimeout(timeout.current);
      request.current++;
    };
  }, []);

  useEffect(() => {
    if (!image) {
      inputCanvas.current?.getContext("2d").clearRect(0, 0, SIZE, SIZE);
      outputCanvas.current?.getContext("2d").clearRect(0, 0, SIZE, SIZE);
      return;
    }
    inputCanvas.current
      ?.getContext("2d")
      .putImageData(
        new ImageData(new Uint8ClampedArray(image.pixels), SIZE, SIZE),
        0,
        0,
      );
    const pixels = renderPixels(
      image.pixels,
      probability,
      threshold,
      view,
      opacity,
    );
    outputCanvas.current
      ?.getContext("2d")
      .putImageData(new ImageData(pixels, SIZE, SIZE), 0, 0);
    if (!probability) {
      setExports(null);
      return;
    }
    const maskCanvas = document.createElement("canvas");
    maskCanvas.width = maskCanvas.height = SIZE;
    maskCanvas
      .getContext("2d")
      .putImageData(
        new ImageData(
          renderPixels(image.pixels, probability, threshold, "mask"),
          SIZE,
          SIZE,
        ),
        0,
        0,
      );
    const summary = {
      model: "SegFormer-B0",
      modelSha256: manifest?.onnxSha256,
      inputSize: [SIZE, SIZE],
      originalSize: image.original,
      cropBox: image.box,
      threshold,
      inferenceSeconds: seconds,
      ...thresholdStats(probability, threshold),
      scope:
        "Research output, not clinical diagnosis. Browser inference; no image upload to a server.",
    };
    setExports({
      mask: maskCanvas.toDataURL("image/png"),
      view: outputCanvas.current.toDataURL("image/png"),
      summary: `data:application/json;charset=utf-8,${encodeURIComponent(JSON.stringify(summary, null, 2))}`,
    });
  }, [image, probability, threshold, view, opacity, manifest, seconds]);

  function run() {
    if (!image || busy) return;
    setBusy(true);
    setError("");
    setProbability(null);
    setSeconds(null);
    setStatus("Starting inference worker...");
    if (!worker.current) {
      worker.current = new Worker(
        new URL("./inference.worker.js", import.meta.url),
        { type: "module" },
      );
      worker.current.onmessage = ({ data }) => {
        if (data.type === "status") setStatus(data.text);
        else {
          clearTimeout(timeout.current);
          setBusy(false);
          if (data.type === "result") {
            setProbability(data.probability);
            setSeconds(data.seconds);
            setStatus("Inference complete.");
          } else {
            setError(data.text);
            setStatus("Inference unavailable.");
          }
        }
      };
      worker.current.onerror = () => {
        stop();
        setError(
          "The inference worker could not start. Reload and retry in a recent browser.",
        );
      };
    }
    const tensor = normalizePixels(image.pixels);
    worker.current.postMessage({ tensor }, [tensor.buffer]);
    timeout.current = setTimeout(() => {
      stop();
      setError(
        "Inference timed out after three minutes. Try a desktop browser.",
      );
    }, 180000);
  }

  return (
    <>
      <header className="topbar">
        <a
          className="brand"
          href="https://khanh-portfolio-ochre.vercel.app/projects/cross-domain-retinal"
        >
          <FlaskConical size={21} />
          Retinal Lab
        </a>
        <nav>
          <a href="https://khanh-portfolio-ochre.vercel.app/projects/cross-domain-retinal">
            <ArrowLeft size={16} />
            Portfolio
          </a>
          <a
            href="https://github.com/KhanhGiauTen/Cross-Domain-Retinal-Vessel-Intelligence-Platform"
            target="_blank"
            rel="noreferrer"
          >
            Source
            <ArrowUpRight size={16} />
          </a>
        </nav>
      </header>
      <main>
        <div className="page-heading">
          <div>
            <p className="eyebrow">CHASEDB1 to DRIVE / Research checkpoint</p>
            <h1>Cross-Domain Vessel Lab</h1>
          </div>
          <span className="privacy">
            <ShieldCheck size={16} />
            Images stay on your device
          </span>
        </div>
        <p className="notice">
          <strong>Academic research only.</strong> No diagnostic or clinical
          use. Predictions depend on the checkpoint, image domain and selected
          threshold.
        </p>
        <div className="workspace">
          <aside className="controls" aria-label="Inference settings">
            <h2>Input</h2>
            <button className="sample" onClick={loadSample} disabled={busy}>
              <img src={SAMPLE} alt="Published retinal fundus example" />
              <span>
                Published example<small>From the project report</small>
              </span>
              <RotateCcw size={16} />
            </button>
            <input
              ref={fileInput}
              type="file"
              accept="image/png,image/jpeg"
              hidden
              onChange={(event) => {
                const file = event.target.files?.[0];
                if (file) loadImage(file, file.name);
                event.target.value = "";
              }}
            />
            <button
              className="secondary wide"
              onClick={() => fileInput.current.click()}
              disabled={busy}
            >
              <Upload size={16} />
              Choose image
            </button>
            <label className="check">
              <input
                type="checkbox"
                checked={crop}
                disabled={busy}
                onChange={(event) => {
                  setCrop(event.target.checked);
                  if (image)
                    loadImage(image.blob, image.name, event.target.checked);
                }}
              />
              Crop fundus region
            </label>
            <div className="setting">
              <span className="field-title">Model</span>
              <strong>SegFormer-B0</strong>
              <small>Trained checkpoint / ONNX / CPU</small>
              <small>512 x 512 / ImageNet normalization</small>
            </div>
            <label className="setting">
              Threshold<output>{threshold.toFixed(2)}</output>
              <input
                aria-label="Prediction threshold"
                type="range"
                min=".05"
                max=".95"
                step=".01"
                value={threshold}
                onChange={(event) => setThreshold(Number(event.target.value))}
              />
            </label>
            <label className="setting">
              Overlay opacity<output>{Math.round(opacity * 100)}%</output>
              <input
                aria-label="Overlay opacity"
                type="range"
                min=".1"
                max=".9"
                step=".05"
                value={opacity}
                onChange={(event) => setOpacity(Number(event.target.value))}
              />
            </label>
            {busy ? (
              <button className="primary wide" onClick={stop}>
                <Square size={15} />
                Stop inference
              </button>
            ) : (
              <button className="primary wide" onClick={run} disabled={!image}>
                <Play size={16} />
                Run segmentation
              </button>
            )}
            <div className="status" role="status" aria-live="polite">
              {busy && <LoaderCircle size={15} className="spinner" />}
              {status}
            </div>
            {error && (
              <p className="error" role="alert">
                {error}
              </p>
            )}
          </aside>
          <section className="results" aria-label="Segmentation workspace">
            <div className="image-grid">
              <div className="image-pane">
                <div className="pane-heading">
                  <h2>Model input</h2>
                  <span>512 x 512</span>
                </div>
                <div className="canvas-frame">
                  <canvas
                    ref={inputCanvas}
                    width={SIZE}
                    height={SIZE}
                    aria-label="Prepared retinal image"
                  />
                  {!image && <FileImage size={40} />}
                </div>
                <p className="image-caption" title={image?.name}>
                  {image?.name ?? "No image selected"}
                </p>
              </div>
              <div className="image-pane">
                <div className="pane-heading">
                  <h2>Prediction</h2>
                  <span>{probability ? "Live model output" : "Not run"}</span>
                </div>
                <div className="canvas-frame">
                  <canvas
                    ref={outputCanvas}
                    width={SIZE}
                    height={SIZE}
                    aria-label={`${view} segmentation view`}
                  />
                  {busy && (
                    <div className="processing">
                      <LoaderCircle className="spinner" size={22} />
                      <span>Processing</span>
                    </div>
                  )}
                </div>
                <div
                  className="segmented"
                  role="tablist"
                  aria-label="Result view"
                >
                  {["overlay", "mask", "probability", "input"].map((mode) => (
                    <button
                      role="tab"
                      key={mode}
                      aria-selected={view === mode}
                      disabled={!probability && mode !== "input"}
                      onClick={() => setView(mode)}
                    >
                      {mode === "input"
                        ? "Original"
                        : mode[0].toUpperCase() + mode.slice(1)}
                    </button>
                  ))}
                </div>
              </div>
            </div>
            <div className="metrics" aria-label="Model output metrics">
              <div>
                <span>Vessel density</span>
                <strong>{stats ? `${stats.density.toFixed(2)}%` : "--"}</strong>
              </div>
              <div>
                <span>Vessel pixels</span>
                <strong>
                  {stats ? stats.vesselPixels.toLocaleString() : "--"}
                </strong>
              </div>
              <div>
                <span>Inference time</span>
                <strong>
                  {seconds !== null ? `${seconds.toFixed(2)}s` : "--"}
                </strong>
              </div>
              <div>
                <span>Execution</span>
                <strong>On-device CPU</strong>
              </div>
            </div>
            <div className="export-row">
              <span>
                {probability
                  ? `Threshold ${threshold.toFixed(2)} / ${SIZE * SIZE} analyzed pixels`
                  : "Checkpoint ready for browser inference"}
              </span>
              <div>
                <a
                  className="secondary"
                  aria-disabled={!exports}
                  href={exports?.mask}
                  download="segformer-vessel-mask.png"
                >
                  <Download size={16} />
                  Mask PNG
                </a>
                <a
                  className="secondary"
                  aria-disabled={!exports}
                  href={exports?.view}
                  download={`vessel-${view}.png`}
                >
                  <Download size={16} />
                  Current view
                </a>
                <a
                  className="secondary"
                  aria-disabled={!exports}
                  href={exports?.summary}
                  download="vessel-analysis.json"
                >
                  <Download size={16} />
                  Summary JSON
                </a>
              </div>
            </div>
          </section>
        </div>
        <footer>
          <span>SegFormer-B0 / CHASEDB1 to DRIVE research</span>
          <a href="/model/manifest.json" target="_blank" rel="noreferrer">
            Checkpoint provenance
            <ArrowUpRight size={14} />
          </a>
          <span>
            DeepLab results remain in the research report; its checkpoint is not
            included.
          </span>
        </footer>
      </main>
    </>
  );
}

createRoot(document.getElementById("root")).render(<App />);
