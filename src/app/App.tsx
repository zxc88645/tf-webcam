import { useEffect, useMemo, useRef, useState } from "react";
import type { ModelKey } from "../features/detection/modelConfigs";
import { MODEL_CONFIGS } from "../features/detection/modelConfigs";
import { useTfYoloModel } from "../features/detection/useTfYoloModel";
import { useCamera } from "../features/camera/useCamera";
import { useCanvasOverlay } from "../features/detection/useCanvasOverlay";
import { runYolo } from "../lib/yolo/yoloCore";
import type { YoloPrediction } from "../lib/yolo/types";
import { clearOverlay, renderOverlay } from "../lib/render/overlay";

type Tone = "ok" | "warn";

function baseUrl(path: string) {
  return `${import.meta.env.BASE_URL}${path}`.replace(/\/{2,}/g, "/");
}

export function App() {
  const [tone, setTone] = useState<Tone>("ok");
  const [statusText, setStatusText] = useState("初始化中");
  const [version, setVersion] = useState("v?");
  const [activeModelKey, setActiveModelKey] = useState<ModelKey>("pose");
  const [predictions, setPredictions] = useState<YoloPrediction[]>([]);
  const [viewMode, setViewMode] = useState<"image" | "video">("image");
  const [imageUrl, setImageUrl] = useState<string | null>(null);

  const imageRef = useRef<HTMLImageElement | null>(null);
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  const activeModelConfig = useMemo(
    () => MODEL_CONFIGS[activeModelKey],
    [activeModelKey],
  );
  const model = useTfYoloModel(activeModelConfig);
  const camera = useCamera(videoRef);
  const overlayForImage = useCanvasOverlay(canvasRef, imageRef);
  const overlayForVideo = useCanvasOverlay(canvasRef, videoRef);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const res = await fetch(baseUrl("version.json"), { cache: "no-cache" });
        if (!res.ok) return;
        const json = (await res.json()) as {
          commit?: string;
          builtAt?: string;
        };
        if (cancelled) return;
        if (json?.commit) setVersion(`v${json.commit}`);
      } catch {
        // ignore
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    if (model.state.status === "loading") {
      setTone("ok");
      setStatusText(model.state.message);
    } else if (model.state.status === "ready") {
      setTone("ok");
      setStatusText(`模型就緒：${activeModelConfig.label}`);
    } else if (model.state.status === "error") {
      setTone("warn");
      setStatusText(model.state.message);
    } else {
      setTone("ok");
      setStatusText("初始化中");
    }
  }, [activeModelConfig.label, model.state]);

  const isCameraRunning = camera.state.status === "running";
  const canDetectImage = model.state.status === "ready" && !isCameraRunning;
  const canStartCamera = model.state.status === "ready" && !isCameraRunning;
  const canLiveDetect = model.state.status === "ready" && isCameraRunning;
  const canCaptureDetect = model.state.status === "ready" && isCameraRunning;

  function clearOverlaySafe() {
    const canvas = canvasRef.current;
    if (!canvas) return;
    clearOverlay(canvas);
  }

  async function onSelectFile(file: File) {
    const image = imageRef.current;
    if (!image) return;

    // stop camera if running
    if (camera.state.status === "running") camera.api.stop();
    stopLiveDetect();

    if (imageUrl) URL.revokeObjectURL(imageUrl);
    const url = URL.createObjectURL(file);
    setImageUrl(url);
    setViewMode("image");

    await new Promise<void>((resolve) => {
      if (image.complete && image.naturalWidth) return resolve();
      const onLoad = () => resolve();
      image.addEventListener("load", onLoad, { once: true });
      image.addEventListener("error", onLoad, { once: true });
    });

    setPredictions([]);
    clearOverlaySafe();
  }

  async function detectOnImage() {
    if (!model.api) return;
    const image = imageRef.current;
    if (!image?.src) return;
    const { scaleX, scaleY } = overlayForImage.getDrawScale();
    const preds = runYolo(
      model.api.model,
      image,
      {
        id: activeModelConfig.id,
        label: activeModelConfig.label,
        url: activeModelConfig.url,
        inputSize: activeModelConfig.inputSize,
      },
      model.api.classNames,
    );
    setPredictions(preds);
    if (canvasRef.current) {
      renderOverlay(canvasRef.current, preds, {
        dpr: window.devicePixelRatio || 1,
        scaleX,
        scaleY,
      });
    }
  }

  async function startCamera() {
    if (imageUrl) {
      URL.revokeObjectURL(imageUrl);
      setImageUrl(null);
    }
    setViewMode("video");
    clearOverlaySafe();
    setPredictions([]);
    await camera.api.start();
  }

  const liveDetectRef = useRef({
    raf: 0 as number | 0,
    running: false,
    processing: false,
  });

  function stopLiveDetect() {
    const st = liveDetectRef.current;
    st.running = false;
    if (st.raf) cancelAnimationFrame(st.raf);
    st.raf = 0;
  }

  function startLiveDetect() {
    if (!model.api) return;
    const video = videoRef.current;
    if (!video) return;

    const st = liveDetectRef.current;
    if (st.running) return;
    st.running = true;

    const tick = () => {
      st.raf = requestAnimationFrame(tick);
      if (!st.running || st.processing) return;
      if (!video.videoWidth || !video.videoHeight) return;
      st.processing = true;
      try {
        const { scaleX, scaleY } = overlayForVideo.getDrawScale();
        const preds = runYolo(
          model.api!.model,
          video,
          {
            id: activeModelConfig.id,
            label: activeModelConfig.label,
            url: activeModelConfig.url,
            inputSize: activeModelConfig.inputSize,
          },
          model.api!.classNames,
        );
        setPredictions(preds);
        if (canvasRef.current) {
          renderOverlay(canvasRef.current, preds, {
            dpr: window.devicePixelRatio || 1,
            scaleX,
            scaleY,
          });
        }
      } catch {
        // ignore per-frame
      } finally {
        st.processing = false;
      }
    };

    tick();
  }

  async function captureAndDetect() {
    // For now, reuse live frame detection (same as above) once.
    if (!model.api) return;
    const video = videoRef.current;
    if (!video) return;
    const { scaleX, scaleY } = overlayForVideo.getDrawScale();
    const preds = runYolo(
      model.api.model,
      video,
      {
        id: activeModelConfig.id,
        label: activeModelConfig.label,
        url: activeModelConfig.url,
        inputSize: activeModelConfig.inputSize,
      },
      model.api.classNames,
    );
    setPredictions(preds);
    if (canvasRef.current) {
      renderOverlay(canvasRef.current, preds, {
        dpr: window.devicePixelRatio || 1,
        scaleX,
        scaleY,
      });
    }
  }

  useEffect(() => {
    // stop live loop when camera stops / model changes
    if (camera.state.status !== "running") stopLiveDetect();
    return () => {
      stopLiveDetect();
    };
  }, [camera.state.status, activeModelKey]);

  useEffect(() => {
    return () => {
      if (imageUrl) URL.revokeObjectURL(imageUrl);
    };
  }, [imageUrl]);

  return (
    <main className="mx-auto max-w-[1100px] p-6">
      <h1 className="text-2xl font-semibold tracking-tight">
        TF.js 物件偵測（圖片 / 相機）
      </h1>
      <p className="mt-2 text-sm text-slate-600">
        上傳圖片或開啟相機，使用 YOLO 模型進行物件偵測。
      </p>

      <div
        className="mt-4 flex flex-wrap items-center gap-3 rounded-xl border border-slate-200 bg-slate-50 p-3 shadow-sm"
        aria-label="操作區"
      >
        <label
          htmlFor="modelSelect"
          className="flex items-center gap-2 text-sm text-slate-600"
        >
          模型
          <select
            id="modelSelect"
            aria-label="選擇模型"
            value={activeModelKey}
            onChange={(e) => setActiveModelKey(e.target.value as ModelKey)}
            className="h-9 rounded-lg border border-slate-200 bg-white px-3 text-sm text-slate-900 shadow-sm outline-none focus:border-cyan-400 focus:ring-2 focus:ring-cyan-200"
          >
            <option value="pose">YOLOv26n Pose</option>
            <option value="detect">YOLOv26n Detect</option>
            <option value="seg">YOLOv26n Seg</option>
          </select>
        </label>

        <input
          type="file"
          id="fileInput"
          accept="image/*"
          disabled={model.state.status !== "ready"}
          onChange={(e) => {
            const f = e.target.files?.[0];
            if (f) void onSelectFile(f);
          }}
          className="max-w-full text-sm text-slate-700 file:mr-3 file:rounded-lg file:border file:border-slate-200 file:bg-white file:px-3 file:py-2 file:text-sm file:font-medium file:text-slate-900 file:shadow-sm disabled:opacity-60"
        />

        <button
          id="detectButton"
          disabled={!canDetectImage || !imageUrl}
          onClick={() => void detectOnImage()}
          className="h-9 rounded-lg border border-slate-200 bg-white px-3 text-sm font-medium text-slate-900 shadow-sm hover:border-slate-300 disabled:cursor-not-allowed disabled:opacity-60"
        >
          執行物件偵測
        </button>
        <button
          id="cameraButton"
          disabled={!canStartCamera}
          onClick={() => void startCamera()}
          className="h-9 rounded-lg border border-slate-200 bg-white px-3 text-sm font-medium text-slate-900 shadow-sm hover:border-slate-300 disabled:cursor-not-allowed disabled:opacity-60"
        >
          開啟相機
        </button>
        <button
          id="liveDetectButton"
          disabled={!canLiveDetect}
          onClick={() =>
            liveDetectRef.current.running ? stopLiveDetect() : startLiveDetect()
          }
          className="h-9 rounded-lg border border-slate-200 bg-white px-3 text-sm font-medium text-slate-900 shadow-sm hover:border-slate-300 disabled:cursor-not-allowed disabled:opacity-60"
        >
          {liveDetectRef.current.running ? "停止即時偵測" : "開始即時偵測"}
        </button>
        <button
          id="captureButton"
          disabled={!canCaptureDetect}
          onClick={() => void captureAndDetect()}
          className="h-9 rounded-lg border border-slate-200 bg-white px-3 text-sm font-medium text-slate-900 shadow-sm hover:border-slate-300 disabled:cursor-not-allowed disabled:opacity-60"
        >
          拍照偵測
        </button>
      </div>

      <section className="mt-4 grid grid-cols-1 items-start gap-4 lg:grid-cols-[minmax(280px,1fr)_340px]">
        <div
          className="relative block max-w-full overflow-auto rounded-xl border border-slate-200 bg-white shadow-sm"
          id="stage"
          aria-label="畫面區"
        >
          <img
            id="image"
            ref={imageRef}
            alt="預覽圖片"
            src={imageUrl ?? undefined}
            className={[
              "relative z-10 block max-w-full",
              viewMode === "image" ? "block" : "hidden",
            ].join(" ")}
          />
          <video
            id="video"
            ref={videoRef}
            autoPlay
            playsInline
            className={[
              "relative z-10 max-w-full",
              viewMode === "video" ? "block" : "hidden",
            ].join(" ")}
          />
          <canvas
            id="overlay"
            ref={canvasRef}
            className="pointer-events-none absolute left-0 top-0 z-20 block"
          />
        </div>

        <aside
          className="rounded-xl border border-slate-200 bg-slate-50 p-3 shadow-sm"
          aria-label="狀態與結果"
        >
          <div className="mb-2 flex items-baseline justify-between gap-2">
            <h2 className="text-sm font-semibold text-slate-900">狀態</h2>
            <div className="flex items-baseline gap-2">
              <span
                id="appVersion"
                className="inline-flex items-center gap-2 rounded-full border border-slate-200 bg-white px-2.5 py-1 text-xs text-slate-600 shadow-sm"
                data-tone="ok"
                title="部署版本"
              >
                {version}
              </span>
              <span
                id="statusBadge"
                className={[
                  "inline-flex items-center gap-2 rounded-full border px-2.5 py-1 text-xs shadow-sm",
                  tone === "warn"
                    ? "border-red-200 bg-red-50 text-red-800"
                    : "border-cyan-200 bg-cyan-50 text-cyan-800",
                ].join(" ")}
                data-tone={tone}
              >
                {tone === "warn" ? "注意" : "就緒"}
              </span>
            </div>
          </div>
          <div
            id="status"
            role="status"
            className="whitespace-pre-wrap font-sans text-xs text-slate-900"
          >
            {statusText}
          </div>

          <h2 className="mt-3 text-sm font-semibold text-slate-900">結果</h2>
          <ol
            id="results"
            className="mt-2 list-decimal space-y-1 pl-5 text-sm text-slate-800"
          >
            {predictions.map((p, idx) => (
              <li key={`${p.classId}-${idx}`}>
                {p.class} — {(p.score * 100).toFixed(1)}%（x=
                {p.bbox[0].toFixed(0)}, y={p.bbox[1].toFixed(0)}, w=
                {p.bbox[2].toFixed(0)}, h={p.bbox[3].toFixed(0)}）
              </li>
            ))}
          </ol>
        </aside>
      </section>
    </main>
  );
}
