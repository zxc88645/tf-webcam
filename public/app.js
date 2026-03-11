import {
  COCO17_EDGES,
  KEYPOINT_THRESHOLD,
  normalizeModelOutputs,
  pickYoloTensors,
  loadClassNames,
  runYolo,
} from "./yoloCore.js";

(() => {
  "use strict";

  const MODEL_CONFIGS = {
    pose: {
      id: "pose",
      label: "YOLOv26n Pose",
      url: "./models/yolo26n-pose/model.json",
      inputSize: 640,
    },
    detect: {
      id: "detect",
      label: "YOLOv26n Detect",
      url: "./models/yolo26n-detect/model.json",
      inputSize: 640,
    },
    seg: {
      id: "seg",
      label: "YOLOv26n Seg",
      url: "./models/yolo26n-seg/model.json",
      inputSize: 640,
    },
  };

  const fileInput = document.getElementById("fileInput");
  const detectButton = document.getElementById("detectButton");
  const cameraButton = document.getElementById("cameraButton");
  const liveDetectButton = document.getElementById("liveDetectButton");
  const captureButton = document.getElementById("captureButton");
  const modelSelect = document.getElementById("modelSelect");
  const image = document.getElementById("image");
  const video = document.getElementById("video");
  const overlay = document.getElementById("overlay");
  const stage = document.getElementById("stage");
  const status = document.getElementById("status");
  const statusBadge = document.getElementById("statusBadge");
  const appVersion = document.getElementById("appVersion");
  const results = document.getElementById("results");
  const ctx = overlay.getContext("2d");

  let model = null;
  let currentImageUrl = null;
  let stream = null;
  let isLiveDetecting = false;
  let isProcessingFrame = false;
  let canvasCssWidth = 0;
  let canvasCssHeight = 0;
  let canvasDpr = 1;
  let classNames = null;
  let activeModelKey = "pose";
  let activeModelConfig = MODEL_CONFIGS[activeModelKey];

  function showImagePreview() {
    image.style.display = image.src ? "block" : "none";
  }

  function hideImagePreview() {
    image.style.display = "none";
  }

  function hideVideoPreview() {
    video.style.display = "none";
  }

  function setBadge(tone, text) {
    if (!statusBadge) return;
    statusBadge.dataset.tone = tone;
    statusBadge.textContent = text;
  }

  function setStatus(message, tone = "ok") {
    status.textContent = message;
    setBadge(tone, tone === "warn" ? "注意" : "就緒");
  }

  function updateButtons() {
    detectButton.disabled = !model || !image.src || Boolean(stream);
    liveDetectButton.disabled = !model || !stream;
    captureButton.disabled = !model || !stream;
  }

  function clearCanvas() {
    // Clear in backing-store pixels
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, overlay.width, overlay.height);
  }

  function syncCanvasSizeFor(element) {
    const intrinsicWidth =
      element.videoWidth ||
      element.naturalWidth ||
      element.width ||
      element.clientWidth;
    const intrinsicHeight =
      element.videoHeight ||
      element.naturalHeight ||
      element.height ||
      element.clientHeight;

    if (!intrinsicWidth || !intrinsicHeight) {
      return;
    }

    const rect = element.getBoundingClientRect?.();
    const displayWidth = rect?.width ? Math.max(1, rect.width) : intrinsicWidth;
    const displayHeight = rect?.height ? Math.max(1, rect.height) : intrinsicHeight;

    canvasCssWidth = displayWidth;
    canvasCssHeight = displayHeight;
    canvasDpr = window.devicePixelRatio || 1;

    overlay.width = Math.round(displayWidth * canvasDpr);
    overlay.height = Math.round(displayHeight * canvasDpr);
    overlay.style.width = `${displayWidth}px`;
    overlay.style.height = `${displayHeight}px`;
  }

  function getDrawScaleFor(element) {
    const intrinsicWidth =
      element.videoWidth ||
      element.naturalWidth ||
      element.width ||
      element.clientWidth;
    const intrinsicHeight =
      element.videoHeight ||
      element.naturalHeight ||
      element.height ||
      element.clientHeight;

    if (!intrinsicWidth || !intrinsicHeight || !canvasCssWidth || !canvasCssHeight) {
      return { scaleX: 1, scaleY: 1 };
    }

    return {
      scaleX: canvasCssWidth / intrinsicWidth,
      scaleY: canvasCssHeight / intrinsicHeight,
    };
  }

  function colorForClassId(classId) {
    // deterministic-ish color per class (use hue wheel for clear separation)
    const n = Number.parseInt(classId, 10);
    const hue = Number.isFinite(n) ? (n * 47) % 360 : 200;
    return `hsla(${hue}, 95%, 55%, 0.35)`;
  }

  function hslaToRgba(h, s, l, a) {
    // h: 0-360, s/l: 0-100, a: 0-1
    const hh = (((h % 360) + 360) % 360) / 360;
    const ss = Math.max(0, Math.min(1, s / 100));
    const ll = Math.max(0, Math.min(1, l / 100));

    const q = ll < 0.5 ? ll * (1 + ss) : ll + ss - ll * ss;
    const p = 2 * ll - q;

    const hue2rgb = (t) => {
      let tt = t;
      if (tt < 0) tt += 1;
      if (tt > 1) tt -= 1;
      if (tt < 1 / 6) return p + (q - p) * 6 * tt;
      if (tt < 1 / 2) return q;
      if (tt < 2 / 3) return p + (q - p) * (2 / 3 - tt) * 6;
      return p;
    };

    const r = Math.round(hue2rgb(hh + 1 / 3) * 255);
    const g = Math.round(hue2rgb(hh) * 255);
    const b = Math.round(hue2rgb(hh - 1 / 3) * 255);
    const aa = Math.round(Math.max(0, Math.min(1, a)) * 255);
    return { r, g, b, a: aa };
  }

  function drawPredictions(predictions, scaleX = 1, scaleY = 1) {
    clearCanvas();

    // Draw in CSS pixels; backing store is scaled by DPR.
    ctx.setTransform(canvasDpr, 0, 0, canvasDpr, 0, 0);
    ctx.lineWidth = 2;
    ctx.font = "16px Arial";

    for (const prediction of predictions) {
      const [x, y, width, height] = prediction.bbox;
      const dx = x * scaleX;
      const dy = y * scaleY;
      const dWidth = width * scaleX;
      const dHeight = height * scaleY;
      const score = (prediction.score * 100).toFixed(1);
      const label = `${prediction.class} ${score}%`;

      // seg mask (draw first, under bbox)
      if (prediction.maskCrop?.data && prediction.maskCrop.inputW && prediction.maskCrop.inputH) {
        const mask = prediction.maskCrop;
        const outW = Math.max(1, Math.round(dWidth));
        const outH = Math.max(1, Math.round(dHeight));
        const imgData = ctx.createImageData(outW, outH);
        const rgba = imgData.data;

        // nearest-neighbor sample from input mask crop to bbox display size
        const srcW = mask.inputW;
        const srcH = mask.inputH;
        const src = mask.data;
        const n = Number.parseInt(prediction.classId, 10);
        const hue = Number.isFinite(n) ? (n * 47) % 360 : 200;
        const { r, g, b, a } = hslaToRgba(hue, 95, 55, 0.38);

        for (let oy = 0; oy < outH; oy++) {
          const sy = Math.min(srcH - 1, Math.floor((oy / outH) * srcH));
          for (let ox = 0; ox < outW; ox++) {
            const sx = Math.min(srcW - 1, Math.floor((ox / outW) * srcW));
            const on = src[sy * srcW + sx] ? 1 : 0;
            if (!on) continue;
            const p = (oy * outW + ox) * 4;
            rgba[p + 0] = r;
            rgba[p + 1] = g;
            rgba[p + 2] = b;
            rgba[p + 3] = a;
          }
        }

        const px = Math.round(dx);
        const py = Math.round(dy);
        ctx.putImageData(imgData, px, py);
      }

      ctx.strokeStyle = "#00FFFF";
      ctx.strokeRect(dx, dy, dWidth, dHeight);

      if (Array.isArray(prediction.keypoints) && prediction.keypoints.length) {
        const kpts = prediction.keypoints;

        // skeleton
        if (kpts.length === 17) {
          ctx.strokeStyle = "rgba(255, 64, 129, 0.9)";
          ctx.lineWidth = 2;
          for (const [a, b] of COCO17_EDGES) {
            const ka = kpts[a];
            const kb = kpts[b];
            if (!ka || !kb) continue;
            if ((ka.score ?? 0) < KEYPOINT_THRESHOLD || (kb.score ?? 0) < KEYPOINT_THRESHOLD) continue;
            ctx.beginPath();
            ctx.moveTo(ka.x * scaleX, ka.y * scaleY);
            ctx.lineTo(kb.x * scaleX, kb.y * scaleY);
            ctx.stroke();
          }
        }

        // points
        ctx.fillStyle = "rgba(255, 235, 59, 0.95)";
        for (const kp of kpts) {
          if (!kp) continue;
          if ((kp.score ?? 0) < KEYPOINT_THRESHOLD) continue;
          const px = kp.x * scaleX;
          const py = kp.y * scaleY;
          ctx.beginPath();
          ctx.arc(px, py, 3, 0, Math.PI * 2);
          ctx.fill();
        }
      }

      const textWidth = ctx.measureText(label).width;
      const textHeight = 20;
      const textX = dx;
      const textY = dy > textHeight ? dy - 4 : dy + textHeight;

      ctx.fillStyle = "#00FFFF";
      ctx.fillRect(textX, textY - textHeight, textWidth + 10, textHeight);

      ctx.fillStyle = "#000000";
      ctx.fillText(label, textX + 5, textY - 5);
    }
  }

  function renderResults(predictions) {
    if (!results) return;
    results.innerHTML = "";

    if (predictions.length === 0) {
      const li = document.createElement("li");
      li.className = "muted";
      li.textContent = "未找到可辨識物件";
      results.appendChild(li);
      return;
    }

    for (const item of predictions) {
      const li = document.createElement("li");
      const score = (item.score * 100).toFixed(1);
      li.textContent = `類別 ${item.class}（${score}%）`;
      results.appendChild(li);
    }
  }

  function formatPredictionsSummary(predictions, prefix = "偵測完成") {
    if (predictions.length === 0) {
      return `${prefix}，未找到可辨識物件`;
    }

    const summary = predictions
      .map((item, index) => {
        const score = (item.score * 100).toFixed(1);
        return `${index + 1}. 類別 ${item.class} (${score}%)`;
      })
      .join("\n");

    return `${prefix}，共找到 ${predictions.length} 個物件：\n${summary}`;
  }

  async function loadModel() {
    try {
      setBadge("ok", "載入中");
      status.textContent = `正在載入模型：${activeModelConfig.label}...`;
      updateButtons();

      if (model?.dispose) {
        try {
          model.dispose();
        } catch {
          // ignore
        }
      }
      model = null;
      classNames = null;
      updateButtons();

      model = await tf.loadGraphModel(activeModelConfig.url);
      classNames = await loadClassNames(activeModelConfig);

      // Debug: 印出模型輸出資訊（pose/detect 可能輸出 tensor 意義不同）
      try {
        const dummy = tf.zeros([1, activeModelConfig.inputSize, activeModelConfig.inputSize, 3]);
        const raw = model.execute(dummy);
        const isArray = Array.isArray(raw);
        const isMap = raw && typeof raw === "object" && !raw.dataSync && !isArray;
        const keys = isMap ? Object.keys(raw) : null;
        const outputs = normalizeModelOutputs(raw);

        console.groupCollapsed(
          `[tf-webcam] model loaded: ${activeModelConfig.id} (${activeModelConfig.url})`,
        );
        console.log("execute() return type:", isArray ? "Tensor[]" : isMap ? "NamedTensorMap" : "Tensor");
        if (keys) console.log("output keys:", keys);
        console.log(
          "normalized outputs:",
          outputs.map((t, i) => ({
            i,
            name: t?.name,
            dtype: t?.dtype,
            shape: t?.shape,
          })),
        );
        const picked = pickYoloTensors(outputs);
        console.log("picked tensors:", {
          scores: { name: picked.scores?.name, dtype: picked.scores?.dtype, shape: picked.scores?.shape },
          det: { name: picked.det?.name, dtype: picked.det?.dtype, shape: picked.det?.shape },
          proto: { name: picked.proto?.name, dtype: picked.proto?.dtype, shape: picked.proto?.shape },
        });
        console.groupEnd();

        if (raw && raw.dispose) {
          raw.dispose();
        } else if (Array.isArray(raw)) {
          for (const t of raw) t?.dispose?.();
        } else if (isMap && keys) {
          for (const k of keys) raw[k]?.dispose?.();
        }
        dummy.dispose();
      } catch (e) {
        console.warn("[tf-webcam] debug execute() failed:", e);
      }

      setStatus(`模型載入完成（${activeModelConfig.label}），請選擇圖片`, "ok");
      updateButtons();
    } catch (error) {
      console.error(error);
      setStatus(`模型載入失敗：${error.message}`, "warn");
      updateButtons();
    }
  }

  async function loadAppVersion() {
    if (!appVersion) return;
    try {
      const res = await fetch(`./version.json?t=${Date.now()}`, { cache: "no-store" });
      if (!res.ok) {
        appVersion.textContent = "v?";
        return;
      }
      const data = await res.json();
      const short = (data.commit || data.version || "").toString().trim();
      appVersion.textContent = short ? `v${short}` : "v?";
      if (data.builtAt) {
        appVersion.title = `部署版本\n${short}\n${data.builtAt}`;
      }
    } catch {
      appVersion.textContent = "v?";
    }
  }

  async function detectForElement(sourceElement, modeLabel) {
    if (!model) {
      setStatus("模型尚未載入完成", "warn");
      return;
    }

    if (sourceElement === image && !image.src) {
      setStatus("請先選擇一張圖片", "warn");
      return;
    }

    try {
      updateButtons();
      setBadge("ok", "偵測中");
      status.textContent = `正在進行物件偵測（${activeModelConfig.label}）${
        modeLabel ? `：${modeLabel}` : ""
      }...`;

      syncCanvasSizeFor(sourceElement);
      const { scaleX, scaleY } = getDrawScaleFor(sourceElement);
      const predictions = await runYolo(model, sourceElement, activeModelConfig, classNames);
      drawPredictions(predictions, scaleX, scaleY);
      renderResults(predictions);

      setStatus(formatPredictionsSummary(predictions), "ok");
    } catch (error) {
      console.error(error);
      setStatus(`偵測失敗：${error.message}`, "warn");
    } finally {
      updateButtons();
      setBadge("ok", "就緒");
    }
  }

  async function startCamera() {
    try {
      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        setStatus("此瀏覽器不支援相機存取", "warn");
        return;
      }

      if (stream) {
        return;
      }

      setStatus("正在開啟相機...", "ok");

      hideVideoPreview();
      hideImagePreview();
      stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "environment" },
        audio: false,
      });

      video.srcObject = stream;
      await video.play();

      // 等待第一幀可用，避免剛開啟時顯示「死圖/黑屏」或錯誤尺寸
      await new Promise((resolve) => {
        if (video.readyState >= 2 && video.videoWidth) return resolve();
        const onReady = () => {
          video.removeEventListener("loadeddata", onReady);
          resolve();
        };
        video.addEventListener("loadeddata", onReady, { once: true });
        // 備援：若 loadeddata 沒觸發，輪詢直到拿到尺寸
        const start = performance.now();
        const tick = () => {
          if (video.videoWidth) return resolve();
          if (performance.now() - start > 2000) return resolve();
          requestAnimationFrame(tick);
        };
        requestAnimationFrame(tick);
      });

      video.style.display = "block";

      syncCanvasSizeFor(video);
      clearCanvas();

      cameraButton.textContent = "關閉相機";
      updateButtons();

      setStatus("相機已開啟，可按「即時偵測」或「拍照偵測」", "ok");
    } catch (error) {
      console.error(error);
      setStatus(`開啟相機失敗：${error.message}`, "warn");
      updateButtons();
    }
  }

  function stopCamera() {
    if (stream) {
      for (const track of stream.getTracks()) {
        track.stop();
      }
      stream = null;
    }

    isLiveDetecting = false;
    isProcessingFrame = false;
    liveDetectButton.textContent = "開始即時偵測";

    video.srcObject = null;
    video.style.display = "none";
    showImagePreview();

    clearCanvas();
    cameraButton.textContent = "開啟相機";
    updateButtons();
  }

  async function captureFromCameraOnce() {
    if (!stream || !video.videoWidth) {
      setStatus("請先開啟相機", "warn");
      return;
    }
    if (!model) {
      setStatus("模型尚未載入完成", "warn");
      return;
    }

    await detectForElement(video, "相機畫面");
  }

  async function liveDetectLoop() {
    if (!isLiveDetecting || !stream) {
      return;
    }

    if (!video.videoWidth || isProcessingFrame) {
      requestAnimationFrame(liveDetectLoop);
      return;
    }

    isProcessingFrame = true;

    try {
      syncCanvasSizeFor(video);
      const { scaleX, scaleY } = getDrawScaleFor(video);
      const predictions = await runYolo(model, video, activeModelConfig, classNames);
      drawPredictions(predictions, scaleX, scaleY);
      renderResults(predictions);
      setBadge("ok", "即時中");
    } catch (error) {
      console.error(error);
      setStatus(`即時偵測錯誤：${error.message}`, "warn");
      isLiveDetecting = false;
      liveDetectButton.textContent = "開始即時偵測";
      setBadge("warn", "已停止");
    } finally {
      isProcessingFrame = false;
      if (isLiveDetecting) {
        requestAnimationFrame(liveDetectLoop);
      } else {
        setBadge("ok", "就緒");
      }
    }
  }

  fileInput.addEventListener("change", (event) => {
    const file = event.target.files?.[0];
    if (!file) {
      return;
    }

    if (stream) {
      stopCamera();
    }

    if (currentImageUrl) {
      URL.revokeObjectURL(currentImageUrl);
      currentImageUrl = null;
    }

    // 先隱藏，避免舊快取影像/破圖在載入新圖前一瞬間顯示
    hideImagePreview();
    hideVideoPreview();
    clearCanvas();
    renderResults([]);
    setStatus("正在載入圖片...", "ok");

    currentImageUrl = URL.createObjectURL(file);
    image.src = currentImageUrl;
    updateButtons();
  });

  image.addEventListener("load", () => {
    showImagePreview();
    syncCanvasSizeFor(image);
    clearCanvas();
    setStatus("圖片已載入，請點擊「執行物件偵測」", "ok");
    updateButtons();
  });

  image.addEventListener("error", () => {
    hideImagePreview();
    clearCanvas();
    renderResults([]);
    setStatus("圖片載入失敗，請重新選擇檔案", "warn");
    updateButtons();
  });

  detectButton.addEventListener("click", () => {
    detectForElement(image, "圖片");
  });

  cameraButton.addEventListener("click", () => {
    if (stream) {
      stopCamera();
      setStatus("相機已關閉", "ok");
    } else {
      startCamera();
    }
  });

  liveDetectButton.addEventListener("click", () => {
    if (!stream) {
      setStatus("請先開啟相機", "warn");
      return;
    }
    if (!model) {
      setStatus("模型尚未載入完成", "warn");
      return;
    }

    isLiveDetecting = !isLiveDetecting;

    if (isLiveDetecting) {
      liveDetectButton.textContent = "停止即時偵測";
      setStatus("即時偵測中...", "ok");
      liveDetectLoop();
    } else {
      liveDetectButton.textContent = "開始即時偵測";
      setStatus("已停止即時偵測", "ok");
      setBadge("ok", "就緒");
    }
  });

  captureButton.addEventListener("click", () => {
    captureFromCameraOnce();
  });

  async function switchModel(nextKey) {
    const nextConfig = MODEL_CONFIGS[nextKey];
    if (!nextConfig) return;
    if (activeModelKey === nextKey && model) return;

    // 切換模型前先停止即時偵測，避免同時推論
    isLiveDetecting = false;
    isProcessingFrame = false;
    if (liveDetectButton) liveDetectButton.textContent = "開始即時偵測";

    activeModelKey = nextKey;
    activeModelConfig = nextConfig;

    clearCanvas();
    renderResults([]);
    setBadge("ok", "載入中");
    setStatus(`切換模型中：${activeModelConfig.label}...`, "ok");
    updateButtons();

    await loadModel();
  }

  if (modelSelect) {
    modelSelect.value = activeModelKey;
    modelSelect.addEventListener("change", () => {
      const nextKey = modelSelect.value;
      switchModel(nextKey);
    });
  }

  window.addEventListener("beforeunload", () => {
    if (currentImageUrl) {
      URL.revokeObjectURL(currentImageUrl);
      currentImageUrl = null;
    }
    stopCamera();
    if (model?.dispose) {
      try {
        model.dispose();
      } catch {
        // ignore
      }
    }
  });

  // 初始化時先隱藏預覽，避免載入瞬間出現「死圖/破圖」
  hideImagePreview();
  hideVideoPreview();
  try {
    image.removeAttribute("src");
  } catch {
    // ignore
  }
  clearCanvas();
  updateButtons();
  renderResults([]);
  setBadge("ok", "載入中");
  loadAppVersion();
  // 預設使用 pose
  if (modelSelect) {
    modelSelect.value = activeModelKey;
  }
  loadModel();
})();

