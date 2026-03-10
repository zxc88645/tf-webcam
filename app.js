(() => {
  "use strict";

  const SCORE_THRESHOLD = 0.3;
  const KEYPOINT_THRESHOLD = 0.25;

  // COCO-17 keypoints skeleton (常見 YOLO pose 輸出)
  const COCO17_EDGES = [
    [0, 1],
    [0, 2],
    [1, 3],
    [2, 4],
    [3, 5],
    [4, 6],
    [5, 6],
    [5, 7],
    [7, 9],
    [6, 8],
    [8, 10],
    [5, 11],
    [6, 12],
    [11, 12],
    [11, 13],
    [13, 15],
    [12, 14],
    [14, 16],
  ];

  const MODEL_CONFIGS = {
    pose: {
      id: "pose",
      label: "YOLOv26n Pose",
      url: "yolo26n-pose_web_model/model.json",
      inputSize: 640,
    },
    detect: {
      id: "detect",
      label: "YOLOv26n Detect",
      url: "yolo26n_web_model/model.json",
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

  function parseMetadataYamlNames(yamlText) {
    const lines = String(yamlText || "").split(/\r?\n/);
    const names = [];
    let inNames = false;
    let baseIndent = null;

    for (const line of lines) {
      if (!inNames) {
        if (/^\s*names\s*:\s*$/.test(line)) {
          inNames = true;
        }
        continue;
      }

      if (baseIndent == null) {
        const mIndent = line.match(/^(\s+)\S/);
        if (!mIndent) continue;
        baseIndent = mIndent[1].length;
      }

      const currentIndent = (line.match(/^(\s*)/)?.[1]?.length) ?? 0;
      if (currentIndent < baseIndent || /^\s*\w+\s*:/.test(line) && currentIndent === 0) {
        break;
      }

      const m = line.match(/^\s*(\d+)\s*:\s*(.+?)\s*$/);
      if (!m) continue;
      const idx = Number(m[1]);
      const name = m[2];
      names[idx] = name;
    }

    return names.length ? names : null;
  }

  async function loadClassNames() {
    try {
      const metadataUrl = activeModelConfig.url.replace(
        /model\.json(\?.*)?$/i,
        "metadata.yaml$1",
      );
      const res = await fetch(metadataUrl, { cache: "no-cache" });
      if (!res.ok) return null;
      const text = await res.text();
      return parseMetadataYamlNames(text);
    } catch {
      return null;
    }
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
      classNames = await loadClassNames();

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
        const picked = pickScoresAndDetTensors(outputs);
        console.log("picked tensors:", {
          scores: { name: picked.scores?.name, dtype: picked.scores?.dtype, shape: picked.scores?.shape },
          det: { name: picked.det?.name, dtype: picked.det?.dtype, shape: picked.det?.shape },
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

  function normalizeModelOutputs(raw) {
    // GraphModel.execute() 可能回傳：
    // - Tensor[]
    // - NamedTensorMap（key順序不保證）
    // - Tensor
    if (Array.isArray(raw)) return raw;

    if (raw && typeof raw === "object" && !raw.dataSync) {
      const map = raw;
      const keys = Object.keys(map);

      // 這兩個模型的 signature.outputs 會包含：
      // - Identity:0
      // - .../TopKV2:0
      const topkKey = keys.find((k) => /TopKV2:0$/i.test(k));
      const identityKey = keys.find((k) => /^Identity:0$/i.test(k) || /\/Identity:0$/i.test(k));

      if (topkKey && identityKey) {
        return [map[topkKey], map[identityKey]];
      }

      const values = Object.values(map);
      if (values.length) return values;
    }

    if (raw) return [raw];
    return [];
  }

  function pickScoresAndDetTensors(outputs) {
    // 目標：在不同模型/不同輸出順序下，穩定挑出
    // - scores tensor（通常 rank=2: [1, N]）
    // - det tensor（通常 rank=3: [1, N, K]，且 K>=6）
    if (!Array.isArray(outputs) || outputs.length < 2) return { scores: outputs?.[0], det: outputs?.[1] };

    const a = outputs[0];
    const b = outputs[1];
    const aRank = Array.isArray(a?.shape) ? a.shape.length : null;
    const bRank = Array.isArray(b?.shape) ? b.shape.length : null;

    const aLast = aRank ? a.shape[aRank - 1] : null;
    const bLast = bRank ? b.shape[bRank - 1] : null;

    const aLooksLikeDet = (aRank === 3 && (aLast == null || aLast >= 6)) || (aRank === 2 && aLast >= 6);
    const bLooksLikeDet = (bRank === 3 && (bLast == null || bLast >= 6)) || (bRank === 2 && bLast >= 6);

    if (aLooksLikeDet && !bLooksLikeDet) return { scores: b, det: a };
    if (bLooksLikeDet && !aLooksLikeDet) return { scores: a, det: b };

    // 後備：rank 大的多半是 det
    if ((aRank ?? 0) > (bRank ?? 0)) return { scores: b, det: a };
    if ((bRank ?? 0) > (aRank ?? 0)) return { scores: a, det: b };

    return { scores: a, det: b };
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

  async function runYolo(imageElement) {
    if (!model) {
      throw new Error("模型尚未載入");
    }

    return tf.tidy(() => {
      let img = tf.browser.fromPixels(imageElement).toFloat();

      img = tf.image.resizeBilinear(img, [activeModelConfig.inputSize, activeModelConfig.inputSize]);
      img = img.expandDims(0).div(255.0);

      const raw = model.execute(img);
      const outputs = normalizeModelOutputs(raw);
      const { scores: scoresTensor, det: detTensor } = pickScoresAndDetTensors(outputs);

      if (!scoresTensor || !detTensor) {
        console.warn("Unexpected YOLO output format:", raw);
        throw new Error("YOLO 模型輸出格式與預期不同（需要至少 2 個 tensor）。");
      }

      const scoresArr = scoresTensor.dataSync();
      const detArr = detTensor.dataSync();

      const numDet = scoresArr.length;
      const imgWidth =
        imageElement.videoWidth || imageElement.naturalWidth || imageElement.width;
      const imgHeight =
        imageElement.videoHeight || imageElement.naturalHeight || imageElement.height;
      const scaleX = imgWidth / activeModelConfig.inputSize;
      const scaleY = imgHeight / activeModelConfig.inputSize;

      const predictions = [];
      const stride = Math.max(6, Math.floor(detArr.length / Math.max(1, numDet)));
      const canParseKeypoints = stride > 6 && (stride - 6) % 3 === 0;
      const numKeypoints = canParseKeypoints ? (stride - 6) / 3 : 0;
      if (typeof window !== "undefined") {
        // Debug: 用 collapsed group 避免刷屏
        console.groupCollapsed?.(`[tf-webcam] runYolo debug (${activeModelConfig.id})`);
        console.log("scoresTensor:", {
          name: scoresTensor?.name,
          dtype: scoresTensor?.dtype,
          shape: scoresTensor?.shape,
          sample: Array.from(scoresArr.slice(0, Math.min(5, scoresArr.length))),
        });
        console.log("detTensor:", {
          name: detTensor?.name,
          dtype: detTensor?.dtype,
          shape: detTensor?.shape,
          sample: Array.from(detArr.slice(0, Math.min(12, detArr.length))),
        });
        console.log("numDet:", numDet, "detArr.length:", detArr.length, "stride:", stride, "numKeypoints:", numKeypoints);
        console.groupEnd?.();
      }

      for (let i = 0; i < numDet; i++) {
        const base = i * stride;
        if (base + 5 >= detArr.length) break;

        const x1 = detArr[base + 0];
        const y1 = detArr[base + 1];
        const x2 = detArr[base + 2];
        const y2 = detArr[base + 3];
        const scoreFromDet = detArr[base + 4];
        const clsId = detArr[base + 5];
        const clsIndex = Number.isFinite(clsId) ? Math.round(clsId) : Number(clsId);
        const classLabel =
          (classNames && Number.isInteger(clsIndex) && classNames[clsIndex]) ||
          String(clsIndex);

        const score = Math.max(scoresArr[i] ?? 0, scoreFromDet ?? 0);
        if (score < SCORE_THRESHOLD) continue;

        const x = x1 * scaleX;
        const y = y1 * scaleY;
        const wBox = (x2 - x1) * scaleX;
        const hBox = (y2 - y1) * scaleY;

        const prediction = {
          bbox: [x, y, wBox, hBox],
          class: classLabel,
          classId: String(clsIndex),
          score,
        };

        if (numKeypoints) {
          const keypoints = [];
          for (let k = 0; k < numKeypoints; k++) {
            const off = base + 6 + k * 3;
            if (off + 2 >= detArr.length) break;
            const kx = detArr[off + 0] * scaleX;
            const ky = detArr[off + 1] * scaleY;
            const ks = detArr[off + 2];
            keypoints.push({ x: kx, y: ky, score: ks });
          }
          prediction.keypoints = keypoints;
        }

        predictions.push(prediction);
      }

      return predictions;
    });
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
      const predictions = await runYolo(sourceElement);
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
      const predictions = await runYolo(video);
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

