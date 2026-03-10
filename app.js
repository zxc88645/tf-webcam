(() => {
  "use strict";

  const MODEL_URL = "yolo26n_web_model/model.json";
  const INPUT_SIZE = 640; // 匯出 YOLO 時的輸入尺寸
  const SCORE_THRESHOLD = 0.3;

  const fileInput = document.getElementById("fileInput");
  const detectButton = document.getElementById("detectButton");
  const cameraButton = document.getElementById("cameraButton");
  const liveDetectButton = document.getElementById("liveDetectButton");
  const captureButton = document.getElementById("captureButton");
  const image = document.getElementById("image");
  const video = document.getElementById("video");
  const overlay = document.getElementById("overlay");
  const stage = document.getElementById("stage");
  const status = document.getElementById("status");
  const statusBadge = document.getElementById("statusBadge");
  const results = document.getElementById("results");
  const ctx = overlay.getContext("2d");

  let model = null;
  let currentImageUrl = null;
  let stream = null;
  let isLiveDetecting = false;
  let isProcessingFrame = false;

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
    ctx.clearRect(0, 0, overlay.width, overlay.height);
  }

  function syncCanvasSizeFor(element) {
    const width =
      element.videoWidth ||
      element.naturalWidth ||
      element.width ||
      element.clientWidth;
    const height =
      element.videoHeight ||
      element.naturalHeight ||
      element.height ||
      element.clientHeight;

    if (!width || !height) {
      return;
    }

    overlay.width = width;
    overlay.height = height;
    overlay.style.width = `${width}px`;
    overlay.style.height = `${height}px`;

    element.style.width = `${width}px`;
    element.style.height = `${height}px`;

    stage.style.width = `${width}px`;
  }

  function drawPredictions(predictions) {
    clearCanvas();

    ctx.lineWidth = 2;
    ctx.font = "16px Arial";

    for (const prediction of predictions) {
      const [x, y, width, height] = prediction.bbox;
      const score = (prediction.score * 100).toFixed(1);
      const label = `${prediction.class} ${score}%`;

      ctx.strokeStyle = "#00FFFF";
      ctx.strokeRect(x, y, width, height);

      const textWidth = ctx.measureText(label).width;
      const textHeight = 20;
      const textX = x;
      const textY = y > textHeight ? y - 4 : y + textHeight;

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
      status.textContent = "正在載入 YOLO 模型...";
      updateButtons();

      model = await tf.loadGraphModel(MODEL_URL);

      setStatus("YOLO 模型載入完成，請選擇圖片", "ok");
      updateButtons();
    } catch (error) {
      console.error(error);
      setStatus(`模型載入失敗：${error.message}`, "warn");
      updateButtons();
    }
  }

  async function runYolo(imageElement) {
    if (!model) {
      throw new Error("模型尚未載入");
    }

    return tf.tidy(() => {
      let img = tf.browser.fromPixels(imageElement).toFloat();

      img = tf.image.resizeBilinear(img, [INPUT_SIZE, INPUT_SIZE]);
      img = img.expandDims(0).div(255.0);

      const raw = model.execute(img);

      if (!Array.isArray(raw) || raw.length < 2) {
        console.warn("Unexpected YOLO output format:", raw);
        throw new Error("YOLO 模型輸出格式與預期不同（需要 2 個 tensor）。");
      }

      const scoresArr = raw[0].dataSync();
      const detArr = raw[1].dataSync();

      const numDet = scoresArr.length;
      const imgWidth = imageElement.naturalWidth || imageElement.width;
      const imgHeight = imageElement.naturalHeight || imageElement.height;
      const scaleX = imgWidth / INPUT_SIZE;
      const scaleY = imgHeight / INPUT_SIZE;

      const predictions = [];

      for (let i = 0; i < numDet; i++) {
        const base = i * 6;
        if (base + 5 >= detArr.length) break;

        const x1 = detArr[base + 0];
        const y1 = detArr[base + 1];
        const x2 = detArr[base + 2];
        const y2 = detArr[base + 3];
        const scoreFromDet = detArr[base + 4];
        const clsId = detArr[base + 5];

        const score = Math.max(scoresArr[i] ?? 0, scoreFromDet ?? 0);
        if (score < SCORE_THRESHOLD) continue;

        const x = x1 * scaleX;
        const y = y1 * scaleY;
        const wBox = (x2 - x1) * scaleX;
        const hBox = (y2 - y1) * scaleY;

        predictions.push({
          bbox: [x, y, wBox, hBox],
          class: String(clsId),
          score,
        });
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
      status.textContent = `正在進行物件偵測 (YOLO)${modeLabel ? `：${modeLabel}` : ""}...`;

      syncCanvasSizeFor(sourceElement);
      const predictions = await runYolo(sourceElement);
      drawPredictions(predictions);
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

      stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "environment" },
        audio: false,
      });

      video.srcObject = stream;
      await video.play();

      image.style.display = "none";
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
    image.style.display = "block";

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
      const predictions = await runYolo(video);
      drawPredictions(predictions);
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

    currentImageUrl = URL.createObjectURL(file);
    image.src = currentImageUrl;
    clearCanvas();
    renderResults([]);
    setStatus("圖片已載入，請點擊「執行物件偵測」", "ok");
    updateButtons();
  });

  image.addEventListener("load", () => {
    syncCanvasSizeFor(image);
    clearCanvas();
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

  window.addEventListener("beforeunload", () => {
    if (currentImageUrl) {
      URL.revokeObjectURL(currentImageUrl);
      currentImageUrl = null;
    }
    stopCamera();
  });

  updateButtons();
  renderResults([]);
  setBadge("ok", "載入中");
  loadModel();
})();

