// 模型與推論相關的核心工具（不直接操作 DOM / Canvas）
// - YOLO 輸出解析
// - Metadata 解析
// - 門檻常數

export const SCORE_THRESHOLD = 0.3;
export const KEYPOINT_THRESHOLD = 0.25;

// COCO-17 keypoints skeleton (常見 YOLO pose 輸出)
export const COCO17_EDGES = [
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
    if (currentIndent < baseIndent || (/^\s*\w+\s*:/.test(line) && currentIndent === 0)) {
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

export async function loadClassNames(activeModelConfig) {
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

export function normalizeModelOutputs(raw) {
  // GraphModel.execute() 可能回傳：
  // - Tensor[]
  // - NamedTensorMap（key順序不保證）
  // - Tensor
  if (Array.isArray(raw)) return raw;

  if (raw && typeof raw === "object" && !raw.dataSync) {
    const map = raw;
    const keys = Object.keys(map);

    // 這些模型的 signature.outputs 常見包含：
    // - Identity:0 (det / boxes+attrs)
    // - Identity_1:0 (seg proto)
    // - .../TopKV2:0 (scores)
    const topkKey = keys.find((k) => /TopKV2:0$/i.test(k));
    const identityKey = keys.find((k) => /^Identity:0$/i.test(k) || /\/Identity:0$/i.test(k));
    const identity1Key = keys.find(
      (k) => /^Identity_1:0$/i.test(k) || /\/Identity_1:0$/i.test(k),
    );

    if (topkKey && identityKey && identity1Key) {
      return [map[topkKey], map[identityKey], map[identity1Key]];
    }

    if (topkKey && identityKey) {
      return [map[topkKey], map[identityKey]];
    }

    const values = Object.values(map);
    if (values.length) return values;
  }

  if (raw) return [raw];
  return [];
}

export function pickYoloTensors(outputs) {
  // 目標：在不同模型/不同輸出順序下，穩定挑出
  // - scores tensor（通常 rank=2: [1, N]）
  // - det tensor（通常 rank=3: [1, N, K]，且 K>=6）
  // - proto tensor（seg 才有，常見 rank=4: [1, mh, mw, nm] 或 [1, nm, mh, mw]）
  if (!Array.isArray(outputs) || outputs.length === 0) {
    return { scores: null, det: null, proto: null };
  }

  let scores = null;
  let det = null;
  let proto = null;

  for (const t of outputs) {
    const rank = Array.isArray(t?.shape) ? t.shape.length : null;
    const last = rank ? t.shape[rank - 1] : null;

    if (!scores && rank === 2) {
      scores = t;
      continue;
    }
    if (!det && (rank === 3 || rank === 2) && (last == null || last >= 6)) {
      det = t;
      continue;
    }
    if (!proto && rank === 4) {
      proto = t;
      continue;
    }
  }

  // 後備：若沒找到 scores/det，沿用舊邏輯（前 2 個）
  if (!scores || !det) {
    const a = outputs[0];
    const b = outputs[1];
    return { scores: scores ?? a, det: det ?? b, proto };
  }

  return { scores, det, proto };
}

export function runYolo(model, imageElement, activeModelConfig, classNames) {
  if (!model) {
    throw new Error("模型尚未載入");
  }

  return tf.tidy(() => {
    let img = tf.browser.fromPixels(imageElement).toFloat();

    img = tf.image.resizeBilinear(img, [activeModelConfig.inputSize, activeModelConfig.inputSize]);
    img = img.expandDims(0).div(255.0);

    const raw = model.execute(img);
    const outputs = normalizeModelOutputs(raw);
    const { scores: scoresTensor, det: detTensor, proto: protoTensor } = pickYoloTensors(outputs);

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
    const extra = Math.max(0, stride - 6);
    const canParseKeypoints = extra > 0 && extra % 3 === 0;
    const numKeypoints = canParseKeypoints ? extra / 3 : 0;
    const canParseMasks = Boolean(protoTensor) && extra > 0 && !canParseKeypoints;
    const numMaskCoeffs = canParseMasks ? extra : 0;
    if (typeof window !== "undefined") {
      // Debug: 用 collapsed group 避免刷屏
      console.groupCollapsed?.("[tf-webcam] runYolo debug");
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
      console.log("numDet:", numDet, "detArr.length:", detArr.length, "stride:", stride, "numKeypoints:", numKeypoints, "numMaskCoeffs:", numMaskCoeffs);
      console.groupEnd?.();
    }

    // seg: 先準備 proto2d，避免每個物件都 reshape 一次
    let proto2d = null;
    let protoMh = 0;
    let protoMw = 0;
    let protoNm = 0;
    if (canParseMasks && protoTensor) {
      const shape = protoTensor.shape || [];
      // 支援 [1, mh, mw, nm] 與 [1, nm, mh, mw]
      if (shape.length === 4) {
        const [, d1, d2, d3] = shape;
        if (d3 === numMaskCoeffs) {
          protoMh = d1;
          protoMw = d2;
          protoNm = d3;
          proto2d = tf.reshape(protoTensor, [protoMh * protoMw, protoNm]);
        } else if (d1 === numMaskCoeffs) {
          protoNm = d1;
          protoMh = d2;
          protoMw = d3;
          const transposed = tf.transpose(protoTensor, [0, 2, 3, 1]); // -> [1,mh,mw,nm]
          proto2d = tf.reshape(transposed, [protoMh * protoMw, protoNm]);
        } else {
          // 格式未知，放棄 masks
          proto2d = null;
        }
      }
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

      if (numMaskCoeffs && proto2d) {
        const coeffs = new Float32Array(numMaskCoeffs);
        for (let m = 0; m < numMaskCoeffs; m++) {
          const off = base + 6 + m;
          if (off >= detArr.length) break;
          coeffs[m] = detArr[off];
        }

        // 依照 Ultralytics YOLO seg：mask = sigmoid(proto @ coeffs)
        // 我們只回傳 bbox 範圍的二值遮罩（在 inputSize 座標系），以降低傳輸量與繪圖負擔。
        const coeffCol = tf.tensor2d(coeffs, [numMaskCoeffs, 1], "float32");
        const maskFlat = tf.sigmoid(tf.matMul(proto2d, coeffCol)); // [mh*mw,1]
        const maskProto = tf.reshape(maskFlat, [protoMh, protoMw, 1]); // [mh,mw,1]
        const maskUp = tf.image.resizeBilinear(
          maskProto,
          [activeModelConfig.inputSize, activeModelConfig.inputSize],
          true,
        ); // [S,S,1]

        const sx1 = Math.max(0, Math.min(activeModelConfig.inputSize - 1, Math.floor(x1)));
        const sy1 = Math.max(0, Math.min(activeModelConfig.inputSize - 1, Math.floor(y1)));
        const sx2 = Math.max(0, Math.min(activeModelConfig.inputSize, Math.ceil(x2)));
        const sy2 = Math.max(0, Math.min(activeModelConfig.inputSize, Math.ceil(y2)));
        const cropW = Math.max(1, sx2 - sx1);
        const cropH = Math.max(1, sy2 - sy1);

        const maskCrop = tf.slice(maskUp, [sy1, sx1, 0], [cropH, cropW, 1]); // [h,w,1]
        const maskBin = tf.greater(maskCrop, 0.5).toInt(); // 0/1
        const maskData = maskBin.dataSync(); // Int32Array

        prediction.maskCrop = {
          // bbox in original image pixels (already scaled in bbox field), plus crop metadata
          inputX: sx1,
          inputY: sy1,
          inputW: cropW,
          inputH: cropH,
          data: maskData,
        };
      }

      predictions.push(prediction);
    }

    return predictions;
  });
}

