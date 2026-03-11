import type { Tensor } from "@tensorflow/tfjs";
import * as tf from "@tensorflow/tfjs";
import type { ModelConfig, YoloPrediction } from "./types";

export const SCORE_THRESHOLD = 0.3;
export const KEYPOINT_THRESHOLD = 0.25;

export const COCO17_EDGES: ReadonlyArray<readonly [number, number]> = [
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

function parseMetadataYamlNames(yamlText: string): string[] | null {
  const lines = String(yamlText || "").split(/\r?\n/);
  const names: string[] = [];
  let inNames = false;
  let baseIndent: number | null = null;

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

    const currentIndent = line.match(/^(\s*)/)?.[1]?.length ?? 0;
    if (
      currentIndent < baseIndent ||
      (/^\s*\w+\s*:/.test(line) && currentIndent === 0)
    ) {
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

export async function loadClassNames(
  activeModelConfig: ModelConfig,
): Promise<string[] | null> {
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

export type NamedTensorMapLike = Record<string, Tensor>;

export function normalizeModelOutputs(raw: unknown): Tensor[] {
  if (Array.isArray(raw)) return raw as Tensor[];

  // NamedTensorMap: keys order not guaranteed
  if (raw && typeof raw === "object" && !(raw as Tensor).dataSync) {
    const map = raw as NamedTensorMapLike;
    const keys = Object.keys(map);

    const topkKey = keys.find((k) => /TopKV2:0$/i.test(k));
    const identityKey = keys.find(
      (k) => /^Identity:0$/i.test(k) || /\/Identity:0$/i.test(k),
    );
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

  if (raw) return [raw as Tensor];
  return [];
}

export function pickYoloTensors(outputs: Tensor[]) {
  if (!Array.isArray(outputs) || outputs.length === 0) {
    return {
      scores: null as Tensor | null,
      det: null as Tensor | null,
      proto: null as Tensor | null,
    };
  }

  let scores: Tensor | null = null;
  let det: Tensor | null = null;
  let proto: Tensor | null = null;

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

  if (!scores || !det) {
    const a = outputs[0];
    const b = outputs[1];
    return { scores: scores ?? a ?? null, det: det ?? b ?? null, proto };
  }

  return { scores, det, proto };
}

export function runYolo(
  model: { execute: (x: Tensor) => unknown },
  imageElement: HTMLImageElement | HTMLVideoElement,
  activeModelConfig: ModelConfig,
  classNames: string[] | null,
): YoloPrediction[] {
  if (!model) {
    throw new Error("模型尚未載入");
  }

  return tf.tidy(() => {
    let img = tf.browser.fromPixels(imageElement).toFloat();
    img = tf.image.resizeBilinear(img, [
      activeModelConfig.inputSize,
      activeModelConfig.inputSize,
    ]);
    img = img.expandDims(0).div(255.0);

    const raw = model.execute(img);
    const outputs = normalizeModelOutputs(raw);
    const {
      scores: scoresTensor,
      det: detTensor,
      proto: protoTensor,
    } = pickYoloTensors(outputs);

    if (!scoresTensor || !detTensor) {
      console.warn("Unexpected YOLO output format:", raw);
      throw new Error("YOLO 模型輸出格式與預期不同（需要至少 2 個 tensor）。");
    }

    const scoresArr = scoresTensor.dataSync() as
      | Float32Array
      | Int32Array
      | Uint8Array;
    const detArr = detTensor.dataSync() as
      | Float32Array
      | Int32Array
      | Uint8Array;

    const numDet = scoresArr.length;
    const imgWidth =
      (imageElement as HTMLVideoElement).videoWidth ||
      (imageElement as HTMLImageElement).naturalWidth ||
      imageElement.width;
    const imgHeight =
      (imageElement as HTMLVideoElement).videoHeight ||
      (imageElement as HTMLImageElement).naturalHeight ||
      imageElement.height;
    const scaleX = imgWidth / activeModelConfig.inputSize;
    const scaleY = imgHeight / activeModelConfig.inputSize;

    const predictions: YoloPrediction[] = [];
    const stride = Math.max(6, Math.floor(detArr.length / Math.max(1, numDet)));
    const extra = Math.max(0, stride - 6);
    const canParseKeypoints = extra > 0 && extra % 3 === 0;
    const numKeypoints = canParseKeypoints ? extra / 3 : 0;
    const canParseMasks =
      Boolean(protoTensor) && extra > 0 && !canParseKeypoints;
    const numMaskCoeffs = canParseMasks ? extra : 0;

    // seg: prepare proto2d once
    let proto2d: Tensor | null = null;
    let protoMh = 0;
    let protoMw = 0;
    let protoNm = 0;
    if (canParseMasks && protoTensor) {
      const shape = protoTensor.shape || [];
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
          proto2d = null;
        }
      }
    }

    for (let i = 0; i < numDet; i++) {
      const base = i * stride;
      if (base + 5 >= detArr.length) break;

      const x1 = Number(detArr[base + 0]);
      const y1 = Number(detArr[base + 1]);
      const x2 = Number(detArr[base + 2]);
      const y2 = Number(detArr[base + 3]);
      const scoreFromDet = Number(detArr[base + 4]);
      const clsId = Number(detArr[base + 5]);
      const clsIndex = Number.isFinite(clsId)
        ? Math.round(clsId)
        : Number(clsId);
      const classLabel =
        (classNames && Number.isInteger(clsIndex) && classNames[clsIndex]) ||
        String(clsIndex);

      const score = Math.max(
        Number(scoresArr[i] ?? 0),
        Number.isFinite(scoreFromDet) ? scoreFromDet : 0,
      );
      if (score < SCORE_THRESHOLD) continue;

      const x = x1 * scaleX;
      const y = y1 * scaleY;
      const wBox = (x2 - x1) * scaleX;
      const hBox = (y2 - y1) * scaleY;

      const prediction: YoloPrediction = {
        bbox: [x, y, wBox, hBox],
        class: classLabel,
        classId: String(clsIndex),
        score,
      };

      if (numKeypoints) {
        const keypoints: Array<{ x: number; y: number; score: number }> = [];
        for (let k = 0; k < numKeypoints; k++) {
          const off = base + 6 + k * 3;
          if (off + 2 >= detArr.length) break;
          const kx = Number(detArr[off + 0]) * scaleX;
          const ky = Number(detArr[off + 1]) * scaleY;
          const ks = Number(detArr[off + 2]);
          keypoints.push({ x: kx, y: ky, score: ks });
        }
        prediction.keypoints = keypoints;
      }

      if (numMaskCoeffs && proto2d) {
        const coeffs = new Float32Array(numMaskCoeffs);
        for (let m = 0; m < numMaskCoeffs; m++) {
          const off = base + 6 + m;
          if (off >= detArr.length) break;
          coeffs[m] = Number(detArr[off]);
        }

        const coeffCol = tf.tensor2d(coeffs, [numMaskCoeffs, 1], "float32");
        const maskFlat = tf.sigmoid(tf.matMul(proto2d, coeffCol)); // [mh*mw,1]
        const maskProto = tf.reshape(maskFlat, [protoMh, protoMw, 1]); // [mh,mw,1]
        const maskUp = tf.image.resizeBilinear(
          maskProto,
          [activeModelConfig.inputSize, activeModelConfig.inputSize],
          true,
        ); // [S,S,1]

        const sx1 = Math.max(
          0,
          Math.min(activeModelConfig.inputSize - 1, Math.floor(x1)),
        );
        const sy1 = Math.max(
          0,
          Math.min(activeModelConfig.inputSize - 1, Math.floor(y1)),
        );
        const sx2 = Math.max(
          0,
          Math.min(activeModelConfig.inputSize, Math.ceil(x2)),
        );
        const sy2 = Math.max(
          0,
          Math.min(activeModelConfig.inputSize, Math.ceil(y2)),
        );
        const cropW = Math.max(1, sx2 - sx1);
        const cropH = Math.max(1, sy2 - sy1);

        const maskCrop = tf.slice(maskUp, [sy1, sx1, 0], [cropH, cropW, 1]); // [h,w,1]
        const maskBin = tf.greater(maskCrop, 0.5).toInt(); // 0/1
        const maskData = maskBin.dataSync() as unknown as Int32Array;

        prediction.maskCrop = {
          data: Uint8Array.from(maskData, (v) => (v ? 1 : 0)),
          inputW: cropW,
          inputH: cropH,
        };
      }

      predictions.push(prediction);
    }

    return predictions;
  });
}
