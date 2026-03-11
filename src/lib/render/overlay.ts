import type { YoloPrediction } from "../yolo/types";
import { COCO17_EDGES, KEYPOINT_THRESHOLD } from "../yolo/yoloCore";

export type OverlayRenderOptions = {
  dpr?: number;
  scaleX?: number;
  scaleY?: number;
};

function clampByte(x: number) {
  return Math.max(0, Math.min(255, x | 0));
}

function hashStringToHue(input: string) {
  // fast, deterministic hash -> hue
  let h = 2166136261;
  for (let i = 0; i < input.length; i++) {
    h ^= input.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return ((h >>> 0) % 360) | 0;
}

function hslToRgb(h: number, s: number, l: number) {
  // h in [0,360), s,l in [0,1]
  const c = (1 - Math.abs(2 * l - 1)) * s;
  const hp = (h % 360) / 60;
  const x = c * (1 - Math.abs((hp % 2) - 1));
  let r1 = 0,
    g1 = 0,
    b1 = 0;
  if (hp >= 0 && hp < 1) [r1, g1, b1] = [c, x, 0];
  else if (hp < 2) [r1, g1, b1] = [x, c, 0];
  else if (hp < 3) [r1, g1, b1] = [0, c, x];
  else if (hp < 4) [r1, g1, b1] = [0, x, c];
  else if (hp < 5) [r1, g1, b1] = [x, 0, c];
  else [r1, g1, b1] = [c, 0, x];
  const m = l - c / 2;
  return {
    r: clampByte((r1 + m) * 255),
    g: clampByte((g1 + m) * 255),
    b: clampByte((b1 + m) * 255),
  };
}

let scratchCanvas: HTMLCanvasElement | null = null;
let scratchCtx: CanvasRenderingContext2D | null = null;

export function clearOverlay(canvas: HTMLCanvasElement) {
  const ctx = canvas.getContext("2d");
  if (!ctx) return;
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.clearRect(0, 0, canvas.width, canvas.height);
}

export function renderOverlay(
  canvas: HTMLCanvasElement,
  predictions: YoloPrediction[],
  opts: OverlayRenderOptions = {},
) {
  const ctx = canvas.getContext("2d");
  if (!ctx) return;

  const dpr = opts.dpr ?? window.devicePixelRatio ?? 1;
  const scaleX = opts.scaleX ?? 1;
  const scaleY = opts.scaleY ?? 1;

  // draw in CSS pixels; backing store is scaled by dpr
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  ctx.lineWidth = 2;
  ctx.font =
    "14px ui-sans-serif, system-ui, -apple-system, Segoe UI, Arial, Microsoft JhengHei, sans-serif";

  for (const p of predictions) {
    const [x, y, w, h] = p.bbox;
    const dx = x * scaleX;
    const dy = y * scaleY;
    const dw = w * scaleX;
    const dh = h * scaleY;

    // seg mask (if present)
    if (p.maskCrop && p.maskCrop.inputW > 0 && p.maskCrop.inputH > 0) {
      if (!scratchCanvas) scratchCanvas = document.createElement("canvas");
      if (!scratchCtx) scratchCtx = scratchCanvas.getContext("2d");
      const sctx = scratchCtx;

      const { data, inputW, inputH } = p.maskCrop;
      if (sctx && data && data.length >= inputW * inputH) {
        scratchCanvas.width = inputW;
        scratchCanvas.height = inputH;

        const hue = hashStringToHue(p.classId ?? p.class ?? "0");
        const rgb = hslToRgb(hue, 0.85, 0.55);
        const alpha = 110; // 0..255
        const img = sctx.createImageData(inputW, inputH);
        const out = img.data;
        for (let i = 0, px = 0; i < inputW * inputH; i++, px += 4) {
          if (data[i]) {
            out[px + 0] = rgb.r;
            out[px + 1] = rgb.g;
            out[px + 2] = rgb.b;
            out[px + 3] = alpha;
          } else {
            out[px + 3] = 0;
          }
        }
        sctx.putImageData(img, 0, 0);
        ctx.save();
        ctx.imageSmoothingEnabled = true;
        ctx.drawImage(scratchCanvas, dx, dy, dw, dh);
        ctx.restore();
      }
    }

    // bbox
    ctx.strokeStyle = "rgba(6,182,212,0.95)";
    ctx.fillStyle = "rgba(6,182,212,0.12)";
    ctx.fillRect(dx, dy, dw, dh);
    ctx.strokeRect(dx, dy, dw, dh);

    // label
    const label = `${p.class} ${(p.score * 100).toFixed(1)}%`;
    const paddingX = 6;
    const labelW = Math.min(dw, ctx.measureText(label).width + paddingX * 2);
    const labelX = dx;
    const labelY = Math.max(0, dy - 18);
    ctx.fillStyle = "rgba(2,132,199,0.95)";
    ctx.fillRect(labelX, labelY, labelW, 18);
    ctx.fillStyle = "white";
    ctx.fillText(label, labelX + paddingX, labelY + 13);

    // pose keypoints + skeleton (if present)
    if (p.keypoints?.length) {
      const kps = p.keypoints;

      // skeleton edges
      ctx.save();
      ctx.lineWidth = 2;
      ctx.strokeStyle = "rgba(34,197,94,0.95)"; // green
      for (const [a, b] of COCO17_EDGES) {
        const ka = kps[a];
        const kb = kps[b];
        if (!ka || !kb) continue;
        if (ka.score < KEYPOINT_THRESHOLD || kb.score < KEYPOINT_THRESHOLD)
          continue;
        ctx.beginPath();
        ctx.moveTo(ka.x * scaleX, ka.y * scaleY);
        ctx.lineTo(kb.x * scaleX, kb.y * scaleY);
        ctx.stroke();
      }

      // keypoint dots
      ctx.fillStyle = "rgba(34,197,94,0.95)";
      for (const k of kps) {
        if (k.score < KEYPOINT_THRESHOLD) continue;
        ctx.beginPath();
        ctx.arc(k.x * scaleX, k.y * scaleY, 3.2, 0, Math.PI * 2);
        ctx.fill();
      }
      ctx.restore();
    }
  }
}
