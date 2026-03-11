import type { YoloPrediction } from "../yolo/types";
import { COCO17_EDGES, KEYPOINT_THRESHOLD } from "../yolo/yoloCore";

export type OverlayRenderOptions = {
  dpr?: number;
  scaleX?: number;
  scaleY?: number;
};

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
