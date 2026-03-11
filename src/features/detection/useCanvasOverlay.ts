import { useEffect, useRef } from "react";

export function useCanvasOverlay(
  canvasRef: React.RefObject<HTMLCanvasElement | null>,
  targetRef: React.RefObject<HTMLImageElement | HTMLVideoElement | null>,
) {
  const dprRef = useRef(1);

  useEffect(() => {
    const canvas = canvasRef.current;
    const target = targetRef.current;
    if (!canvas || !target) return;

    const sync = () => {
      const el = target as HTMLVideoElement & HTMLImageElement;
      const intrinsicWidth =
        el.videoWidth || el.naturalWidth || el.width || el.clientWidth;
      const intrinsicHeight =
        el.videoHeight || el.naturalHeight || el.height || el.clientHeight;
      if (!intrinsicWidth || !intrinsicHeight) return;

      const rect = el.getBoundingClientRect?.();
      const displayWidth = rect?.width
        ? Math.max(1, rect.width)
        : intrinsicWidth;
      const displayHeight = rect?.height
        ? Math.max(1, rect.height)
        : intrinsicHeight;

      const dpr = window.devicePixelRatio || 1;
      dprRef.current = dpr;

      canvas.width = Math.round(displayWidth * dpr);
      canvas.height = Math.round(displayHeight * dpr);
      canvas.style.width = `${displayWidth}px`;
      canvas.style.height = `${displayHeight}px`;
    };

    sync();

    const ro = new ResizeObserver(() => sync());
    try {
      ro.observe(target);
    } catch {
      // ignore
    }

    window.addEventListener("resize", sync, { passive: true });

    return () => {
      window.removeEventListener("resize", sync);
      ro.disconnect();
    };
  }, [canvasRef, targetRef]);

  function getDrawScale() {
    const target = targetRef.current;
    if (!target) return { scaleX: 1, scaleY: 1 };
    const el = target as HTMLVideoElement & HTMLImageElement;
    const intrinsicWidth =
      el.videoWidth || el.naturalWidth || el.width || el.clientWidth;
    const intrinsicHeight =
      el.videoHeight || el.naturalHeight || el.height || el.clientHeight;
    const cssWidth =
      Number.parseFloat(canvasRef.current?.style.width || "") || intrinsicWidth;
    const cssHeight =
      Number.parseFloat(canvasRef.current?.style.height || "") ||
      intrinsicHeight;
    if (!intrinsicWidth || !intrinsicHeight || !cssWidth || !cssHeight)
      return { scaleX: 1, scaleY: 1 };
    return {
      scaleX: cssWidth / intrinsicWidth,
      scaleY: cssHeight / intrinsicHeight,
    };
  }

  return { dprRef, getDrawScale };
}
