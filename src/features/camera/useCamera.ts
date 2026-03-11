import { useCallback, useEffect, useMemo, useRef, useState } from "react";

type CameraState =
  | { status: "idle" }
  | { status: "starting" }
  | { status: "running"; stream: MediaStream }
  | { status: "error"; message: string };

export function useCamera(videoRef: React.RefObject<HTMLVideoElement | null>) {
  const [state, setState] = useState<CameraState>({ status: "idle" });
  const [capability, setCapability] = useState<{
    status: "checking" | "ready" | "unavailable";
    message?: string;
  }>({ status: "checking" });
  const streamRef = useRef<MediaStream | null>(null);

  const stop = useCallback(() => {
    const s = streamRef.current;
    if (s) {
      for (const t of s.getTracks()) t.stop();
    }
    streamRef.current = null;
    setState({ status: "idle" });
  }, []);

  const start = useCallback(async () => {
    try {
      setState({ status: "starting" });
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "environment" },
        audio: false,
      });
      streamRef.current = stream;
      const video = videoRef.current;
      if (video) {
        video.srcObject = stream;
        await video.play().catch(() => {});
      }
      setState({ status: "running", stream });
    } catch (err) {
      setState({
        status: "error",
        message: err instanceof Error ? err.message : "無法開啟相機",
      });
    }
  }, [videoRef]);

  useEffect(() => stop, [stop]);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      const mediaDevices = navigator.mediaDevices;
      if (!mediaDevices?.getUserMedia) {
        if (!cancelled) {
          setCapability({
            status: "unavailable",
            message: "此裝置或瀏覽器不支援相機 API",
          });
        }
        return;
      }

      try {
        const devices = await mediaDevices.enumerateDevices?.();
        const hasVideoInput = !!devices?.some((d) => d.kind === "videoinput");
        if (!hasVideoInput) {
          if (!cancelled) {
            setCapability({
              status: "unavailable",
              message: "找不到可用的相機裝置",
            });
          }
          return;
        }
      } catch {
        // Some environments can throw here; don't block camera button solely on this.
      }

      if (!cancelled) setCapability({ status: "ready" });
    })();

    return () => {
      cancelled = true;
    };
  }, []);

  const api = useMemo(() => {
    return { start, stop };
  }, [start, stop]);

  return { state, api, capability };
}
