import { useCallback, useEffect, useMemo, useRef, useState } from "react";

type CameraState =
  | { status: "idle" }
  | { status: "starting" }
  | { status: "running"; stream: MediaStream }
  | { status: "error"; message: string };

export function useCamera(videoRef: React.RefObject<HTMLVideoElement | null>) {
  const [state, setState] = useState<CameraState>({ status: "idle" });
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

  const api = useMemo(() => {
    return { start, stop };
  }, [start, stop]);

  return { state, api };
}
