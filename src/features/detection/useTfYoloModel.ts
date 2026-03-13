import { useEffect, useMemo, useRef, useState } from "react";
import type { GraphModel } from "@tensorflow/tfjs-converter";
import * as tf from "@tensorflow/tfjs";
import "@tensorflow/tfjs-converter";

import type { ModelConfig } from "./modelConfigs";
import { loadClassNames } from "../../lib/yolo/yoloCore";
import type { ModelConfig as CoreModelConfig } from "../../lib/yolo/types";

type ModelState =
  | { status: "idle" }
  | { status: "loading"; message: string; progress?: number }
  | { status: "ready"; model: GraphModel; classNames: string[] | null }
  | { status: "error"; message: string };

function toCoreConfig(cfg: ModelConfig): CoreModelConfig {
  return {
    id: cfg.id,
    label: cfg.label,
    url: cfg.url,
    inputSize: cfg.inputSize,
  };
}

export function useTfYoloModel(activeModelConfig: ModelConfig) {
  const [state, setState] = useState<ModelState>({ status: "idle" });
  const [tfBackendReady, setTfBackendReady] = useState(false);
  const modelRef = useRef<GraphModel | null>(null);
  const classNamesRef = useRef<string[] | null>(null);
  const loadSeq = useRef(0);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        setState({ status: "loading", message: "初始化 TensorFlow.js…" });
        await tf.ready();
        if (cancelled) return;
        setTfBackendReady(true);
      } catch {
        if (cancelled) return;
        setState({ status: "error", message: "TensorFlow.js 初始化失敗" });
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    if (!tfBackendReady) return;
    const seq = ++loadSeq.current;
    setState({
      status: "loading",
      message: `載入模型：${activeModelConfig.label}（準備中）…`,
    });

    let cancelled = false;
    (async () => {
      try {
        modelRef.current?.dispose?.();
        modelRef.current = null;
        classNamesRef.current = null;

        setState({
          status: "loading",
          message: `載入模型：${activeModelConfig.label}（下載/解析）…`,
          progress: 0,
        });
        const model = (await tf.loadGraphModel(activeModelConfig.url, {
          fromTFHub: false,
          onProgress: (fraction: number) => {
            setState({
              status: "loading",
              message: `載入模型：${activeModelConfig.label}（下載/解析）… ${Math.round(fraction * 100)}%`,
              progress: fraction,
            });
          },
        })) as GraphModel;

        setState({
          status: "loading",
          message: `載入模型：${activeModelConfig.label}（載入標籤）…`,
          progress: 0.9,
        });
        const names = await loadClassNames(toCoreConfig(activeModelConfig));

        // warmup (best-effort)
        try {
          setState({
            status: "loading",
            message: `載入模型：${activeModelConfig.label}（warmup）…`,
            progress: 0.95,
          });
          tf.tidy(() => {
            const x = tf.zeros(
              [1, activeModelConfig.inputSize, activeModelConfig.inputSize, 3],
              "float32",
            );
            // GraphModel supports execute; predict sometimes too.
            (
              model as unknown as { execute: (t: tf.Tensor) => unknown }
            ).execute(x);
          });
        } catch {
          // ignore warmup failures
        }

        if (cancelled || loadSeq.current !== seq) return;
        modelRef.current = model;
        classNamesRef.current = names;
        setState({ status: "ready", model, classNames: names });
      } catch (err) {
        if (cancelled || loadSeq.current !== seq) return;
        setState({
          status: "error",
          message: err instanceof Error ? err.message : "模型載入失敗",
        });
      }
    })();

    return () => {
      cancelled = true;
    };
  }, [activeModelConfig, tfBackendReady]);

  const api = useMemo(() => {
    if (state.status !== "ready") return null;
    return {
      model: state.model,
      classNames: state.classNames,
    };
  }, [state]);

  return { state, api };
}
