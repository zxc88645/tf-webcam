export type ModelKey = "pose" | "detect" | "seg";

export type ModelConfig = {
  id: ModelKey;
  label: string;
  url: string;
  inputSize: number;
};

function baseUrl(path: string) {
  return `${import.meta.env.BASE_URL}${path}`.replace(/\/{2,}/g, "/");
}

export const MODEL_CONFIGS: Record<ModelKey, ModelConfig> = {
  pose: {
    id: "pose",
    label: "YOLOv26n Pose",
    url: baseUrl("models/yolo26n-pose/model.json"),
    inputSize: 640,
  },
  detect: {
    id: "detect",
    label: "YOLOv26n Detect",
    url: baseUrl("models/yolo26n-detect/model.json"),
    inputSize: 640,
  },
  seg: {
    id: "seg",
    label: "YOLOv26n Seg",
    url: baseUrl("models/yolo26n-seg/model.json"),
    inputSize: 640,
  },
};
