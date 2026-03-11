export type YoloPrediction = {
  bbox: [number, number, number, number];
  score: number;
  classId: string;
  class: string;
  // pose
  keypoints?: Array<{ x: number; y: number; score: number }>;
  // seg
  maskCrop?: { data: Uint8Array; inputW: number; inputH: number };
};

export type ModelConfig = {
  id: string;
  label: string;
  url: string;
  inputSize: number;
};
