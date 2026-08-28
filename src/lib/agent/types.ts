// ── Image metadata ──────────────────────────────────────────────────
export type ImageModality = "optical" | "sar" | "unknown";

export interface ImageMetadata {
  id: string;
  fileName: string;
  modality: ImageModality;
  width: number;
  height: number;
  bands: number;
  bandNames: string[];
  acquisitionDate: string | null;
  crs: string | null;
  pixelSize: { x: number; y: number } | null;
  sensor: string | null;
  fileSize: number;
  format: string;
  dataUrl: string; // base64 or object URL for display
}

// ── Agent task types ───────────────────────────────────────────────
export type TaskType =
  | "vqa"
  | "captioning"
  | "grounding"
  | "change_analysis"
  | "cross_modal_analysis"
  | "text_context";

export type InputType =
  | "single_image"
  | "bi_temporal"
  | "optical_sar_pair"
  | "text_only";

export interface TaskPlan {
  task: TaskType;
  inputType: InputType;
  models: string[];
  description: string;
}

// ── Analysis results ───────────────────────────────────────────────
export interface AnalysisResult {
  task: TaskType;
  inputType: InputType;
  modelsUsed: string[];
  answer: string;
  confidence: number;
  changeMap?: string; // data URL of change map visualization
  boundingBoxes?: BoundingBox[];
  executionTrace: ExecutionStep[];
  timestamp: number;
}

export interface BoundingBox {
  x: number;
  y: number;
  width: number;
  height: number;
  label: string;
  confidence: number;
  color: string;
}

export interface ExecutionStep {
  step: string;
  status: "completed" | "processing" | "pending" | "error";
  detail?: string;
  duration?: number;
}

// ── Analysis session ───────────────────────────────────────────────
export interface AnalysisSession {
  id: string;
  images: ImageMetadata[];
  query: string;
  result: AnalysisResult | null;
  status: "idle" | "routing" | "processing" | "complete" | "error";
}
