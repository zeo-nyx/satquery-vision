/**
 * Evaluation & Benchmark Infrastructure for SatQuery AI
 *
 * Provides tools for evaluating against prescribed benchmarks:
 * - VRSBench: remote-sensing image captioning and grounding
 * - RSVQA: remote-sensing visual question answering
 * - CDVQA: change-based visual question answering
 * - BigEarthNet: domain adaptation evaluation
 *
 * This module defines evaluation protocols, metrics, and
 * result formatting for judge/reviewer assessment.
 */

// ── Benchmark Definitions ────────────────────────────────────────

export interface BenchmarkConfig {
  name: string;
  description: string;
  taskType: "captioning" | "grounding" | "vqa" | "change_vqa" | "cross_modal";
  metrics: string[];
  datasetUrl: string;
  splitInfo: string;
}

export const BENCHMARKS: Record<string, BenchmarkConfig> = {
  vrsbench: {
    name: "VRSBench",
    description:
      "Remote-sensing vision-language benchmark for captioning and grounding evaluation",
    taskType: "captioning",
    metrics: ["BLEU-4", "METEOR", "CIDEr", "SPICE", "IoU"],
    datasetUrl: "https://github.com/VIPL-Fresh/VRSBench",
    splitInfo: "Test split as prescribed by evaluation committee",
  },
  rsvqa: {
    name: "RSVQA",
    description:
      "Remote-sensing visual question answering benchmark",
    taskType: "vqa",
    metrics: ["Accuracy", "F1 Score", "Yes/No Accuracy"],
    datasetUrl: "https://rsvqa.thu.ai/",
    splitInfo: "Prescribed test subset",
  },
  cdvqa: {
    name: "CDVQA",
    description:
      "Change-based visual question answering from bi-temporal imagery",
    taskType: "change_vqa",
    metrics: ["Accuracy", "F1 Score", "Change Detection IoU"],
    datasetUrl: "https://github.com/whirc-d/CDVQA",
    splitInfo: "Prescribed test subset with reference answers",
  },
  bigearthnet: {
    name: "BigEarthNet",
    description:
      "Primary dataset for remote-sensing domain adaptation with Sentinel-1 SAR and Sentinel-2 multispectral data",
    taskType: "cross_modal",
    metrics: [
      "mAP",
      "F1 per class",
      "Domain adaptation accuracy",
      "Label transfer rate",
    ],
    datasetUrl: "https://arxiv.org/abs/2603.29630",
    splitInfo: "Prescribed training and evaluation splits",
  },
  isro_sac: {
    name: "ISRO/SAC Evaluation",
    description:
      "Pre-georeferenced Cartosat-2S optical and RISAT SAR image pairs with task-specific annotations",
    taskType: "cross_modal",
    metrics: [
      "Task-specific accuracy",
      "Change detection IoU",
      "Caption quality (human + automated)",
      "Bounding box mAP",
    ],
    datasetUrl: "Provided by evaluation committee",
    splitInfo: "Not disclosed — final evaluation set",
  },
};

// ── Evaluation Metrics ───────────────────────────────────────────

export interface EvaluationResult {
  benchmark: string;
  taskId: string;
  metrics: Record<string, number>;
  sampleCount: number;
  timestamp: number;
  modelConfig: string;
  notes: string[];
}

/**
 * Compute VQA accuracy metrics.
 */
export function computeVQAMetrics(
  predictions: string[],
  groundTruth: string[],
): {
  accuracy: number;
  yesNoAccuracy: number;
  f1Score: number;
} {
  let correct = 0;
  let yesNoCorrect = 0;
  let yesNoTotal = 0;
  let tp = 0;
  let fp = 0;
  let fn = 0;

  for (let i = 0; i < predictions.length; i++) {
    const pred = normalizeAnswer(predictions[i]);
    const gt = normalizeAnswer(groundTruth[i]);

    // Exact match accuracy
    if (pred === gt) correct++;

    // Yes/No accuracy
    const isYesNo =
      gt === "yes" || gt === "no" || gt === "true" || gt === "false";
    if (isYesNo) {
      yesNoTotal++;
      if (pred === gt) yesNoCorrect++;
    }

    // F1 for binary answers
    if (pred === "yes" && gt === "yes") tp++;
    else if (pred === "yes" && gt === "no") fp++;
    else if (pred === "no" && gt === "yes") fn++;
  }

  const precision = tp + fp > 0 ? tp / (tp + fp) : 0;
  const recall = tp + fn > 0 ? tp / (tp + fn) : 0;
  const f1 = precision + recall > 0 ? (2 * precision * recall) / (precision + recall) : 0;

  return {
    accuracy: predictions.length > 0 ? correct / predictions.length : 0,
    yesNoAccuracy: yesNoTotal > 0 ? yesNoCorrect / yesNoTotal : 0,
    f1Score: f1,
  };
}

/**
 * Compute caption quality metrics (simplified BLEU-1).
 */
export function computeCaptionMetrics(
  predictions: string[],
  groundTruth: string[],
): {
  bleu1: number;
  averageLength: number;
  uniqueTokenRatio: number;
} {
  let totalBleu1 = 0;
  let totalLength = 0;
  const allTokens = new Set<string>();

  for (let i = 0; i < predictions.length; i++) {
    const predTokens = tokenize(predictions[i]);
    const gtTokens = new Set(tokenize(groundTruth[i]));

    // BLEU-1 (unigram precision)
    let matches = 0;
    for (const t of predTokens) {
      if (gtTokens.has(t)) matches++;
      allTokens.add(t);
    }
    totalBleu1 += predTokens.length > 0 ? matches / predTokens.length : 0;
    totalLength += predTokens.length;
  }

  return {
    bleu1: predictions.length > 0 ? totalBleu1 / predictions.length : 0,
    averageLength: predictions.length > 0 ? totalLength / predictions.length : 0,
    uniqueTokenRatio: allTokens.size,
  };
}

/**
 * Compute change detection IoU (Intersection over Union).
 */
export function computeChangeIoU(
  predictedMask: boolean[],
  groundTruthMask: boolean[],
): {
  iou: number;
  precision: number;
  recall: number;
  f1: number;
} {
  let tp = 0;
  let fp = 0;
  let fn = 0;
  let tn = 0;

  for (let i = 0; i < Math.min(predictedMask.length, groundTruthMask.length); i++) {
    if (predictedMask[i] && groundTruthMask[i]) tp++;
    else if (predictedMask[i] && !groundTruthMask[i]) fp++;
    else if (!predictedMask[i] && groundTruthMask[i]) fn++;
    else tn++;
  }

  const iou = tp + fp + fn > 0 ? tp / (tp + fp + fn) : 0;
  const precision = tp + fp > 0 ? tp / (tp + fp) : 0;
  const recall = tp + fn > 0 ? tp / (tp + fn) : 0;
  const f1 = precision + recall > 0 ? (2 * precision * recall) / (precision + recall) : 0;

  return { iou, precision, recall, f1 };
}

// ── Helpers ──────────────────────────────────────────────────────

function normalizeAnswer(answer: string): string {
  return answer
    .toLowerCase()
    .trim()
    .replace(/[^\w\s]/g, "")
    .replace(/\s+/g, " ");
}

function tokenize(text: string): string[] {
  return normalizeAnswer(text)
    .split(" ")
    .filter((t) => t.length > 0);
}

/**
 * Generate an evaluation summary report for judges.
 */
export function generateEvaluationReport(
  results: EvaluationResult[],
): string {
  const lines: string[] = [
    "=".repeat(70),
    "SATQUERY AI — EVALUATION REPORT",
    "=".repeat(70),
    "",
    `Generated: ${new Date().toISOString()}`,
    `Benchmarks evaluated: ${results.length}`,
    "",
  ];

  for (const r of results) {
    lines.push("-".repeat(70));
    lines.push(`Benchmark: ${r.benchmark}`);
    lines.push(`Task: ${r.taskId}`);
    lines.push(`Samples: ${r.sampleCount}`);
    lines.push(`Model: ${r.modelConfig}`);
    lines.push("");
    lines.push("Metrics:");
    for (const [key, val] of Object.entries(r.metrics)) {
      lines.push(`  ${key}: ${(val * 100).toFixed(2)}%`);
    }
    if (r.notes.length > 0) {
      lines.push("");
      lines.push("Notes:");
      for (const note of r.notes) {
        lines.push(`  • ${note}`);
      }
    }
    lines.push("");
  }

  lines.push("=".repeat(70));
  lines.push("End of Evaluation Report");
  lines.push("=".repeat(70));

  return lines.join("\n");
}
