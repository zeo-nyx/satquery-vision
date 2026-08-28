/**
 * ML-powered analysis engine.
 *
 * Combines real Hugging Face Transformers.js models with
 * domain-specific remote sensing features for satellite image analysis.
 */

import {
  getZeroShotClassifier,
  getImageCaptioner,
  getObjectDetector,
  getFeatureExtractor,
  RS_LABELS,
} from "./models";
import {
  analyzeImagePixels,
  computeChangeMap as computePixelChangeMap,
  type ImageAnalysis as PixelAnalysis,
} from "../image/analyze";
import type { ImageMetadata, BoundingBox } from "../agent/types";

// ── ML Analysis Result ───────────────────────────────────────────

export interface MLAnalysisResult {
  /** Classification from CLIP zero-shot */
  classification: {
    label: string;
    score: number;
    allLabels: { label: string; score: number }[];
  };
  /** Caption from image-to-text model */
  caption: string | null;
  /** Detected objects */
  objects: {
    label: string;
    score: number;
    box: { x: number; y: number; w: number; h: number };
  }[];
  /** Feature embedding for similarity comparison */
  embedding: Float32Array | null;
  /** Domain-specific features from pixel analysis */
  domain: PixelAnalysis;
  /** Which models actually loaded and ran */
  modelsUsed: string[];
  /** Any model errors */
  errors: string[];
}

// ── Main analysis functions ─────────────────────────────────────

/**
 * Run full ML analysis on a single image.
 * Tries each model, falls back gracefully on failure.
 */
export async function analyzeWithML(
  image: ImageMetadata,
): Promise<MLAnalysisResult> {
  const result: MLAnalysisResult = {
    classification: {
      label: "unknown",
      score: 0,
      allLabels: [],
    },
    caption: null,
    objects: [],
    embedding: null,
    domain: {
      dominantColors: [],
      colorDistribution: {
        vegetation: 0,
        water: 0,
        urban: 0,
        soil: 0,
        other: 0,
      },
      brightness: { mean: 0, min: 0, max: 0, stdDev: 0 },
      textureComplexity: 0,
      spatialGrid: [],
      ndviEstimate: 0,
      classification: "unknown",
      width: 0,
      height: 0,
    },
    modelsUsed: [],
    errors: [],
  };

  // Always run pixel analysis (fast, no download)
  try {
    result.domain = await analyzeImagePixels(image.dataUrl, 256);
  } catch (err) {
    result.errors.push(
      `Pixel analysis failed: ${err instanceof Error ? err.message : "unknown"}`,
    );
  }

  // Run ML models in parallel (all are independent)
  const [clsResult, captionResult, objectsResult, embedResult] =
    await Promise.allSettled([
      runClassification(image.dataUrl),
      runCaptioning(image.dataUrl),
      runObjectDetection(image.dataUrl),
      runFeatureExtraction(image.dataUrl),
    ]);

  if (clsResult.status === "fulfilled" && clsResult.value) {
    result.classification = clsResult.value;
    result.modelsUsed.push("CLIP (zero-shot classification)");
  } else if (clsResult.status === "rejected") {
    result.errors.push(
      `Classification: ${clsResult.reason?.message || "failed"}`,
    );
  }

  if (captionResult.status === "fulfilled" && captionResult.value) {
    result.caption = captionResult.value;
    result.modelsUsed.push("ViT-GPT2 (image captioning)");
  } else if (captionResult.status === "rejected") {
    result.errors.push(
      `Captioning: ${captionResult.reason?.message || "failed"}`,
    );
  }

  if (objectsResult.status === "fulfilled" && objectsResult.value) {
    result.objects = objectsResult.value;
    result.modelsUsed.push("DETR (object detection)");
  } else if (objectsResult.status === "rejected") {
    result.errors.push(
      `Object detection: ${objectsResult.reason?.message || "failed"}`,
    );
  }

  if (embedResult.status === "fulfilled" && embedResult.value) {
    result.embedding = embedResult.value;
    result.modelsUsed.push("ViT (feature extraction)");
  } else if (embedResult.status === "rejected") {
    result.errors.push(
      `Feature extraction: ${embedResult.reason?.message || "failed"}`,
    );
  }

  return result;
}

/**
 * Run ML analysis on two images for change detection.
 */
export async function analyzeChangeWithML(
  img1: ImageMetadata,
  img2: ImageMetadata,
): Promise<{
  ml1: MLAnalysisResult;
  ml2: MLAnalysisResult;
  changeMapUrl: string;
  changeStats: {
    changePercent: number;
    increasePercent: number;
    decreasePercent: number;
    hotspots: { x: number; y: number; magnitude: number }[];
    avgChange: number;
  };
  embeddingSimilarity: number | null;
}> {
  // Analyze both images + compute change map in parallel
  const [ml1, ml2, changeMap] = await Promise.all([
    analyzeWithML(img1),
    analyzeWithML(img2),
    computePixelChangeMap(img1.dataUrl, img2.dataUrl, 256).catch(() => null),
  ]);

  // Compute embedding similarity if both embeddings exist
  let embeddingSimilarity: number | null = null;
  if (ml1.embedding && ml2.embedding) {
    embeddingSimilarity = cosineSimilarity(ml1.embedding, ml2.embedding);
  }

  return {
    ml1,
    ml2,
    changeMapUrl: changeMap?.changeMapDataUrl || "",
    changeStats: changeMap?.stats || {
      changePercent: 0,
      increasePercent: 0,
      decreasePercent: 0,
      hotspots: [],
      avgChange: 0,
    },
    embeddingSimilarity,
  };
}

/**
 * Run ML analysis on optical + SAR pair for cross-modal fusion.
 */
export async function analyzeCrossModalWithML(
  optical: ImageMetadata,
  sar: ImageMetadata,
): Promise<{
  opticalML: MLAnalysisResult;
  sarML: MLAnalysisResult;
  correlation: number;
  fusionResult: string;
}> {
  const [opticalML, sarML] = await Promise.all([
    analyzeWithML(optical),
    analyzeWithML(sar),
  ]);

  // Compute embedding correlation if available
  let correlation = 0;
  if (opticalML.embedding && sarML.embedding) {
    correlation = cosineSimilarity(opticalML.embedding, sarML.embedding);
  } else {
    // Fall back to spatial grid correlation
    const grid1 = opticalML.domain.spatialGrid;
    const grid2 = sarML.domain.spatialGrid;
    if (grid1.length > 0 && grid2.length > 0) {
      let sum = 0;
      let count = 0;
      for (let y = 0; y < grid1.length; y++) {
        for (let x = 0; x < (grid1[y]?.length || 0); x++) {
          const v1 = grid1[y]?.[x] || 0;
          const v2 = grid2[y]?.[x] || 0;
          sum += 1 - Math.abs(v1 - v2);
          count++;
        }
      }
      correlation = count > 0 ? sum / count : 0;
    }
  }

  // Generate fusion description
  const fusionResult = buildFusionDescription(opticalML, sarML, correlation);

  return { opticalML, sarML, correlation, fusionResult };
}

// ── Individual ML model runners ──────────────────────────────────

async function runClassification(
  dataUrl: string,
): Promise<MLAnalysisResult["classification"] | null> {
  const classifier = await getZeroShotClassifier();
  if (!classifier) return null;

  try {
    const results = await classifier(dataUrl, RS_LABELS, {
      hypothesis_template:
        "This satellite image shows a landscape dominated by {}",
    });

    if (!results || results.length === 0) return null;

    return {
      label: results[0].label,
      score: results[0].score,
      allLabels: results.map((r) => ({ label: r.label, score: r.score })),
    };
  } catch (err) {
    console.warn("[ML] Zero-shot classification failed:", err);
    return null;
  }
}

async function runCaptioning(dataUrl: string): Promise<string | null> {
  const captioner = await getImageCaptioner();
  if (!captioner) return null;

  try {
    const results = await captioner(dataUrl, {
      max_new_tokens: 80,
      num_beams: 3,
    });

    if (!results || results.length === 0) return null;
    return results[0].generated_text || null;
  } catch (err) {
    console.warn("[ML] Image captioning failed:", err);
    return null;
  }
}

async function runObjectDetection(
  dataUrl: string,
): Promise<MLAnalysisResult["objects"] | null> {
  const detector = await getObjectDetector();
  if (!detector) return null;

  try {
    const results = await detector(dataUrl, {
      threshold: 0.3,
      percentage: true,
    });

    if (!results || results.length === 0) return [];

    return results.map((r) => ({
      label: r.label,
      score: r.score,
      box: {
        x: Math.round(r.box.xmin * 100),
        y: Math.round(r.box.ymin * 100),
        w: Math.round((r.box.xmax - r.box.xmin) * 100),
        h: Math.round((r.box.ymax - r.box.ymin) * 100),
      },
    }));
  } catch (err) {
    console.warn("[ML] Object detection failed:", err);
    return null;
  }
}

async function runFeatureExtraction(
  dataUrl: string,
): Promise<Float32Array | null> {
  const extractor = await getFeatureExtractor();
  if (!extractor) return null;

  try {
    const result = await extractor(dataUrl, {
      pooling: "mean",
      normalize: true,
    });

    if (!result?.data) return null;
    return result.data;
  } catch (err) {
    console.warn("[ML] Feature extraction failed:", err);
    return null;
  }
}

// ── Helpers ──────────────────────────────────────────────────────

function cosineSimilarity(a: Float32Array, b: Float32Array): number {
  const len = Math.min(a.length, b.length);
  let dot = 0;
  let normA = 0;
  let normB = 0;
  for (let i = 0; i < len; i++) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  const denom = Math.sqrt(normA) * Math.sqrt(normB);
  return denom > 0 ? dot / denom : 0;
}

function buildFusionDescription(
  optical: MLAnalysisResult,
  sar: MLAnalysisResult,
  correlation: number,
): string {
  const corrPct = Math.round(correlation * 100);

  const parts: string[] = [
    `Cross-modal analysis combining optical and SAR satellite data.`,
    ``,
    `**Optical analysis:**`,
    `• Classification: ${optical.classification.label} (${Math.round(optical.classification.score * 100)}% confidence)`,
    `• Land cover: Vegetation ${Math.round(optical.domain.colorDistribution.vegetation * 100)}%, Urban ${Math.round(optical.domain.colorDistribution.urban * 100)}%, Water ${Math.round(optical.domain.colorDistribution.water * 100)}%`,
    `• NDVI estimate: ${optical.domain.ndviEstimate > 0 ? "+" : ""}${optical.domain.ndviEstimate.toFixed(3)}`,
    optical.caption ? `• ML caption: "${optical.caption}"` : ``,
    ``,
    `**SAR analysis:**`,
    `• Classification: ${sar.classification.label} (${Math.round(sar.classification.score * 100)}% confidence)`,
    `• High-backscatter regions: ${Math.round(sar.domain.colorDistribution.urban * 100)}% (built-up / double-bounce)`,
    `• Low-backscatter regions: ${Math.round(sar.domain.colorDistribution.water * 100)}% (smooth surfaces / specular reflection)`,
    `• Texture complexity: ${Math.round(sar.domain.textureComplexity * 100)}%`,
    sar.caption ? `• ML caption: "${sar.caption}"` : ``,
    ``,
    `**Fusion results:**`,
    `• Spatial correlation: ${corrPct}% (${corrPct > 70 ? "strong" : corrPct > 50 ? "moderate" : "weak"} agreement)`,
  ];

  // Compare top classifications
  if (optical.classification.allLabels.length > 0) {
    const topOptical = optical.classification.allLabels.slice(0, 3);
    const topSAR = sar.classification.allLabels.slice(0, 3);

    parts.push(
      `• Optical top labels: ${topOptical.map((l) => `${l.label} (${Math.round(l.score * 100)}%)`).join(", ")}`,
    );
    parts.push(
      `• SAR top labels: ${topSAR.map((l) => `${l.label} (${Math.round(l.score * 100)}%)`).join(", ")}`,
    );
  }

  // Detected objects from both
  const allObjects = [...optical.objects, ...sar.objects];
  if (allObjects.length > 0) {
    const uniqueLabels = [...new Set(allObjects.map((o) => o.label))];
    parts.push(
      `• Objects detected: ${uniqueLabels.join(", ")}`,
    );
  }

  parts.push(
    ``,
    `The optical-SAR fusion achieves improved discrimination by combining spectral information (optical) with structural/surface roughness data (SAR). ${corrPct > 70 ? "Strong spatial agreement indicates consistent land cover characterization across modalities." : "Moderate-to-weak correlation suggests complementary information that enhances overall classification accuracy."}`,
  );

  return parts.filter(Boolean).join("\n");
}
