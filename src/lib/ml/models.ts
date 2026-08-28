/**
 * ML Model Manager — lazily loads and caches Transformers.js pipelines.
 *
 * Models used:
 *   - image-classification:  Xenova/vit-base-patch16-224
 *   - zero-shot-classification:  Xenova/clip-vit-base-patch32
 *   - image-to-text:  Xenova/vit-gpt2-image-captioning
 *   - object-detection:  Xenova/detr-resnet-50
 *   - feature-extraction:  Xenova/vit-base-patch16-224
 *
 * Each pipeline is loaded on first use and reused for subsequent calls.
 */

import { pipeline, type PipelineType } from "@huggingface/transformers";

// ── Model registry ───────────────────────────────────────────────

const MODEL_IDS: Record<string, string> = {
  "image-classification": "Xenova/vit-base-patch16-224",
  "zero-shot-image-classification": "Xenova/clip-vit-base-patch32",
  "image-to-text": "Xenova/vit-gpt2-image-captioning",
  "object-detection": "Xenova/detr-resnet-50",
  "feature-extraction": "Xenova/vit-base-patch16-224",
};

// Remote-sensing-specific labels for zero-shot classification
export const RS_LABELS = [
  "urban area with buildings",
  "dense forest and vegetation",
  "water body such as river or lake",
  "agricultural land with crops",
  "bare soil and exposed terrain",
  "road and transportation infrastructure",
  "industrial zone",
  "residential neighborhood",
  "wetland and marsh",
  "snow or ice cover",
  "desert and sand dunes",
  "cloud cover",
];

// Land cover labels for fine-grained classification
export const LAND_COVER_LABELS = [
  "built-up urban area",
  "dense vegetation forest",
  "sparse vegetation shrubland",
  "cropland agriculture",
  "water permanent",
  "water seasonal",
  "bare rock soil",
  "sand desert",
  "snow ice",
  "wetland",
  "mangrove",
  "impervious surface road",
];

// ── Pipeline cache ───────────────────────────────────────────────

const pipelineCache = new Map<string, unknown>();
const loadingPromises = new Map<string, Promise<unknown>>();

export type PipelineTask =
  | "image-classification"
  | "zero-shot-image-classification"
  | "image-to-text"
  | "object-detection"
  | "feature-extraction";

/**
 * Get a Transformers.js pipeline, loading it on first use.
 * Returns null if loading fails (network error, model too large, etc.)
 */
export async function getPipeline<T>(
  task: PipelineTask,
): Promise<T | null> {
  const cacheKey = task;

  // Return cached if available
  if (pipelineCache.has(cacheKey)) {
    return pipelineCache.get(cacheKey) as T;
  }

  // Return existing loading promise if in progress
  if (loadingPromises.has(cacheKey)) {
    return loadingPromises.get(cacheKey) as Promise<T | null>;
  }

  // Start loading
  const modelId = MODEL_IDS[task];
  if (!modelId) {
    console.warn(`[ML] Unknown task: ${task}`);
    return null;
  }

  const loadPromise = loadPipeline<T>(task, modelId);
  loadingPromises.set(cacheKey, loadPromise as Promise<unknown>);

  return loadPromise;
}

async function loadPipeline<T>(
  task: string,
  modelId: string,
): Promise<T | null> {
  try {
    console.log(`[ML] Loading pipeline: ${task} (${modelId})...`);

    const p = await pipeline(task as PipelineType, modelId, {
      progress_callback: (progress: { status?: string; file?: string; progress?: number }) => {
        if (progress.status === "progress") {
          const pct = progress.progress ?? 0;
          if (pct % 20 === 0 || pct === 100) {
            console.log(
              `[ML] ${task}: ${progress.file} — ${pct}%`,
            );
          }
        }
      },
    });

    pipelineCache.set(task, p as unknown);
    console.log(`[ML] Pipeline loaded: ${task}`);
    return p as T;
  } catch (err) {
    console.error(`[ML] Failed to load pipeline ${task}:`, err);
    loadingPromises.delete(task);
    return null;
  }
}

/**
 * Check which pipelines are currently loaded.
 */
export function getLoadedPipelines(): string[] {
  return Array.from(pipelineCache.keys());
}

/**
 * Dispose all cached pipelines to free memory.
 */
export function disposeAllPipelines(): void {
  for (const [key, p] of pipelineCache) {
    if (p && typeof p === "object" && "dispose" in p) {
      (p as { dispose: () => void }).dispose();
    }
  }
  pipelineCache.clear();
  loadingPromises.clear();
}

// ── Typed pipeline accessors ─────────────────────────────────────

export type ImageClassificationPipeline = (
  image: string | URL | Blob,
  options?: { top_k?: number },
) => Promise<{ label: string; score: number }[]>;

export type ZeroShotImageClassificationPipeline = (
  image: string | URL | Blob,
  labels: string[],
  options?: { hypothesis_template?: string },
) => Promise<{ label: string; score: number }[]>;

export type ImageToTextPipeline = (
  image: string | URL | Blob,
  options?: { max_new_tokens?: number; num_beams?: number },
) => Promise<{ generated_text: string }[]>;

export type ObjectDetectionPipeline = (
  image: string | URL | Blob,
  options?: { threshold?: number; percentage?: boolean },
) => Promise<{ label: string; score: number; box: { xmin: number; ymin: number; xmax: number; ymax: number } }[]>;

export type FeatureExtractionPipeline = (
  image: string | URL | Blob,
  options?: { pooling?: string; normalize?: boolean },
) => Promise<{ data: Float32Array }>;

export async function getImageClassifier() {
  return getPipeline<ImageClassificationPipeline>("image-classification");
}

export async function getZeroShotClassifier() {
  return getPipeline<ZeroShotImageClassificationPipeline>(
    "zero-shot-image-classification",
  );
}

export async function getImageCaptioner() {
  return getPipeline<ImageToTextPipeline>("image-to-text");
}

export async function getObjectDetector() {
  return getPipeline<ObjectDetectionPipeline>("object-detection");
}

export async function getFeatureExtractor() {
  return getPipeline<FeatureExtractionPipeline>("feature-extraction");
}
