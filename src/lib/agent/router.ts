import type { ImageMetadata, TaskPlan, TaskType, InputType } from "./types";

/**
 * Rule-based query router — classifies user queries and selects
 * the appropriate model pipeline based on input images and query text.
 */
export function routeQuery(
  query: string,
  images: ImageMetadata[],
): TaskPlan {
  const q = query.toLowerCase().trim();
  const numImages = images.length;

  // ── Single image ────────────────────────────────────────────────
  if (numImages === 1) {
    return routeSingleImage(q, images[0]);
  }

  // ── Two images ──────────────────────────────────────────────────
  if (numImages === 2) {
    return routeTwoImages(q, images[0], images[1]);
  }

  // ── Fallback ────────────────────────────────────────────────────
  return {
    task: "vqa",
    inputType: "single_image",
    models: ["satellite-vqa-base"],
    description: "Default VQA analysis",
  };
}

function routeSingleImage(query: string, image: ImageMetadata): TaskPlan {
  // Captioning keywords
  if (
    query.includes("describe") ||
    query.includes("what do you see") ||
    query.includes("what is in") ||
    query.includes("tell me about") ||
    query.includes("summarize") ||
    query.includes("overview") ||
    query.includes("caption")
  ) {
    return {
      task: "captioning",
      inputType: "single_image",
      models: ["rs-captioner"],
      description: "Generating scene description",
    };
  }

  // Grounding / localization keywords
  if (
    query.includes("where") ||
    query.includes("locate") ||
    query.includes("highlight") ||
    query.includes("show me") ||
    query.includes("find") ||
    query.includes("point out") ||
    query.includes("bounding box")
  ) {
    return {
      task: "grounding",
      inputType: "single_image",
      models: ["rs-grounding-model"],
      description: "Localizing regions of interest",
    };
  }

  // Default: VQA
  return {
    task: "vqa",
    inputType: "single_image",
    models: ["satellite-vqa-base"],
    description: "Answering visual question about satellite image",
  };
}

function routeTwoImages(
  query: string,
  img1: ImageMetadata,
  img2: ImageMetadata,
): TaskPlan {
  const hasOptical = (m: ImageMetadata) =>
    m.modality === "optical" || m.bands >= 3;
  const hasSAR = (m: ImageMetadata) => m.modality === "sar";

  // Cross-modal: optical + SAR
  if ((hasOptical(img1) && hasSAR(img2)) || (hasSAR(img1) && hasOptical(img2))) {
    return {
      task: "cross_modal_analysis",
      inputType: "optical_sar_pair",
      models: ["optical-sar-fusion", "rs-vqa-fusion"],
      description: "Joint optical-SAR analysis",
    };
  }

  // Bi-temporal: different dates
  if (img1.acquisitionDate && img2.acquisitionDate) {
    if (img1.acquisitionDate !== img2.acquisitionDate) {
      return {
        task: "change_analysis",
        inputType: "bi_temporal",
        models: ["change-detection", "change-vqa"],
        description: "Bi-temporal change analysis",
      };
    }
  }

  // Check for change-related keywords even without date difference
  if (
    query.includes("change") ||
    query.includes("difference") ||
    query.includes("before") ||
    query.includes("after") ||
    query.includes("compare") ||
    query.includes("increased") ||
    query.includes("decreased") ||
    query.includes("shifted") ||
    query.includes("moved")
  ) {
    return {
      task: "change_analysis",
      inputType: "bi_temporal",
      models: ["change-detection", "change-vqa"],
      description: "Bi-temporal change analysis",
    };
  }

  // Default for two images: change analysis
  return {
    task: "change_analysis",
    inputType: "bi_temporal",
    models: ["change-detection", "change-vqa"],
    description: "Comparative analysis of two satellite images",
  };
}
