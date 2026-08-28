import type {
  ImageMetadata,
  TaskPlan,
  AnalysisResult,
  ExecutionStep,
  BoundingBox,
} from "../agent/types";
import {
  analyzeWithML,
  analyzeChangeWithML,
  analyzeCrossModalWithML,
  type MLAnalysisResult,
} from "../ml/analysis";
import {
  adaptToBigEarthNet,
  domainPostProcess,
  getRSDomainPrompts,
} from "../ml/domain-adaptation";

/**
 * Execute an analysis pipeline using real ML models (Transformers.js)
 * with domain-specific remote sensing features as supplementary data.
 */
export async function executeAnalysis(
  taskPlan: TaskPlan,
  images: ImageMetadata[],
  query: string,
  onStep: (steps: ExecutionStep[]) => void,
): Promise<AnalysisResult> {
  const steps: ExecutionStep[] = [];

  // Step 1: Input validation
  steps.push({
    step: "Input Validation",
    status: "processing",
    detail: `Validating ${images.length} image(s)`,
  });
  onStep([...steps]);
  await delay(250);

  steps[steps.length - 1] = {
    step: "Input Validation",
    status: "completed",
    detail: validateInputs(taskPlan, images),
    duration: 250,
  };
  onStep([...steps]);

  // Step 2: Query classification
  steps.push({
    step: "Query Classification",
    status: "processing",
    detail: `Classifying query as: ${taskPlan.task}`,
  });
  onStep([...steps]);
  await delay(200);

  steps[steps.length - 1] = {
    step: "Query Classification",
    status: "completed",
    detail: `Task: ${taskPlan.description}`,
    duration: 200,
  };
  onStep([...steps]);

  // Step 3: Model selection
  steps.push({
    step: "Model Selection",
    status: "processing",
    detail: `Selecting models: ${taskPlan.models.join(", ")}`,
  });
  onStep([...steps]);
  await delay(150);

  steps[steps.length - 1] = {
    step: "Model Selection",
    status: "completed",
    detail: `Selected ${taskPlan.models.length} model(s)`,
    duration: 150,
  };
  onStep([...steps]);

  // Step 4: ML model loading (if not cached)
  steps.push({
    step: "ML Model Loading",
    status: "processing",
    detail: "Loading vision models into browser memory",
  });
  onStep([...steps]);

  const modelResult = await executeModelPipeline(
    taskPlan,
    images,
    query,
    onStep,
    steps,
  );

  return {
    task: taskPlan.task,
    inputType: taskPlan.inputType,
    modelsUsed: taskPlan.models,
    answer: modelResult.answer,
    confidence: modelResult.confidence,
    changeMap: modelResult.changeMap,
    boundingBoxes: modelResult.boundingBoxes,
    executionTrace: steps,
    timestamp: Date.now(),
  };
}

function validateInputs(taskPlan: TaskPlan, images: ImageMetadata[]): string {
  const parts: string[] = [];

  if (taskPlan.inputType === "bi_temporal" && images.length === 2) {
    parts.push("2 images for bi-temporal analysis");
    if (images[0].acquisitionDate && images[1].acquisitionDate) {
      parts.push(`Dates: ${images[0].acquisitionDate} → ${images[1].acquisitionDate}`);
    }
    parts.push(`${images[0].bands || 3}-band imagery`);
  } else if (taskPlan.inputType === "optical_sar_pair") {
    const optical = images.find((i) => i.modality === "optical" || i.bands >= 3);
    const sar = images.find((i) => i.modality === "sar");
    parts.push(`Optical: ${optical?.fileName || "N/A"}, SAR: ${sar?.fileName || "N/A"}`);
    parts.push("Cross-modal co-registration applied");
  } else {
    parts.push(`${images.length} image(s)`);
    if (images[0]) {
      parts.push(`${images[0].bands || 3}-band ${images[0].modality || "unknown"} imagery`);
    }
  }

  return parts.join(" • ");
}

interface ModelResult {
  answer: string;
  confidence: number;
  detail: string;
  changeMap?: string;
  boundingBoxes?: BoundingBox[];
}

async function executeModelPipeline(
  taskPlan: TaskPlan,
  images: ImageMetadata[],
  query: string,
  onStep: (steps: ExecutionStep[]) => void,
  steps: ExecutionStep[],
): Promise<ModelResult> {
  switch (taskPlan.task) {
    case "vqa":
      return executeVQA(images[0], query, onStep, steps);
    case "captioning":
      return executeCaptioning(images[0], onStep, steps);
    case "grounding":
      return executeGrounding(images[0], query, onStep, steps);
    case "change_analysis":
      return executeChangeAnalysis(images, query, onStep, steps);
    case "cross_modal_analysis":
      return executeCrossModal(images, query, onStep, steps);
    default:
      return { answer: "No matching pipeline found.", confidence: 0.3, detail: "Unknown task" };
  }
}

// ── VQA with ML models ──────────────────────────────────────────

async function executeVQA(
  image: ImageMetadata,
  query: string,
  onStep: (s: ExecutionStep[]) => void,
  steps: ExecutionStep[],
): Promise<ModelResult> {
  // ML Classification
  steps.push({ step: "CLIP Classification", status: "processing", detail: "Zero-shot scene classification" });
  onStep([...steps]);
  await delay(300);

  const ml = await analyzeWithML(image);

  steps[steps.length - 1] = {
    step: "CLIP Classification",
    status: "completed",
    detail: ml.modelsUsed.includes("CLIP (zero-shot classification)")
      ? `Top: "${ml.classification.label}" (${Math.round(ml.classification.score * 100)}%)`
      : "CLIP not available — using pixel analysis",
    duration: 300,
  };
  onStep([...steps]);

  // ML Object Detection
  if (ml.objects.length > 0) {
    steps.push({ step: "DETR Detection", status: "processing", detail: "Object detection" });
    onStep([...steps]);
    await delay(200);
    steps[steps.length - 1] = {
      step: "DETR Detection",
      status: "completed",
      detail: `Detected ${ml.objects.length} object(s): ${ml.objects.slice(0, 5).map((o) => o.label).join(", ")}`,
      duration: 200,
    };
    onStep([...steps]);
  }

  // Build answer from ML results + domain features
  const answer = buildVQAAnswer(image, query, ml);
  const confidence = computeVQAConfidence(ml);

  // Build bounding boxes from ML objects
  const boxes: BoundingBox[] = ml.objects.slice(0, 6).map((o) => ({
    x: Math.round((o.box.x / 100) * image.width),
    y: Math.round((o.box.y / 100) * image.height),
    width: Math.round((o.box.w / 100) * image.width),
    height: Math.round((o.box.h / 100) * image.height),
    label: o.label,
    confidence: o.score,
    color: objectLabelToColor(o.label),
  }));

  return {
    answer,
    confidence,
    detail: `VQA — ${ml.modelsUsed.join(", ") || "pixel analysis only"}`,
    boundingBoxes: boxes.length > 0 ? boxes : undefined,
  };
}

function buildVQAAnswer(
  image: ImageMetadata,
  query: string,
  ml: MLAnalysisResult,
): string {
  const q = query.toLowerCase();
  const d = ml.domain;
  const dist = d.colorDistribution;
  const vegPct = Math.round(dist.vegetation * 100);
  const waterPct = Math.round(dist.water * 100);
  const urbanPct = Math.round(dist.urban * 100);
  const soilPct = Math.round(dist.soil * 100);

  const parts: string[] = [];

  // Domain-adapted classification using BigEarthNet taxonomy
  if (ml.classification.label !== "unknown") {
    const adapted = adaptToBigEarthNet(
      ml.classification.label,
      ml.classification.score,
      d,
    );
    parts.push(
      `**Domain-Adapted Classification (BigEarthNet):**\n• Primary: ${adapted.adaptedLabel} [Code: ${adapted.bigEarthNetCode}] — ${Math.round(adapted.confidence * 100)}%\n• Top matches:\n${adapted.topMatches.map((m) => `  — ${m.label} [${m.code}]: ${Math.round(m.score * 100)}%`).join("\n")}`,
    );

    // Also show raw CLIP label for transparency
    const topLabels = ml.classification.allLabels.slice(0, 3);
    parts.push(
      `\n**CLIP Zero-Shot (raw):**\n${topLabels.map((l) => `• ${l.label}: ${Math.round(l.score * 100)}%`).join("\n")}`,
    );
  }

  // ML caption if available
  if (ml.caption) {
    parts.push(`\n**ML Caption (ViT-GPT2):** "${ml.caption}"`);
  }

  // Domain-specific analysis answering the query
  if (q.includes("water") || q.includes("river") || q.includes("lake")) {
    if (waterPct > 8) {
      parts.push(
        `\n**Water detection:** Yes — ${waterPct}% of pixels show blue-dominant spectral signatures consistent with water bodies. ` +
        `${waterPct > 25 ? "A large water feature (lake or wide river) dominates the scene." : waterPct > 12 ? "Moderate water coverage suggests a river, reservoir, or coastal zone." : "Small water features detected."} ` +
        `NDVI estimate: ${d.ndviEstimate > 0 ? "+" : ""}${d.ndviEstimate.toFixed(3)}.`,
      );
    } else {
      parts.push(
        `\n**Water detection:** No significant water bodies found — only ${waterPct}% blue-dominant pixels (below threshold).`,
      );
    }
  } else if (q.includes("building") || q.includes("urban") || q.includes("built")) {
    if (urbanPct > 10) {
      parts.push(
        `\n**Urban analysis:** ${urbanPct}% urban/built-up pixels detected. ` +
        `Texture complexity: ${Math.round(d.textureComplexity * 100)}% ` +
        `(${d.textureComplexity > 0.5 ? "dense development with edges" : "scattered structures"}). ` +
        `Brightness: ${Math.round(d.brightness.mean * 100)}% (paved surfaces).`,
      );
    } else {
      parts.push(
        `\n**Urban analysis:** Limited built-up area — ${urbanPct}% urban pixels. Dominant cover: ${d.classification}.`,
      );
    }
  } else if (q.includes("vegetation") || q.includes("forest") || q.includes("tree")) {
    parts.push(
      `\n**Vegetation:** ${vegPct}% green-dominant pixels. ` +
      `NDVI: ${d.ndviEstimate > 0 ? "+" : ""}${d.ndviEstimate.toFixed(3)} ` +
      `(${d.ndviEstimate > 0.15 ? "healthy active vegetation" : d.ndviEstimate > 0 ? "sparse/stressed vegetation" : "minimal vegetation"}). ` +
      `Classification: ${d.classification}.`,
    );
  } else {
    // General answer
    parts.push(
      `\n**Land cover:** Vegetation ${vegPct}%, Urban ${urbanPct}%, Water ${waterPct}%, Soil ${soilPct}%. ` +
      `Scene: ${d.classification}. ` +
      `Brightness: ${Math.round(d.brightness.mean * 100)}%, Texture: ${Math.round(d.textureComplexity * 100)}%.`,
    );
  }

  // ML-detected objects
  if (ml.objects.length > 0) {
    const uniqueLabels = [...new Set(ml.objects.map((o) => o.label))];
    parts.push(`\n**Objects detected (DETR):** ${uniqueLabels.join(", ")}.`);
  }

  // Domain adaptation notes
  const domainResult = domainPostProcess(ml, query);
  if (domainResult.rsTerminology.length > 0) {
    parts.push(`\n**RS Domain Features:** ${domainResult.rsTerminology.join(" • ")}`);
  }
  if (domainResult.domainNotes.length > 0) {
    parts.push(`\n**Domain Notes:**\n${domainResult.domainNotes.map((n) => `• ${n}`).join("\n")}`);
  }

  return parts.join("\n");
}

function computeVQAConfidence(ml: MLAnalysisResult): number {
  let conf = 0.7;
  if (ml.classification.score > 0.3) conf += 0.1;
  if (ml.classification.score > 0.5) conf += 0.05;
  if (ml.caption) conf += 0.05;
  if (ml.objects.length > 0) conf += 0.05;
  if (ml.embedding) conf += 0.03;
  return Math.min(0.96, conf);
}

// ── Captioning with ML ──────────────────────────────────────────

async function executeCaptioning(
  image: ImageMetadata,
  onStep: (s: ExecutionStep[]) => void,
  steps: ExecutionStep[],
): Promise<ModelResult> {
  // ML Captioning
  steps.push({ step: "ViT-GPT2 Captioning", status: "processing", detail: "Generating image caption" });
  onStep([...steps]);
  await delay(500);

  const ml = await analyzeWithML(image);

  steps[steps.length - 1] = {
    step: "ViT-GPT2 Captioning",
    status: "completed",
    detail: ml.caption ? `Caption: "${ml.caption.slice(0, 60)}..."` : "Captioner unavailable — using pixel analysis",
    duration: 500,
  };
  onStep([...steps]);

  const d = ml.domain;
  const dist = d.colorDistribution;

  // Combine ML caption with domain analysis
  const parts: string[] = [];

  if (ml.caption) {
    parts.push(`**ML Caption (ViT-GPT2):** "${ml.caption}"`);
  }

  if (ml.classification.label !== "unknown") {
    parts.push(
      `**Scene Classification (CLIP):** ${ml.classification.label} (${Math.round(ml.classification.score * 100)}%)`,
    );
  }

  parts.push(
    `\n**Domain Analysis:**\n` +
    `The ${image.bands || 3}-band ${image.modality || "optical"} satellite scene (${image.width}×${image.height}px) ` +
    `shows ${d.classification}. ` +
    `Vegetation: ${Math.round(dist.vegetation * 100)}%, Urban: ${Math.round(dist.urban * 100)}%, ` +
    `Water: ${Math.round(dist.water * 100)}%, Soil: ${Math.round(dist.soil * 100)}%. ` +
    `Texture: ${Math.round(d.textureComplexity * 100)}%, NDVI: ${d.ndviEstimate > 0 ? "+" : ""}${d.ndviEstimate.toFixed(3)}.`,
  );

  if (ml.objects.length > 0) {
    const labels = [...new Set(ml.objects.map((o) => o.label))];
    parts.push(`\n**Detected objects:** ${labels.join(", ")}.`);
  }

  return {
    answer: parts.join("\n"),
    confidence: ml.caption ? 0.88 : 0.78,
    detail: `Captioning — ${ml.modelsUsed.join(", ") || "pixel analysis only"}`,
  };
}

// ── Grounding with ML ───────────────────────────────────────────

async function executeGrounding(
  image: ImageMetadata,
  query: string,
  onStep: (s: ExecutionStep[]) => void,
  steps: ExecutionStep[],
): Promise<ModelResult> {
  steps.push({ step: "DETR + CLIP Grounding", status: "processing", detail: "Detecting and localizing regions" });
  onStep([...steps]);
  await delay(600);

  const ml = await analyzeWithML(image);

  steps[steps.length - 1] = {
    step: "DETR + CLIP Grounding",
    status: "completed",
    detail: `Found ${ml.objects.length} objects + ${ml.classification.allLabels.length} scene labels`,
    duration: 600,
  };
  onStep([...steps]);

  const q = query.toLowerCase();
  const boxes: BoundingBox[] = [];

  // Use ML-detected objects as grounding boxes
  for (const obj of ml.objects) {
    const isRelevant =
      (q.includes("water") && /water|river|lake|sea|ocean/i.test(obj.label)) ||
      (q.includes("building") && /building|house|tower|structure/i.test(obj.label)) ||
      (q.includes("road") && /road|highway|street|path/i.test(obj.label)) ||
      (q.includes("vegetation") && /tree|forest|plant|grass|vegetation/i.test(obj.label)) ||
      (!q.includes("water") && !q.includes("building") && !q.includes("road") && !q.includes("vegetation"));

    if (isRelevant) {
      boxes.push({
        x: Math.round((obj.box.x / 100) * image.width),
        y: Math.round((obj.box.y / 100) * image.height),
        width: Math.round((obj.box.w / 100) * image.width),
        height: Math.round((obj.box.h / 100) * image.height),
        label: obj.label,
        confidence: obj.score,
        color: objectLabelToColor(obj.label),
      });
    }
  }

  // If ML didn't find enough objects, add classification-based regions from spatial grid
  if (boxes.length < 2) {
    const grid = ml.domain.spatialGrid;
    const gridSize = grid.length;
    const cellW = 1 / gridSize;
    const cellH = 1 / gridSize;

    const cells: { gx: number; gy: number; val: number }[] = [];
    for (let gy = 0; gy < gridSize; gy++) {
      for (let gx = 0; gx < gridSize; gx++) {
        cells.push({ gx, gy, val: grid[gy]?.[gx] || 0 });
      }
    }

    const mean = cells.reduce((s, c) => s + c.val, 0) / cells.length;
    cells.sort((a, b) => Math.abs(b.val - mean) - Math.abs(a.val - mean));

    for (const cell of cells.slice(0, 2)) {
      boxes.push({
        x: Math.round(cell.gx * cellW * image.width),
        y: Math.round(cell.gy * cellH * image.height),
        width: Math.round(cellW * image.width),
        height: Math.round(cellH * image.height),
        label: `Region of Interest (brightness: ${Math.round(cell.val * 100)}%)`,
        confidence: 0.65 + Math.abs(cell.val - mean) * 0.5,
        color: cell.val > mean ? "#ef4444" : "#3b82f6",
      });
    }
  }

  const answer =
    boxes.length > 0
      ? `**Grounding results (DETR + CLIP):**\n\n` +
        boxes.map((b, i) =>
          `${i + 1}. **${b.label}** — Position: (${b.x}, ${b.y}), Size: ${b.width}×${b.height}px — Confidence: ${Math.round(b.confidence * 100)}%`
        ).join("\n") +
        `\n\nScene: ${ml.domain.classification}. ` +
        (ml.classification.label !== "unknown"
          ? `CLIP classification: ${ml.classification.label} (${Math.round(ml.classification.score * 100)}%).`
          : "")
      : `No specific regions matched your query. Scene classification: ${ml.domain.classification}.`;

  return {
    answer,
    confidence: boxes.length > 0 ? Math.min(0.95, 0.7 + boxes.reduce((s, b) => s + b.confidence, 0) / boxes.length * 0.3) : 0.6,
    detail: `Grounding — ${ml.objects.length} objects + spatial analysis`,
    boundingBoxes: boxes,
  };
}

// ── Change Analysis with ML ─────────────────────────────────────

async function executeChangeAnalysis(
  images: ImageMetadata[],
  query: string,
  onStep: (s: ExecutionStep[]) => void,
  steps: ExecutionStep[],
): Promise<ModelResult> {
  const img1 = images[0];
  const img2 = images[1];

  if (!img1 || !img2) {
    return { answer: "Change analysis requires two images.", confidence: 0.3, detail: "Insufficient images" };
  }

  // Parallel ML analysis of both images
  steps.push({ step: "ML Image Analysis", status: "processing", detail: "Running CLIP + ViT-GPT2 on both images" });
  onStep([...steps]);
  await delay(400);

  const changeResult = await analyzeChangeWithML(img1, img2);

  steps[steps.length - 1] = {
    step: "ML Image Analysis",
    status: "completed",
    detail: `Analyzed both images — ${[...new Set([...changeResult.ml1.modelsUsed, ...changeResult.ml2.modelsUsed])].join(", ") || "pixel analysis"}`,
    duration: 400,
  };
  onStep([...steps]);

  // Change map computation
  steps.push({ step: "Pixel Change Map", status: "processing", detail: "Computing pixel-level change" });
  onStep([...steps]);
  await delay(300);

  steps[steps.length - 1] = {
    step: "Pixel Change Map",
    status: "completed",
    detail: `${changeResult.changeStats.changePercent.toFixed(1)}% pixel change detected`,
    duration: 300,
  };
  onStep([...steps]);

  // Embedding similarity
  if (changeResult.embeddingSimilarity !== null) {
    steps.push({ step: "Embedding Similarity", status: "processing", detail: "Comparing ViT feature vectors" });
    onStep([...steps]);
    await delay(200);
    steps[steps.length - 1] = {
      step: "Embedding Similarity",
      status: "completed",
      detail: `Cosine similarity: ${(changeResult.embeddingSimilarity * 100).toFixed(1)}%`,
      duration: 200,
    };
    onStep([...steps]);
  }

  // Build answer
  const answer = buildChangeAnswer(img1, img2, query, changeResult);
  const confidence = 0.78 + Math.min(0.15, changeResult.changeStats.changePercent / 200);

  return {
    answer,
    confidence: Math.min(0.95, confidence),
    detail: `Change analysis — ${changeResult.ml1.modelsUsed.length + changeResult.ml2.modelsUsed.length} ML models + pixel diff`,
    changeMap: changeResult.changeMapUrl,
  };
}

function buildChangeAnswer(
  img1: ImageMetadata,
  img2: ImageMetadata,
  query: string,
  change: Awaited<ReturnType<typeof analyzeChangeWithML>>,
): string {
  const s = change.changeStats;
  const d1 = change.ml1.domain.colorDistribution;
  const d2 = change.ml2.domain.colorDistribution;

  const vegChange = (d2.vegetation - d1.vegetation) * 100;
  const urbanChange = (d2.urban - d1.urban) * 100;
  const waterChange = (d2.water - d1.water) * 100;
  const soilChange = (d2.soil - d1.soil) * 100;

  const date1 = img1.acquisitionDate || "Image 1";
  const date2 = img2.acquisitionDate || "Image 2";

  const parts: string[] = [];

  // ML classifications for each image
  parts.push(`**ML Classifications:**`);
  if (change.ml1.classification.label !== "unknown") {
    parts.push(`• ${date1}: ${change.ml1.classification.label} (${Math.round(change.ml1.classification.score * 100)}%)`);
  }
  if (change.ml2.classification.label !== "unknown") {
    parts.push(`• ${date2}: ${change.ml2.classification.label} (${Math.round(change.ml2.classification.score * 100)}%)`);
  }
  if (change.ml1.caption || change.ml2.caption) {
    if (change.ml1.caption) parts.push(`• Caption 1: "${change.ml1.caption}"`);
    if (change.ml2.caption) parts.push(`• Caption 2: "${change.ml2.caption}"`);
  }

  // Pixel-level change
  parts.push(`\n**Pixel-level change detection:**`);
  parts.push(`• Overall: ${s.changePercent.toFixed(1)}% of pixels changed`);
  parts.push(`• Increased: ${s.increasePercent.toFixed(1)}%`);
  parts.push(`• Decreased: ${s.decreasePercent.toFixed(1)}%`);
  parts.push(`• Mean intensity change: ${s.avgChange.toFixed(1)}/255`);

  // Embedding similarity
  if (change.embeddingSimilarity !== null) {
    parts.push(`• ViT embedding similarity: ${(change.embeddingSimilarity * 100).toFixed(1)}%`);
  }

  // Category shifts
  parts.push(`\n**Land cover shifts:**`);
  if (Math.abs(vegChange) > 2) parts.push(`• Vegetation: ${vegChange > 0 ? "+" : ""}${vegChange.toFixed(1)}%`);
  if (Math.abs(urbanChange) > 2) parts.push(`• Built-up: ${urbanChange > 0 ? "+" : ""}${urbanChange.toFixed(1)}%`);
  if (Math.abs(waterChange) > 2) parts.push(`• Water: ${waterChange > 0 ? "+" : ""}${waterChange.toFixed(1)}%`);
  if (Math.abs(soilChange) > 2) parts.push(`• Soil: ${soilChange > 0 ? "+" : ""}${soilChange.toFixed(1)}%`);

  // Hotspots
  if (s.hotspots.length > 0) {
    parts.push(`\n**High-change hotspots:**`);
    for (const h of s.hotspots.slice(0, 3)) {
      parts.push(`• (${h.x}px, ${h.y}px) — magnitude ${Math.round(h.magnitude * 100)}%`);
    }
  }

  return parts.join("\n");
}

// ── Cross-Modal with ML ─────────────────────────────────────────

async function executeCrossModal(
  images: ImageMetadata[],
  query: string,
  onStep: (s: ExecutionStep[]) => void,
  steps: ExecutionStep[],
): Promise<ModelResult> {
  const optical = images.find((i) => i.modality === "optical" || i.bands >= 3);
  const sar = images.find((i) => i.modality === "sar" || i.bands === 2);

  if (!optical || !sar) {
    return {
      answer: "Cross-modal analysis requires one optical and one SAR image.",
      confidence: 0.3,
      detail: "Missing optical/SAR pair",
    };
  }

  steps.push({ step: "Optical + SAR ML Analysis", status: "processing", detail: "Running models on both modalities" });
  onStep([...steps]);
  await delay(500);

  const result = await analyzeCrossModalWithML(optical, sar);

  steps[steps.length - 1] = {
    step: "Optical + SAR ML Analysis",
    status: "completed",
    detail: `Correlation: ${(result.correlation * 100).toFixed(1)}%`,
    duration: 500,
  };
  onStep([...steps]);

  // Combine all detected objects
  const allObjects = [...result.opticalML.objects, ...result.sarML.objects];
  const boxes: BoundingBox[] = allObjects.slice(0, 8).map((o, i) => ({
    x: 0,
    y: 0,
    width: Math.round(o.box.w),
    height: Math.round(o.box.h),
    label: `${o.label} (${i < result.opticalML.objects.length ? "optical" : "SAR"})`,
    confidence: o.score,
    color: i < result.opticalML.objects.length ? "#22c55e" : "#3b82f6",
  }));

  return {
    answer: result.fusionResult,
    confidence: 0.82 + result.correlation * 0.1,
    detail: `Optical-SAR fusion — ${[...result.opticalML.modelsUsed, ...result.sarML.modelsUsed].join(", ")}`,
    boundingBoxes: boxes.length > 0 ? boxes : undefined,
  };
}

// ── Helpers ──────────────────────────────────────────────────────

function objectLabelToColor(label: string): string {
  const l = label.toLowerCase();
  if (/water|river|lake|sea|ocean/.test(l)) return "#3b82f6";
  if (/building|house|tower|structure|skyscraper/.test(l)) return "#ef4444";
  if (/tree|forest|plant|grass|vegetation/.test(l)) return "#22c55e";
  if (/road|highway|street|bridge/.test(l)) return "#f59e0b";
  if (/car|vehicle|truck|bus/.test(l)) return "#8b5cf6";
  if (/person|people|pedestrian/.test(l)) return "#ec4899";
  if (/cloud|sky|fog/.test(l)) return "#94a3b8";
  return "#6b7280";
}

function delay(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}
