import type {
  ImageMetadata,
  TaskPlan,
  AnalysisResult,
  ExecutionStep,
  BoundingBox,
} from "../agent/types";

/**
 * Execute an analysis pipeline based on the agent's task plan.
 * In production this would call real model endpoints; here we produce
 * realistic simulated results that demonstrate the full architecture.
 */
export async function executeAnalysis(
  taskPlan: TaskPlan,
  images: ImageMetadata[],
  query: string,
  onStep: (steps: ExecutionStep[]) => void,
): Promise<AnalysisResult> {
  const steps: ExecutionStep[] = [];
  const startTime = Date.now();

  // Step 1: Input validation
  steps.push({
    step: "Input Validation",
    status: "processing",
    detail: `Validating ${images.length} image(s)`,
  });
  onStep([...steps]);
  await delay(400);

  const validationDetail = validateInputs(taskPlan, images);
  steps[steps.length - 1] = {
    step: "Input Validation",
    status: "completed",
    detail: validationDetail,
    duration: 400,
  };
  onStep([...steps]);

  // Step 2: Query classification
  steps.push({
    step: "Query Classification",
    status: "processing",
    detail: `Classifying query as: ${taskPlan.task}`,
  });
  onStep([...steps]);
  await delay(300);

  steps[steps.length - 1] = {
    step: "Query Classification",
    status: "completed",
    detail: `Task: ${taskPlan.description}`,
    duration: 300,
  };
  onStep([...steps]);

  // Step 3: Model selection
  steps.push({
    step: "Model Selection",
    status: "processing",
    detail: `Selecting models: ${taskPlan.models.join(", ")}`,
  });
  onStep([...steps]);
  await delay(250);

  steps[steps.length - 1] = {
    step: "Model Selection",
    status: "completed",
    detail: `Selected ${taskPlan.models.length} model(s) for ${taskPlan.inputType}`,
    duration: 250,
  };
  onStep([...steps]);

  // Step 4: Execute model pipeline
  steps.push({
    step: "Model Execution",
    status: "processing",
    detail: `Running ${taskPlan.task} pipeline`,
  });
  onStep([...steps]);
  await delay(800);

  const modelResult = await executeModelPipeline(taskPlan, images, query);
  steps[steps.length - 1] = {
    step: "Model Execution",
    status: "completed",
    detail: modelResult.detail,
    duration: 800,
  };
  onStep([...steps]);

  // Step 5: Evidence fusion (for multi-model tasks)
  if (taskPlan.models.length > 1) {
    steps.push({
      step: "Evidence Fusion",
      status: "processing",
      detail: "Combining model outputs",
    });
    onStep([...steps]);
    await delay(350);

    steps[steps.length - 1] = {
      step: "Evidence Fusion",
      status: "completed",
      detail: `Fused ${taskPlan.models.length} model outputs with weighted confidence`,
      duration: 350,
    };
    onStep([...steps]);
  }

  // Step 6: Answer generation
  steps.push({
    step: "Answer Generation",
    status: "processing",
    detail: "Generating final response",
  });
  onStep([...steps]);
  await delay(200);

  steps[steps.length - 1] = {
    step: "Answer Generation",
    status: "completed",
    detail: "Response generated with confidence score",
    duration: 200,
  };
  onStep([...steps]);

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
    parts.push("2 GeoTIFF images");
    if (images[0].acquisitionDate && images[1].acquisitionDate) {
      parts.push(
        `Dates: ${images[0].acquisitionDate} → ${images[1].acquisitionDate}`,
      );
    }
    parts.push("Same geographic region assumed");
  } else if (taskPlan.inputType === "optical_sar_pair") {
    const optical = images.find(
      (i) => i.modality === "optical" || i.bands >= 3,
    );
    const sar = images.find((i) => i.modality === "sar");
    parts.push(
      `Optical: ${optical?.fileName || "N/A"}, SAR: ${sar?.fileName || "N/A"}`,
    );
    parts.push("Cross-modal co-registration applied");
  } else {
    parts.push(`${images.length} image(s)`);
    if (images[0]) {
      parts.push(
        `${images[0].bands}-band ${images[0].modality || "unknown"} imagery`,
      );
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
): Promise<ModelResult> {
  switch (taskPlan.task) {
    case "vqa":
      return executeVQA(images[0], query);
    case "captioning":
      return executeCaptioning(images[0]);
    case "grounding":
      return executeGrounding(images[0], query);
    case "change_analysis":
      return executeChangeAnalysis(images, query);
    case "cross_modal_analysis":
      return executeCrossModal(images, query);
    default:
      return {
        answer: "Unable to process this query with available models.",
        confidence: 0.3,
        detail: "No matching pipeline found",
      };
  }
}

function executeVQA(
  image: ImageMetadata,
  query: string,
): ModelResult {
  const q = query.toLowerCase();

  // Generate context-aware simulated answers
  let answer = "";
  let confidence = 0.85;

  if (q.includes("water") || q.includes("river") || q.includes("lake")) {
    answer =
      "Yes, a water body is visible in the satellite image. The spectral signature indicates a permanent water feature with low reflectance in the NIR band, consistent with a river or lake. The water body appears to occupy approximately 12% of the image area.";
    confidence = 0.89;
  } else if (
    q.includes("building") ||
    q.includes("urban") ||
    q.includes("built")
  ) {
    answer =
      "Yes, urban/built-up structures are present in the image. Dense building clusters are visible in the central and southeastern portions, with a regular grid pattern suggesting planned urban development. Road networks connect the built-up areas.";
    confidence = 0.92;
  } else if (q.includes("vegetation") || q.includes("forest") || q.includes("tree")) {
    answer =
      "Vegetation cover is present in the image. The NDVI-like spectral response indicates healthy vegetation in the western and northern regions, with density varying from sparse shrubland to denser canopy cover. Agricultural parcels are visible in the southern area.";
    confidence = 0.87;
  } else if (q.includes("road") || q.includes("infrastructure")) {
    answer =
      "Road infrastructure is identifiable in the image. A primary road network runs through the central area, with secondary roads connecting to peripheral zones. The linear features show high reflectance consistent with paved surfaces.";
    confidence = 0.84;
  } else {
    answer = `Based on the satellite imagery analysis, the image shows a mixed landscape with diverse land cover types. The spectral characteristics suggest a combination of urban development, vegetation, and open terrain. The spatial resolution allows identification of structures and natural features at the sub-hectare scale. Key observations include visible human-made structures, vegetation patterns, and terrain features consistent with a semi-urban to peri-urban environment.`;
    confidence = 0.81;
  }

  const bands = image.bands || 3;
  return {
    answer,
    confidence,
    detail: `VQA processed ${bands}-band ${image.modality || "optical"} imagery (${image.width}×${image.height})`,
  };
}

function executeCaptioning(image: ImageMetadata): ModelResult {
  const modalityDesc = image.modality === "sar" ? "SAR (synthetic aperture radar)" : "optical multispectral";
  const bandDesc = image.bands >= 4 ? "RGB+NIR" : image.bands >= 3 ? "RGB" : image.bands === 2 ? "dual-pol" : "single-band";

  const answer = `The image displays a ${modalityDesc} satellite scene (${bandDesc}, ${image.width}×${image.height}px) capturing a mixed-use landscape. The scene includes urban/residential zones with dense building clusters in the central-eastern portion, surrounded by agricultural parcels with varying crop stages. A drainage network runs through the northern section. Vegetation density ranges from sparse ground cover to moderate canopy, with spectral signatures indicating active growth. Road infrastructure connects the main settlement areas. The overall landscape pattern is consistent with a peri-urban environment undergoing gradual expansion.`;

  return {
    answer,
    confidence: 0.86,
    detail: `Captioning completed on ${image.bands}-band imagery`,
  };
}

function executeGrounding(
  image: ImageMetadata,
  query: string,
): ModelResult {
  const q = query.toLowerCase();
  const w = image.width;
  const h = image.height;

  const boxes: BoundingBox[] = [];

  if (q.includes("water") || q.includes("river") || q.includes("lake")) {
    boxes.push({
      x: Math.round(w * 0.1),
      y: Math.round(h * 0.05),
      width: Math.round(w * 0.35),
      height: Math.round(h * 0.12),
      label: "Water Body",
      confidence: 0.91,
      color: "#3b82f6",
    });
  }

  if (q.includes("building") || q.includes("urban") || q.includes("structure")) {
    boxes.push({
      x: Math.round(w * 0.3),
      y: Math.round(h * 0.25),
      width: Math.round(w * 0.4),
      height: Math.round(h * 0.35),
      label: "Urban Area",
      confidence: 0.88,
      color: "#ef4444",
    });
    boxes.push({
      x: Math.round(w * 0.55),
      y: Math.round(h * 0.6),
      width: Math.round(w * 0.25),
      height: Math.round(h * 0.2),
      label: "Industrial Zone",
      confidence: 0.79,
      color: "#f97316",
    });
  }

  if (q.includes("vegetation") || q.includes("forest") || q.includes("green")) {
    boxes.push({
      x: Math.round(w * 0.02),
      y: Math.round(h * 0.15),
      width: Math.round(w * 0.28),
      height: Math.round(h * 0.5),
      label: "Dense Vegetation",
      confidence: 0.93,
      color: "#22c55e",
    });
    boxes.push({
      x: Math.round(w * 0.65),
      y: Math.round(h * 0.05),
      width: Math.round(w * 0.3),
      height: Math.round(h * 0.35),
      label: "Sparse Vegetation",
      confidence: 0.82,
      color: "#84cc16",
    });
  }

  // Default grounding if no specific keyword matched
  if (boxes.length === 0) {
    boxes.push(
      {
        x: Math.round(w * 0.2),
        y: Math.round(h * 0.15),
        width: Math.round(w * 0.35),
        height: Math.round(h * 0.3),
        label: "Primary Region of Interest",
        confidence: 0.85,
        color: "#f59e0b",
      },
      {
        x: Math.round(w * 0.5),
        y: Math.round(h * 0.5),
        width: Math.round(w * 0.4),
        height: Math.round(h * 0.35),
        label: "Secondary Region",
        confidence: 0.74,
        color: "#8b5cf6",
      },
    );
  }

  const avgConf = boxes.reduce((s, b) => s + b.confidence, 0) / boxes.length;

  return {
    answer: `Identified ${boxes.length} region(s) of interest in the satellite image. ${boxes.map((b) => `"${b.label}" (${Math.round(b.confidence * 100)}% confidence)`).join(", ")}. The bounding boxes highlight areas matching your query.`,
    confidence: avgConf,
    detail: `Grounding completed — ${boxes.length} regions detected`,
    boundingBoxes: boxes,
  };
}

function executeChangeAnalysis(
  images: ImageMetadata[],
  query: string,
): ModelResult {
  const img1 = images[0];
  const img2 = images[1];
  const q = query.toLowerCase();

  // Generate a change map data URL
  const changeMapUrl = generateChangeMapDataUrl(
    img1.width,
    img1.height,
  );

  let answer = "";
  let confidence = 0.88;

  if (
    q.includes("built") ||
    q.includes("urban") ||
    q.includes("building") ||
    q.includes("construction")
  ) {
    answer = `Bi-temporal analysis of the two satellite images reveals significant changes in built-up areas. Between ${img1.acquisitionDate || "the first observation"} and ${img2.acquisitionDate || "the second observation"}, built-up area has increased by approximately 18.5%. The primary expansion occurred in the northeastern quadrant, where new residential structures and road extensions are visible. A secondary cluster of new construction appears in the southern peri-urban zone. Vegetation loss of approximately 7.2% correlates spatially with the new built-up areas, suggesting land conversion for urban development.`;
    confidence = 0.91;
  } else if (
    q.includes("vegetation") ||
    q.includes("deforestation") ||
    q.includes("forest")
  ) {
    answer = `Change analysis reveals a net vegetation loss of approximately 12.4% between the two observation dates. Deforestation is concentrated in the western sector, where dense canopy has been replaced by sparse ground cover and early-stage clearing. Conversely, vegetation regrowth is detected in abandoned agricultural parcels in the southeastern area, indicating secondary succession. The overall NDVI trend shows a -0.08 decline across the scene.`;
    confidence = 0.87;
  } else if (q.includes("water")) {
    answer = `Water body analysis indicates a 5.3% reduction in surface water extent between the two dates. The northern reservoir shows lower water levels, with exposed shoreline visible in the second image. Seasonal variation or upstream water extraction may account for the observed change. No new water bodies have appeared.`;
    confidence = 0.84;
  } else {
    answer = `Change detection analysis between ${img1.acquisitionDate || "Image 1"} and ${img2.acquisitionDate || "Image 2"} reveals multiple categories of change. Built-up area expansion (+18.5%) is the most significant change, concentrated in the northeastern and southern sectors. Vegetation cover has decreased (-7.2%), primarily in areas adjacent to new construction. Road network expansion is visible, with approximately 2.3 km of new road surface detected. The overall landscape shows a transition from agricultural to mixed-use development. The change map provides pixel-level evidence of all detected changes.`;
  }

  return {
    answer,
    confidence,
    detail: `Change detection completed on bi-temporal pair`,
    changeMap: changeMapUrl,
  };
}

function executeCrossModal(
  images: ImageMetadata[],
  query: string,
): ModelResult {
  const optical = images.find(
    (i) => i.modality === "optical" || i.bands >= 3,
  );
  const sar = images.find((i) => i.modality === "sar");

  const answer = `Cross-modal analysis combining optical (${optical?.fileName || "image 1"}) and SAR (${sar?.fileName || "image 2"}) data provides enhanced land cover characterization. The optical imagery reveals surface appearance and spectral properties — vegetation indices, water turbidity, and building material reflectance. The SAR data provides complementary structural information: surface roughness, moisture content, and dielectric properties. 

Fused analysis results:
• Built-up areas: Identified through high optical reflectance combined with high SAR backscatter (double-bounce mechanism). Coverage estimated at 34.2% of the scene.
• Water bodies: Low optical reflectance in visible bands combined with very low SAR backscatter (specular reflection). Water extent covers approximately 8.7%.
• Vegetation: High NDVI from optical data with moderate SAR backscatter (volume scattering). Dense vegetation in the western sector, sparse in agricultural areas.
• Bare soil / cleared land: Moderate optical reflectance with variable SAR return depending on surface moisture.

The optical-SAR fusion achieves higher classification accuracy (estimated 93.4%) compared to either modality alone, particularly in cloud-affected regions where SAR provides reliable structural information.`;

  return {
    answer,
    confidence: 0.90,
    detail: `Optical-SAR fusion completed — ${images.length} modalities combined`,
  };
}

function generateChangeMapDataUrl(width: number, height: number): string {
  const canvas = document.createElement("canvas");
  const size = 256;
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext("2d");
  if (!ctx) return "";

  // Background
  ctx.fillStyle = "#1a1a2e";
  ctx.fillRect(0, 0, size, size);

  // Grid lines
  ctx.strokeStyle = "#2a2a4e";
  ctx.lineWidth = 1;
  for (let i = 0; i < size; i += 16) {
    ctx.beginPath();
    ctx.moveTo(i, 0);
    ctx.lineTo(i, size);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(0, i);
    ctx.lineTo(size, i);
    ctx.stroke();
  }

  // Simulate change regions
  const regions = [
    { x: 160, y: 20, w: 60, h: 40, color: "#ff4444", alpha: 0.7 },
    { x: 180, y: 60, w: 45, h: 50, color: "#ff4444", alpha: 0.5 },
    { x: 20, y: 80, w: 70, h: 60, color: "#44ff44", alpha: 0.5 },
    { x: 100, y: 160, w: 80, h: 40, color: "#ff4444", alpha: 0.6 },
    { x: 40, y: 180, w: 50, h: 50, color: "#44aaff", alpha: 0.4 },
    { x: 130, y: 100, w: 40, h: 30, color: "#ffaa00", alpha: 0.5 },
  ];

  for (const r of regions) {
    ctx.globalAlpha = r.alpha;
    ctx.fillStyle = r.color;
    ctx.fillRect(r.x, r.y, r.w, r.h);
  }
  ctx.globalAlpha = 1;

  // Legend
  ctx.fillStyle = "#ffffff";
  ctx.font = "bold 10px monospace";
  ctx.fillText("CHANGE MAP", 8, size - 30);

  ctx.fillStyle = "#ff4444";
  ctx.fillRect(8, size - 22, 8, 8);
  ctx.fillStyle = "#ffffff";
  ctx.font = "8px monospace";
  ctx.fillText("Increase", 20, size - 15);

  ctx.fillStyle = "#44ff44";
  ctx.fillRect(80, size - 22, 8, 8);
  ctx.fillStyle = "#ffffff";
  ctx.fillText("Decrease", 92, size - 15);

  ctx.fillStyle = "#44aaff";
  ctx.fillRect(152, size - 22, 8, 8);
  ctx.fillStyle = "#ffffff";
  ctx.fillText("Water", 164, size - 15);

  return canvas.toDataURL("image/png");
}

function delay(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}
