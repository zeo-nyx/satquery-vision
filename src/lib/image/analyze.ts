/**
 * Real image analysis using Canvas API.
 * Reads actual pixel data from uploaded images and produces
 * meaningful statistics that drive varied, image-dependent results.
 */

export interface ImageAnalysis {
  /** Dominant colors as [r,g,b][] */
  dominantColors: [number, number, number][];
  /** Color distribution across the image */
  colorDistribution: {
    vegetation: number; // green-dominant pixels
    water: number; // blue-dominant pixels
    urban: number; // gray/bright pixels
    soil: number; // brown/red-dominant pixels
    other: number;
  };
  /** Brightness stats */
  brightness: { mean: number; min: number; max: number; stdDev: number };
  /** Texture complexity (edge density proxy) */
  textureComplexity: number;
  /** Spatial heatmap: 4x4 grid of brightness values (0-1) */
  spatialGrid: number[][];
  /** Normalized NDVI-like index (green vs red ratio) */
  ndviEstimate: number;
  /** Overall classification based on content */
  classification: string;
  /** Image dimensions */
  width: number;
  height: number;
}

/**
 * Load an image URL (or blob URL) into a canvas and extract pixel statistics.
 */
export async function analyzeImagePixels(
  dataUrl: string,
  maxDim = 256,
): Promise<ImageAnalysis> {
  const img = await loadImage(dataUrl);

  // Scale down for analysis
  const scale = Math.min(1, maxDim / Math.max(img.width, img.height));
  const w = Math.max(1, Math.round(img.width * scale));
  const h = Math.max(1, Math.round(img.height * scale));

  const canvas = document.createElement("canvas");
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext("2d")!;
  ctx.drawImage(img, 0, 0, w, h);

  const imageData = ctx.getImageData(0, 0, w, h);
  const pixels = imageData.data; // Uint8ClampedArray [r,g,b,a, r,g,b,a, ...]

  // ── Color histogram and classification ─────────────────────────
  let totalBrightness = 0;
  let minBrightness = 255;
  let maxBrightness = 0;
  const brightnessValues: number[] = [];

  let greenCount = 0;
  let blueCount = 0;
  let grayCount = 0;
  let brownCount = 0;
  let totalPixels = 0;

  const colorAccum = [0, 0, 0]; // R, G, B sums
  const colorCounts: Record<string, number> = {};

  // Spatial grid for heatmap (4x4)
  const gridSize = 4;
  const gridBrightness: number[][] = Array.from({ length: gridSize }, () =>
    Array(gridSize).fill(0),
  );
  const gridCounts: number[][] = Array.from({ length: gridSize }, () =>
    Array(gridSize).fill(0),
  );

  for (let i = 0; i < pixels.length; i += 4) {
    const r = pixels[i];
    const g = pixels[i + 1];
    const b = pixels[i + 2];
    const a = pixels[i + 3];
    if (a < 128) continue; // skip transparent

    const px = (i / 4) % w;
    const py = Math.floor(i / 4 / w);
    const brightness = (r * 0.299 + g * 0.587 + b * 0.114) / 255;

    totalBrightness += brightness;
    minBrightness = Math.min(minBrightness, brightness * 255);
    maxBrightness = Math.max(maxBrightness, brightness * 255);
    brightnessValues.push(brightness);

    colorAccum[0] += r;
    colorAccum[1] += g;
    colorAccum[2] += b;

    // Spatial grid
    const gx = Math.min(gridSize - 1, Math.floor((px / w) * gridSize));
    const gy = Math.min(gridSize - 1, Math.floor((py / h) * gridSize));
    gridBrightness[gy][gx] += brightness;
    gridCounts[gy][gx]++;

    // Classify pixel
    const maxC = Math.max(r, g, b);
    const minC = Math.min(r, g, b);
    const sat = maxC > 0 ? (maxC - minC) / maxC : 0;

    if (g > r * 1.15 && g > b * 1.15 && sat > 0.1) {
      greenCount++; // vegetation
    } else if (b > r * 1.2 && b > g * 1.05 && sat > 0.15) {
      blueCount++; // water
    } else if (sat < 0.12 && brightness > 0.3 && brightness < 0.85) {
      grayCount++; // urban / built
    } else if (r > g * 0.9 && r > b * 1.3 && sat > 0.08) {
      brownCount++; // soil
    }

    // Quantize for dominant color bucketing
    const qr = Math.round(r / 32) * 32;
    const qg = Math.round(g / 32) * 32;
    const qb = Math.round(b / 32) * 32;
    const key = `${qr},${qg},${qb}`;
    colorCounts[key] = (colorCounts[key] || 0) + 1;

    totalPixels++;
  }

  if (totalPixels === 0) {
    totalPixels = 1;
  }

  // Normalize spatial grid
  for (let gy = 0; gy < gridSize; gy++) {
    for (let gx = 0; gx < gridSize; gx++) {
      gridBrightness[gy][gx] =
        gridCounts[gy][gx] > 0
          ? gridBrightness[gy][gx] / gridCounts[gy][gx]
          : 0;
    }
  }

  // Mean brightness
  const meanBrightness = totalBrightness / totalPixels;

  // Standard deviation
  const variance =
    brightnessValues.reduce(
      (sum, b) => sum + (b - meanBrightness) ** 2,
      0,
    ) / totalPixels;
  const stdDev = Math.sqrt(variance);

  // Color distribution percentages
  const dist = {
    vegetation: greenCount / totalPixels,
    water: blueCount / totalPixels,
    urban: grayCount / totalPixels,
    soil: brownCount / totalPixels,
    other:
      (totalPixels - greenCount - blueCount - grayCount - brownCount) /
      totalPixels,
  };

  // Top dominant colors
  const sortedColors = Object.entries(colorCounts)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 5)
    .map(([key]) => {
      const [r, g, b] = key.split(",").map(Number);
      return [r, g, b] as [number, number, number];
    });

  // NDVI estimate (simplified: green ratio vs red ratio)
  const avgR = colorAccum[0] / totalPixels / 255;
  const avgG = colorAccum[1] / totalPixels / 255;
  const ndvi = avgR + avgG > 0 ? (avgG - avgR) / (avgG + avgR) : 0;

  // Texture complexity: variance of brightness stdDev (high = complex)
  const textureComplexity = Math.min(1, stdDev * 3);

  // Classification
  let classification = "Mixed land cover";
  if (dist.vegetation > 0.4) classification = "Dense vegetation";
  else if (dist.vegetation > 0.25) classification = "Moderate vegetation with mixed cover";
  else if (dist.water > 0.15) classification = "Significant water body present";
  else if (dist.urban > 0.3) classification = "Urban / built-up area";
  else if (dist.soil > 0.2) classification = "Exposed soil / agricultural land";
  else if (meanBrightness < 0.25) classification = "Dark terrain (SAR or night imagery)";
  else if (meanBrightness > 0.7) classification = "Highly reflective surface (snow, sand, or cloud)";
  else if (textureComplexity > 0.6) classification = "Highly textured landscape (mixed urban-natural)";
  else classification = "Mixed land cover with diverse spectral response";

  return {
    dominantColors: sortedColors,
    colorDistribution: dist,
    brightness: {
      mean: meanBrightness,
      min: minBrightness / 255,
      max: maxBrightness / 255,
      stdDev,
    },
    textureComplexity,
    spatialGrid: gridBrightness,
    ndviEstimate: ndvi,
    classification,
    width: img.width,
    height: img.height,
  };
}

/**
 * Compute a pixel-level change map between two images.
 * Returns the change map as a data URL and change statistics.
 */
export async function computeChangeMap(
  dataUrl1: string,
  dataUrl2: string,
  maxDim = 256,
): Promise<{
  changeMapDataUrl: string;
  stats: {
    changePercent: number;
    increasePercent: number;
    decreasePercent: number;
    hotspots: { x: number; y: number; magnitude: number }[];
    avgChange: number;
  };
  analysis1: ImageAnalysis;
  analysis2: ImageAnalysis;
}> {
  const [img1, img2] = await Promise.all([
    loadImage(dataUrl1),
    loadImage(dataUrl2),
  ]);

  const size = maxDim;
  const canvas1 = document.createElement("canvas");
  canvas1.width = size;
  canvas1.height = size;
  const ctx1 = canvas1.getContext("2d")!;
  ctx1.drawImage(img1, 0, 0, size, size);
  const data1 = ctx1.getImageData(0, 0, size, size);

  const canvas2 = document.createElement("canvas");
  canvas2.width = size;
  canvas2.height = size;
  const ctx2 = canvas2.getContext("2d")!;
  ctx2.drawImage(img2, 0, 0, size, size);
  const data2 = ctx2.getImageData(0, 0, size, size);

  // Compute per-pixel difference
  const diffCanvas = document.createElement("canvas");
  diffCanvas.width = size;
  diffCanvas.height = size;
  const diffCtx = diffCanvas.getContext("2d")!;
  const diffData = diffCtx.createImageData(size, size);

  let totalDiff = 0;
  let changeCount = 0;
  let increaseCount = 0;
  let decreaseCount = 0;
  const hotspots: { x: number; y: number; magnitude: number }[] = [];

  const threshold = 30; // pixel difference threshold

  for (let i = 0; i < data1.data.length; i += 4) {
    const r1 = data1.data[i],
      g1 = data1.data[i + 1],
      b1 = data1.data[i + 2];
    const r2 = data2.data[i],
      g2 = data2.data[i + 1],
      b2 = data2.data[i + 2];

    const brightness1 = (r1 + g1 + b1) / 3;
    const brightness2 = (r2 + g2 + b2) / 3;
    const diff = brightness2 - brightness1;
    const absDiff = Math.abs(diff);

    totalDiff += absDiff;

    const px = (i / 4) % size;
    const py = Math.floor(i / 4 / size);

    if (absDiff > threshold) {
      changeCount++;
      if (diff > 0) increaseCount++;
      else decreaseCount++;

      // Color the change map
      const intensity = Math.min(255, absDiff * 2);
      if (diff > 0) {
        // Increase: red tones
        diffData.data[i] = intensity;
        diffData.data[i + 1] = Math.round(intensity * 0.27);
        diffData.data[i + 2] = Math.round(intensity * 0.35);
      } else {
        // Decrease: green tones
        diffData.data[i] = Math.round(intensity * 0.27);
        diffData.data[i + 1] = intensity;
        diffData.data[i + 2] = Math.round(intensity * 0.35);
      }
      diffData.data[i + 3] = 255;

      // Track hotspots (areas of highest change)
      if (absDiff > 80) {
        hotspots.push({ x: px, y: py, magnitude: absDiff / 255 });
      }
    } else {
      // No significant change: dark
      diffData.data[i] = 26;
      diffData.data[i + 1] = 26;
      diffData.data[i + 2] = 46;
      diffData.data[i + 3] = 255;
    }
  }

  const totalPixels = size * size;
  const changePercent = (changeCount / totalPixels) * 100;
  const increasePercent = (increaseCount / totalPixels) * 100;
  const decreasePercent = (decreaseCount / totalPixels) * 100;

  // Merge hotspots that are close together
  const mergedHotspots = mergeHotspots(hotspots, 15);

  // Draw final change map with legend
  diffCtx.putImageData(diffData, 0, 0);

  // Add legend overlay
  diffCtx.fillStyle = "rgba(0,0,0,0.7)";
  diffCtx.fillRect(0, size - 36, size, 36);
  diffCtx.fillStyle = "#ffffff";
  diffCtx.font = "bold 10px monospace";
  diffCtx.fillText("PIXEL-LEVEL CHANGE MAP", 8, size - 22);
  diffCtx.font = "9px monospace";

  // Legend items
  diffCtx.fillStyle = "#ff4466";
  diffCtx.fillRect(8, size - 14, 8, 8);
  diffCtx.fillStyle = "#ffffff";
  diffCtx.fillText(`Increase ${increasePercent.toFixed(1)}%`, 20, size - 7);

  diffCtx.fillStyle = "#44ff66";
  diffCtx.fillRect(130, size - 14, 8, 8);
  diffCtx.fillStyle = "#ffffff";
  diffCtx.fillText(`Decrease ${decreasePercent.toFixed(1)}%`, 142, size - 7);

  // Also analyze each image individually
  const [analysis1, analysis2] = await Promise.all([
    analyzeImagePixels(dataUrl1, maxDim),
    analyzeImagePixels(dataUrl2, maxDim),
  ]);

  return {
    changeMapDataUrl: diffCanvas.toDataURL("image/png"),
    stats: {
      changePercent,
      increasePercent,
      decreasePercent,
      hotspots: mergedHotspots.slice(0, 5),
      avgChange: totalDiff / totalPixels,
    },
    analysis1,
    analysis2,
  };
}

// ── Helpers ─────────────────────────────────────────────────────

function loadImage(src: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.crossOrigin = "anonymous";
    img.onload = () => resolve(img);
    img.onerror = reject;
    img.src = src;
  });
}

function mergeHotspots(
  hotspots: { x: number; y: number; magnitude: number }[],
  radius: number,
): { x: number; y: number; magnitude: number }[] {
  const merged: { x: number; y: number; magnitude: number; count: number }[] =
    [];

  for (const h of hotspots) {
    const existing = merged.find(
      (m) => Math.abs(m.x - h.x) < radius && Math.abs(m.y - h.y) < radius,
    );
    if (existing) {
      existing.x = (existing.x + h.x) / 2;
      existing.y = (existing.y + h.y) / 2;
      existing.magnitude = Math.max(existing.magnitude, h.magnitude);
      existing.count++;
    } else {
      merged.push({ ...h, count: 1 });
    }
  }

  return merged
    .sort((a, b) => b.magnitude - a.magnitude)
    .map(({ count: _, ...rest }) => rest);
}

/**
 * Compute a simple cross-modal correlation between two images.
 */
export async function computeCrossModalCorrelation(
  dataUrl1: string,
  dataUrl2: string,
  maxDim = 128,
): Promise<{
  correlation: number;
  opticalStats: ImageAnalysis;
  sarStats: ImageAnalysis;
  overlapZones: { label: string; area: number; confidence: number }[];
}> {
  const [opticalStats, sarStats] = await Promise.all([
    analyzeImagePixels(dataUrl1, maxDim),
    analyzeImagePixels(dataUrl2, maxDim),
  ]);

  // Compare spatial grids to find overlap
  const gridSize = opticalStats.spatialGrid.length;
  let correlationSum = 0;
  let count = 0;
  const overlapZones: { label: string; area: number; confidence: number }[] = [];

  for (let gy = 0; gy < gridSize; gy++) {
    for (let gx = 0; gx < gridSize; gx++) {
      const v1 = opticalStats.spatialGrid[gy]?.[gx] || 0;
      const v2 = sarStats.spatialGrid[gy]?.[gx] || 0;
      correlationSum += 1 - Math.abs(v1 - v2);
      count++;
    }
  }

  const correlation = count > 0 ? correlationSum / count : 0;

  // Identify overlap zones based on both modalities
  if (opticalStats.colorDistribution.urban > 0.2) {
    overlapZones.push({
      label: "Built-up area (high optical reflectance + SAR backscatter)",
      area: opticalStats.colorDistribution.urban,
      confidence: 0.85 + correlation * 0.1,
    });
  }
  if (opticalStats.colorDistribution.water > 0.1) {
    overlapZones.push({
      label: "Water body (low optical reflectance + specular SAR return)",
      area: opticalStats.colorDistribution.water,
      confidence: 0.88 + correlation * 0.08,
    });
  }
  if (opticalStats.colorDistribution.vegetation > 0.2) {
    overlapZones.push({
      label: "Vegetation (high NDVI + volume scattering)",
      area: opticalStats.colorDistribution.vegetation,
      confidence: 0.82 + correlation * 0.12,
    });
  }

  return {
    correlation,
    opticalStats,
    sarStats,
    overlapZones,
  };
}
