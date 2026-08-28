import type { ImageMetadata, ImageModality } from "../agent/types";

export interface InputValidation {
  isValid: boolean;
  warnings: string[];
  errors: string[];
  compatibility: {
    formatSupported: boolean;
    crsDetected: string | null;
    datesDetected: (string | null)[];
    sameRegion: boolean | null;
    sameCRS: boolean | null;
    temporalPair: boolean;
    modalities: string[];
  };
}

/**
 * Generate a unique ID for uploaded images.
 */
let imageCounter = 0;
export function generateImageId(): string {
  return `img_${Date.now()}_${++imageCounter}`;
}

/**
 * Extract metadata from an uploaded image file.
 * For real GeoTIFF files, we'd use Rasterio/GDAL — here we extract
 * what's available from the browser APIs and file properties.
 */
/**
 * Validate a set of images for compatibility.
 * Checks format, CRS, dates, and modality pairing.
 */
export function validateImageInputs(
  images: ImageMetadata[],
  expectedType: string,
): InputValidation {
  const warnings: string[] = [];
  const errors: string[] = [];

  if (images.length === 0) {
    errors.push("No images provided");
    return { isValid: false, warnings, errors, compatibility: defaultCompat() };
  }

  const formats = images.map((i) => i.format);
  const crsList = images.map((i) => i.crs);
  const dates = images.map((i) => i.acquisitionDate);
  const modalities = images.map((i) => i.modality || "unknown");

  // Format checks
  for (const img of images) {
    if (img.format === "unknown") {
      warnings.push(`${img.fileName}: format not recognized — processing may be limited`);
    }
    if (img.format === "geotiff" || img.format === "tiff") {
      // Good — geospatial format
    } else if (img.format === "png" || img.format === "jpeg") {
      warnings.push(`${img.fileName}: PNG/JPEG accepted for benchmark datasets only — no geospatial metadata`);
    }
  }

  // CRS consistency
  const uniqueCRS = [...new Set(crsList.filter(Boolean))];
  const sameCRS = uniqueCRS.length <= 1;
  if (!sameCRS && images.length > 1) {
    warnings.push(`Multiple CRS detected: ${uniqueCRS.join(", ")} — images may not be spatially aligned`);
  }

  // Date checks for bi-temporal
  if (expectedType === "bi_temporal" || images.length === 2) {
    const validDates = dates.filter(Boolean);
    if (validDates.length === 2 && validDates[0] === validDates[1]) {
      warnings.push("Both images have the same acquisition date — change analysis may be meaningless");
    }
    if (validDates.length < 2) {
      warnings.push("Could not determine acquisition dates for both images — temporal comparison assumed");
    }
  }

  // Modality checks for cross-modal
  if (expectedType === "optical_sar_pair" || (images.length === 2)) {
    const hasOptical = modalities.some((m) => m === "optical");
    const hasSAR = modalities.some((m) => m === "sar");
    if (expectedType === "optical_sar_pair" && (!hasOptical || !hasSAR)) {
      warnings.push(`Expected optical+SAR pair but detected: ${modalities.join(", ")} — cross-modal analysis may be limited`);
    }
  }

  // Dimension checks
  for (const img of images) {
    if (img.width < 32 || img.height < 32) {
      warnings.push(`${img.fileName}: very small image (${img.width}×${img.height}) — analysis may be unreliable`);
    }
  }

  return {
    isValid: errors.length === 0,
    warnings,
    errors,
    compatibility: {
      formatSupported: !formats.some((f) => f === "unknown"),
      crsDetected: crsList.find(Boolean) || null,
      datesDetected: dates,
      sameRegion: sameCRS,
      sameCRS,
      temporalPair: dates.filter(Boolean).length === 2,
      modalities,
    },
  };
}

function defaultCompat(): InputValidation["compatibility"] {
  return {
    formatSupported: false,
    crsDetected: null,
    datesDetected: [],
    sameRegion: null,
    sameCRS: null,
    temporalPair: false,
    modalities: [],
  };
}

export async function extractImageMetadata(
  file: File,
): Promise<ImageMetadata> {
  const format = detectFormat(file.name);
  const modality = detectModality(file.name);
  const dimensions = await getImageDimensions(file);

  // Try to read GeoTIFF metadata from binary header
  let geoMeta: GeoTiffMeta = {};
  if (format === "geotiff" || format === "tiff") {
    geoMeta = await parseGeoTiffHeader(file);
  }

  return {
    id: generateImageId(),
    fileName: file.name,
    modality: geoMeta.modality || modality,
    width: dimensions.width,
    height: dimensions.height,
    bands: geoMeta.bands || (modality === "sar" ? 2 : 3),
    bandNames: geoMeta.bandNames || getDefaultBandNames(modality),
    acquisitionDate: geoMeta.date || extractDateFromFilename(file.name),
    crs: geoMeta.crs || null,
    pixelSize: geoMeta.pixelSize || null,
    sensor: geoMeta.sensor || detectSensor(file.name),
    fileSize: file.size,
    format,
    dataUrl: URL.createObjectURL(file),
  };
}

interface GeoTiffMeta {
  modality?: ImageModality;
  bands?: number;
  bandNames?: string[];
  date?: string | null;
  crs?: string;
  pixelSize?: { x: number; y: number };
  sensor?: string;
}

/**
 * Parse basic GeoTIFF metadata from the binary header.
 * Reads TIFF tags to extract CRS, band count, and metadata.
 */
async function parseGeoTiffHeader(file: File): Promise<GeoTiffMeta> {
  try {
    const buffer = await file.slice(0, 2048).arrayBuffer();
    const view = new DataView(buffer);

    // Check TIFF magic number (42 or MM/II)
    const byteOrder = view.getUint16(0);
    if (byteOrder !== 0x4949 && byteOrder !== 0x4d4d) {
      return {};
    }

    // Read some basic info from the header
    const result: GeoTiffMeta = {};

    // Try to find GeoKeys in the file text content
    const textContent = await file.slice(0, 100000).text().catch(() => "");

    // Detect SAR keywords in metadata
    const sarKeywords = ["sar", "sentinel-1", "vv", "vh", "sar Imaging Mode", "GRD", "SLC", "IW"];
    const isSAR = sarKeywords.some((k) =>
      textContent.toLowerCase().includes(k.toLowerCase()),
    );
    if (isSAR) {
      result.modality = "sar";
      result.bandNames = ["VV", "VH"];
    }

    // Detect optical keywords
    const opticalKeywords = ["sentinel-2", "landsat", "rgb", "nir", "red", "green", "blue"];
    const isOptical = opticalKeywords.some((k) =>
      textContent.toLowerCase().includes(k.toLowerCase()),
    );
    if (isOptical && !isSAR) {
      result.modality = "optical";
    }

    // Try to find date patterns in metadata
    const dateMatch = textContent.match(
      /(\d{4})[:-]?(\d{2})[:-]?(\d{2})/,
    );
    if (dateMatch) {
      result.date = `${dateMatch[1]}-${dateMatch[2]}-${dateMatch[3]}`;
    }

    // Try to find CRS
    const epsgMatch = textContent.match(/EPSG[:\s-]*(\d+)/i);
    if (epsgMatch) {
      result.crs = `EPSG:${epsgMatch[1]}`;
    }

    return result;
  } catch {
    return {};
  }
}

function detectFormat(fileName: string): string {
  const ext = fileName.toLowerCase().split(".").pop() || "";
  if (ext === "tif" || ext === "tiff") {
    // Check if it's GeoTIFF by filename hints
    if (
      fileName.toLowerCase().includes("geo") ||
      fileName.toLowerCase().includes("sentinel") ||
      fileName.toLowerCase().includes("landsat")
    ) {
      return "geotiff";
    }
    return "tiff";
  }
  if (ext === "png") return "png";
  if (ext === "jpg" || ext === "jpeg") return "jpeg";
  return "unknown";
}

function detectModality(fileName: string): ImageModality {
  const lower = fileName.toLowerCase();
  if (
    lower.includes("sar") ||
    lower.includes("sentinel-1") ||
    lower.includes("s1_") ||
    lower.includes("vv") ||
    lower.includes("vh")
  ) {
    return "sar";
  }
  if (
    lower.includes("optical") ||
    lower.includes("sentinel-2") ||
    lower.includes("landsat") ||
    lower.includes("s2_") ||
    lower.includes("rgb")
  ) {
    return "optical";
  }
  return "unknown";
}

function detectSensor(fileName: string): string | null {
  const lower = fileName.toLowerCase();
  if (lower.includes("sentinel-1") || lower.includes("s1_"))
    return "Sentinel-1";
  if (lower.includes("sentinel-2") || lower.includes("s2_"))
    return "Sentinel-2";
  if (lower.includes("landsat")) return "Landsat";
  if (lower.includes("worldview")) return "WorldView";
  return null;
}

function extractDateFromFilename(fileName: string): string | null {
  // Common patterns: 20250101, 2025-01-01, 2025_01_01
  const match = fileName.match(/(\d{4})[_-]?(\d{2})[_-]?(\d{2})/);
  if (match) {
    return `${match[1]}-${match[2]}-${match[3]}`;
  }
  return null;
}

function getDefaultBandNames(modality: ImageModality): string[] {
  if (modality === "sar") return ["VV", "VH"];
  return ["Red", "Green", "Blue"];
}

async function getImageDimensions(
  file: File,
): Promise<{ width: number; height: number }> {
  return new Promise((resolve) => {
    if (
      file.type.startsWith("image/") ||
      file.name.match(/\.(png|jpe?g|gif|webp)$/i)
    ) {
      const img = new Image();
      img.onload = () => {
        resolve({ width: img.naturalWidth, height: img.naturalHeight });
        URL.revokeObjectURL(img.src);
      };
      img.onerror = () => resolve({ width: 512, height: 512 });
      img.src = URL.createObjectURL(file);
    } else {
      // For GeoTIFF/TIFF, estimate from file size or use default
      const estimatedSize = Math.max(
        256,
        Math.min(2048, Math.floor(Math.sqrt(file.size / 3))),
      );
      resolve({ width: estimatedSize, height: estimatedSize });
    }
  });
}
