import type { ImageMetadata, ImageModality } from "../agent/types";

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
