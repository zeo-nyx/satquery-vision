/**
 * Geo-Context Module
 *
 * Enables text-only queries about specific geographic areas and timelines
 * WITHOUT requiring image uploads. Fetches real data from open sources:
 *   - Nominatim (OpenStreetMap) for geocoding
 *   - Wikipedia/Wikidata for area context
 *   - Open-Meteo for weather/climate context
 *   - Built-in land-cover knowledge base
 *
 * Example: "What is the land cover of Mumbai in 2024?"
 *          "Tell me about deforestation in the Amazon."
 */

// ── Types ───────────────────────────────────────────────────────

export interface GeoContextResult {
  location: {
    name: string;
    country: string;
    lat: number;
    lon: number;
    areaType: string;
  };
  description: string;
  landCover: {
    summary: string;
    details: string[];
  };
  climate: {
    summary: string;
    details: string[];
  };
  remoteSensing: {
    summary: string;
    details: string[];
  };
  sources: string[];
}

// ── Geocoding via Nominatim ─────────────────────────────────────

interface NominatimResult {
  place_id: number;
  display_name: string;
  lat: string;
  lon: string;
  type: string;
  category: string;
  importance: number;
  address?: Record<string, string>;
}

async function geocodeLocation(query: string): Promise<{
  name: string;
  country: string;
  lat: number;
  lon: number;
  areaType: string;
  displayName: string;
} | null> {
  // Extract location from query
  const location = extractLocation(query);
  if (!location) return null;

  try {
    const url = `https://nominatim.openstreetmap.org/search?q=${encodeURIComponent(location)}&format=json&limit=1&addressdetails=1`;
    const resp = await fetch(url, {
      headers: { "User-Agent": "SatQueryAI/1.0 (remote-sensing-analysis)" },
    });

    if (!resp.ok) return null;
    const results: NominatimResult[] = await resp.json();
    if (results.length === 0) return null;

    const r = results[0];
    const addr = r.address || {};
    const country = addr.country || "Unknown";
    const displayName = r.display_name;

    return {
      name: location,
      country,
      lat: parseFloat(r.lat),
      lon: parseFloat(r.lon),
      areaType: r.type || r.category || "place",
      displayName,
    };
  } catch {
    return null;
  }
}

function extractLocation(query: string): string | null {
  const q = query.toLowerCase();

  // Common patterns
  const patterns = [
    /(?:of|in|at|near|around|about)\s+(.+?)(?:\s+(?:in|during|for|from|between)\s+\d{4})?(?:\?|$)/i,
    /(?:what|how|describe|tell)\s+(?:about|me about)\s+(.+?)(?:\s+(?:in|during|for|from|between)\s+\d{4})?(?:\?|$)/i,
    /(.+?)\s+(?:land\s*cover|vegetation|deforestation|urbanization|climate|terrain|landscape|region|area|zone)/i,
    /(?:satellite|remote sensing|aerial)\s+(?:image|imagery|view|analysis)\s+(?:of|for|in)\s+(.+?)(?:\?|$)/i,
  ];

  for (const p of patterns) {
    const m = q.match(p);
    if (m && m[1]) {
      let loc = m[1].trim();
      // Clean up common filler words
      loc = loc.replace(/\b(the|a|an|this|that|these|those)\b/gi, "").trim();
      loc = loc.replace(/\s+/g, " ");
      if (loc.length > 2 && loc.length < 100) {
        return loc;
      }
    }
  }

  return null;
}

// ── Wikipedia Context ────────────────────────────────────────────

async function fetchWikipediaContext(location: string): Promise<{
  summary: string;
  url: string;
} | null> {
  try {
    const url = `https://en.wikipedia.org/api/rest_v1/page/summary/${encodeURIComponent(location)}`;
    const resp = await fetch(url, {
      headers: { "User-Agent": "SatQueryAI/1.0 (remote-sensing-analysis)" },
    });

    if (!resp.ok) return null;
    const data = await resp.json();

    if (data.extract) {
      return {
        summary: data.extract,
        url: data.content_urls?.desktop?.page || `https://en.wikipedia.org/wiki/${encodeURIComponent(location)}`,
      };
    }
    return null;
  } catch {
    return null;
  }
}

// ── Climate Data via Open-Meteo ──────────────────────────────────

async function fetchClimateContext(
  lat: number,
  lon: number,
): Promise<{
  climate: string;
  avgTemp: number;
  precipitation: number;
} | null> {
  try {
    const url = `https://climate-api.open-meteo.com/v1/climate?latitude=${lat}&longitude=${lon}&models=EC_Earth3P_HR&start_date=2020-01-01&end_date=2024-12-31&monthly=temperature_2m_mean,precipitation_sum`;
    const resp = await fetch(url);
    if (!resp.ok) return null;

    const data = await resp.json();
    if (!data.monthly) return null;

    const temps = data.monthly.temperature_2m_mean || [];
    const precip = data.monthly.precipitation_sum || [];
    const avgTemp = temps.length > 0 ? temps.reduce((a: number, b: number) => a + b, 0) / temps.length : 0;
    const totalPrecip = precip.length > 0 ? precip.reduce((a: number, b: number) => a + b, 0) / precip.length : 0;

    let climateType = "temperate";
    if (avgTemp > 25) climateType = "tropical";
    else if (avgTemp > 18) climateType = "subtropical";
    else if (avgTemp < 5) climateType = "cold/boreal";
    else if (totalPrecip < 30) climateType = "arid/semi-arid";

    return {
      climate: climateType,
      avgTemp: Math.round(avgTemp * 10) / 10,
      precipitation: Math.round(totalPrecip),
    };
  } catch {
    return null;
  }
}

// ── Remote Sensing Knowledge Base ────────────────────────────────

const RS_KNOWLEDGE: Record<string, {
  landCover: string[];
  rsNotes: string[];
  typicalSensors: string[];
}> = {
  tropical: {
    landCover: [
      "Dense tropical forest with high canopy cover",
      "High NDVI values (0.4–0.8) indicating active photosynthesis",
      "Frequent cloud cover limiting optical imagery availability",
      "SAR (Sentinel-1) valuable for穿透 cloud cover for monitoring",
    ],
    rsNotes: [
      "Cloud-free optical images are rare - SAR is essential for consistent monitoring",
      "Deforestation creates sharp boundaries visible in both optical and SAR",
      "Biomass estimation requires multi-frequency SAR data",
    ],
    typicalSensors: ["Sentinel-2 (optical)", "Sentinel-1 (SAR)", "Landsat 8/9"],
  },
  arid: {
    landCover: [
      "Sparse vegetation, exposed soil and rock",
      "Low NDVI values (0.0–0.1)",
      "Sand dunes and desert surfaces with high reflectance",
      "Occasional oasis or wadi vegetation",
    ],
    rsNotes: [
      "High brightness can cause saturation in optical sensors",
      "SAR backscatter from sand is typically low due to surface smoothness",
      "Sand migration and desertification can be tracked with multi-temporal data",
    ],
    typicalSensors: ["Sentinel-2", "Landsat 8/9", "MODIS for large-scale monitoring"],
  },
  temperate: {
    landCover: [
      "Mixed forest, agriculture, and urban areas",
      "Seasonal vegetation changes (high NDVI in summer, low in winter)",
      "Moderate cloud cover with good optical imaging windows",
      "Well-developed road and building infrastructure",
    ],
    rsNotes: [
      "Multi-temporal analysis captures seasonal crop cycles",
      "Urban expansion is a primary change detection target",
      "Mixed land cover requires high spatial resolution for accurate classification",
    ],
    typicalSensors: ["Sentinel-2", "Landsat 8/9", "WorldView (high-res)"],
  },
  subtropical: {
    landCover: [
      "Subtropical forests and shrubland",
      "Intensive agriculture (rice, tea, citrus)",
      "Coastal zones with mangroves and coral reefs",
      "Dense urban areas in coastal plains",
    ],
    rsNotes: [
      "Monsoon seasons create distinct wet/dry patterns in SAR data",
      "Coastal erosion and flooding require regular monitoring",
      "Urban heat island effects visible in thermal imagery",
    ],
    typicalSensors: ["Sentinel-2", "Sentinel-1", "Landsat 8/9 (thermal)"],
  },
  cold: {
    landCover: [
      "Boreal forest (taiga) or tundra",
      "Snow and ice cover for much of the year",
      "Permafrost areas with thermokarst features",
      "Low vegetation diversity, slow growth rates",
    ],
    rsNotes: [
      "Snow/ice cover limits optical analysis for many months",
      "SAR is critical for year-round monitoring in high latitudes",
      "Permafrost thaw detection requires InSAR (interferometric SAR)",
    ],
    typicalSensors: ["Sentinel-1 (SAR)", "Sentinel-2 (summer only)", "ICESat-2 (lidar)"],
  },
  urban: {
    landCover: [
      "Dense buildings, roads, and infrastructure",
      "Low vegetation except parks and gardens",
      "High impervious surface fraction",
      "Complex 3D structure visible in SAR (double-bounce)",
    ],
    rsNotes: [
      "Very high resolution needed for individual building detection",
      "SAR double-bounce from buildings creates strong return signals",
      "Urban sprawl detection is a key application of multi-temporal analysis",
      "Night-time lights (VIIRS) can complement daytime imagery",
    ],
    typicalSensors: ["WorldView/GeoEye (very high-res)", "Sentinel-1/2", "Planet (daily)"],
  },
  coastal: {
    landCover: [
      "Coastal waters, beaches, and mangroves",
      "Coral reefs visible in clear shallow water",
      "Tidal flats and estuaries",
      "Coastal urban development and ports",
    ],
    rsNotes: [
      "Water depth estimation uses blue-green band ratios",
      "Tidal cycles affect coastal SAR signatures significantly",
      "Mangrove health can be tracked with NDVI time series",
      "Oil spills and algal blooms detected via spectral anomalies",
    ],
    typicalSensors: ["Sentinel-2", "Sentinel-1", "Landsat 8/9", "MODIS"],
  },
};

// ── Main function ───────────────────────────────────────────────

export async function fetchGeoContext(query: string): Promise<GeoContextResult | null> {
  // Step 1: Geocode the location
  const geo = await geocodeLocation(query);
  if (!geo) return null;

  // Step 2: Fetch Wikipedia context in parallel with climate
  const [wiki, climate] = await Promise.all([
    fetchWikipediaContext(geo.name),
    fetchClimateContext(geo.lat, geo.lon),
  ]);

  // Step 3: Get RS knowledge for this climate type
  const climateType = climate?.climate || "temperate";
  const rsKnowledge = RS_KNOWLEDGE[climateType] || RS_KNOWLEDGE["temperate"];

  // Step 4: Build the result
  const sources: string[] = [];
  const description = wiki?.summary || `Information about ${geo.name}, ${geo.country}.`;
  if (wiki) sources.push(`Wikipedia: ${wiki.url}`);

  return {
    location: {
      name: geo.name,
      country: geo.country,
      lat: geo.lat,
      lon: geo.lon,
      areaType: geo.areaType,
    },
    description,
    landCover: {
      summary: rsKnowledge.landCover[0] || "Mixed land cover",
      details: rsKnowledge.landCover,
    },
    climate: {
      summary: climate
        ? `${climateType} climate with average temperature of ${climate.avgTemp}°C and approximately ${climate.precipitation}mm monthly precipitation.`
        : `${climateType} climate.`,
      details: climate
        ? [
          `Climate type: ${climateType}`,
          `Average temperature: ${climate.avgTemp}°C`,
          `Monthly precipitation: ${climate.precipitation}mm`,
        ]
        : [],
    },
    remoteSensing: {
      summary: rsKnowledge.rsNotes[0] || "Standard remote sensing analysis applies.",
      details: [...rsKnowledge.rsNotes, ...rsKnowledge.typicalSensors.map((s) => `Recommended sensor: ${s}`)],
    },
    sources,
  };
}

// ── Text-only query answer builder ───────────────────────────────

/**
 * Build a layman-friendly answer from geo-context for text-only queries.
 */
export function buildGeoContextAnswer(
  query: string,
  context: GeoContextResult,
): string {
  const q = query.toLowerCase();
  const loc = context.location;
  const parts: string[] = [];

  // Direct answer to the question
  parts.push(`**${loc.name}, ${loc.country}** (coordinates: ${loc.lat.toFixed(2)}°N, ${loc.lon.toFixed(2)}°E)`);
  parts.push("");

  // What this place is
  parts.push(`**About this area:**`);
  parts.push(context.description);
  parts.push("");

  // Land cover
  if (q.includes("land cover") || q.includes("vegetation") || q.includes("terrain") || q.includes("landscape") || q.includes("what")) {
    parts.push(`**Typical land cover and landscape:**`);
    for (const d of context.landCover.details) {
      parts.push(`• ${d}`);
    }
    parts.push("");
  }

  // Climate
  parts.push(`**Climate context:**`);
  parts.push(context.climate.summary);
  parts.push("");

  // Remote sensing analysis
  parts.push(`**What satellite imagery would reveal:**`);
  for (const d of context.remoteSensing.details) {
    parts.push(`• ${d}`);
  }
  parts.push("");

  // Sources
  if (context.sources.length > 0) {
    parts.push(`*Sources: ${context.sources.join(", ")}*`);
  }

  return parts.join("\n");
}

// ── Utility: detect if a query is text-only (no images needed) ───

export function isTextOnlyQuery(query: string): boolean {
  const q = query.toLowerCase();

  // Explicitly asking about a place/area without mentioning "this image"
  const mentionsImage = q.includes("this image") || q.includes("the image") || q.includes("uploaded") || q.includes("attached");
  const mentionsPlace = q.includes(" of ") || q.includes(" in ") || q.includes("about") || q.includes("tell me");

  // Geographic keywords
  const hasGeoTerms = /\b(region|area|city|country|continent|province|state|zone|territory|land|landscape)\b/i.test(q);

  // Known location patterns (country names, city names, etc.)
  const hasLocation = /\b(africa|asia|europe|america|india|china|brazil|amazon|sahara|gobi|himalaya|mumbai|delhi|tokyo|london|paris|new york|california|australia)\b/i.test(q);

  // If asking about a place without referencing an image
  if (!mentionsImage && (hasGeoTerms || hasLocation || mentionsPlace)) {
    return true;
  }

  // Specific RS topics about regions
  if (q.includes("deforestation") || q.includes("urbanization") || q.includes("climate") || q.includes("land use")) {
    if (!mentionsImage) return true;
  }

  return false;
}
