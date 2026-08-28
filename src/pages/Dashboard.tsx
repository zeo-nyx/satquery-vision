import { useState, useCallback, useRef } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { useNavigate } from "react-router";
import { useAuth } from "@/hooks/use-auth";
import { Button } from "@/components/ui/button";
import {
  Satellite,
  Upload,
  X,
  Search,
  Loader2,
  LogOut,
  ChevronDown,
  ChevronUp,
  Clock,
  Check,
  AlertCircle,
  Map,
  Info,
  Image as ImageIcon,
  Download,
} from "lucide-react";
import type { ImageMetadata, AnalysisResult, ExecutionStep, TaskPlan } from "@/lib/agent/types";
import { getLoadedPipelines } from "@/lib/ml/models";
import { MarkdownText } from "@/components/MarkdownText";
import { extractImageMetadata, validateImageInputs } from "@/lib/image/processing";
import { routeQuery } from "@/lib/agent/router";
import { executeAnalysis } from "@/lib/analysis/pipeline";

const EXAMPLE_QUERIES = [
  { text: "Is there a river in this image?", icon: "🌊", needsImages: true },
  { text: "Describe the land cover in this image.", icon: "🗺️", needsImages: true },
  { text: "Has the built-up area increased?", icon: "🏗️", needsImages: true },
  { text: "Where are the urban structures?", icon: "📍", needsImages: true },
  { text: "What changed between these two images?", icon: "🔄", needsImages: true },
  { text: "Use optical and SAR to identify built-up areas.", icon: "🛰️", needsImages: true },
];

const TEXT_ONLY_QUERIES = [
  { text: "What is the land cover of Mumbai?", icon: "🏙️" },
  { text: "Tell me about deforestation in the Amazon.", icon: "🌲" },
  { text: "What satellite imagery would be useful for monitoring the Sahara?", icon: "🛰️" },
  { text: "Describe the terrain and climate of Iceland.", icon: "🏔️" },
];

export default function Dashboard() {
  const { user, signOut } = useAuth();
  const navigate = useNavigate();
  const fileInputRef = useRef<HTMLInputElement>(null);

  // State
  const [images, setImages] = useState<ImageMetadata[]>([]);
  const [query, setQuery] = useState("");
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisSteps, setAnalysisSteps] = useState<ExecutionStep[]>([]);
  const [currentTaskPlan, setCurrentTaskPlan] = useState<TaskPlan | null>(null);
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [showTrace, setShowTrace] = useState(false);
  const [dragOver, setDragOver] = useState(false);
  const [loadedModels, setLoadedModels] = useState<string[]>([]);
  const [validationWarnings, setValidationWarnings] = useState<string[]>([]);

  const handleSignOut = async () => {
    await signOut();
    navigate("/");
  };

  const handleFiles = useCallback(async (files: FileList | File[]) => {
    const newImages: ImageMetadata[] = [];
    for (const file of Array.from(files)) {
      try {
        const meta = await extractImageMetadata(file);
        newImages.push(meta);
      } catch (err) {
        console.error("Failed to process file:", file.name, err);
      }
    }
    setImages((prev) => [...prev, ...newImages].slice(0, 4)); // Max 4 images
  }, []);

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setDragOver(false);
      if (e.dataTransfer.files.length > 0) {
        handleFiles(e.dataTransfer.files);
      }
    },
    [handleFiles],
  );

  const removeImage = (id: string) => {
    setImages((prev) => prev.filter((img) => img.id !== id));
  };

  const runAnalysis = async (q: string, imgs: ImageMetadata[]) => {
    if (!q.trim()) return;

    setIsAnalyzing(true);
    setResult(null);
    setAnalysisSteps([]);
    setCurrentTaskPlan(null);

    try {
      const taskPlan = routeQuery(q, imgs);
      console.log("[SatQuery] Task plan:", taskPlan);
      const validation = validateImageInputs(imgs, taskPlan.inputType);
      console.log("[SatQuery] Validation:", validation);
      setValidationWarnings(validation.warnings);
      setCurrentTaskPlan(taskPlan);

      console.log("[SatQuery] Starting pipeline...");
      const analysisResult = await executeAnalysis(
        taskPlan,
        imgs,
        q,
        (steps) => setAnalysisSteps([...steps]),
      );
      console.log("[SatQuery] Pipeline result:", analysisResult);

      setResult(analysisResult);
      setLoadedModels(getLoadedPipelines());
    } catch (error) {
      console.error("[SatQuery] Analysis failed:", error);
      setAnalysisSteps((prev) => [
        ...prev,
        {
          step: "Error",
          status: "error",
          detail: error instanceof Error ? error.message : "Analysis failed",
        },
      ]);
    } finally {
      setIsAnalyzing(false);
    }
  };

  const handleAnalyze = () => {
    runAnalysis(query, images);
  };

  const handleExampleQuery = (q: string, needsImages = true) => {
    setQuery(q);
    if (needsImages && images.length === 0) {
      loadDemoImages();
    }
    // For text-only queries, auto-analyze immediately
    if (!needsImages) {
      runAnalysis(q, []);
    }
  };

  const loadDemoImages = async () => {
    // Create synthetic demo images with distinct, analyzable content
    const demoImages: ImageMetadata[] = [
      {
        id: "demo_1",
        fileName: "sentinel2_urban_2025-01-15.tif",
        modality: "optical",
        width: 512,
        height: 512,
        bands: 4,
        bandNames: ["Red", "Green", "Blue", "NIR"],
        acquisitionDate: "2025-01-15",
        crs: "EPSG:4326",
        pixelSize: { x: 10, y: 10 },
        sensor: "Sentinel-2",
        fileSize: 2048000,
        format: "geotiff",
        dataUrl: generateDemoImage("urban"),
      },
      {
        id: "demo_2",
        fileName: "sentinel2_urban_2026-01-20.tif",
        modality: "optical",
        width: 512,
        height: 512,
        bands: 4,
        bandNames: ["Red", "Green", "Blue", "NIR"],
        acquisitionDate: "2026-01-20",
        crs: "EPSG:4326",
        pixelSize: { x: 10, y: 10 },
        sensor: "Sentinel-2",
        fileSize: 2148000,
        format: "geotiff",
        dataUrl: generateDemoImage("urban-grown"),
      },
    ];
    setImages(demoImages);
  };

  const downloadReport = () => {
    if (!result) return;
    const trace = result.executionTrace
      .map((s, i) => `  ${i + 1}. [${s.status.toUpperCase()}] ${s.step}: ${s.detail || ""}${s.duration ? ` (${s.duration}ms)` : ""}`)
      .join("\n");
    const report = [
      "=".repeat(60),
      "SATQUERY AI — ANALYSIS REPORT",
      "=".repeat(60),
      "",
      `Date: ${new Date().toISOString()}`,
      `Query: ${query}`,
      `Images: ${images.map((i) => i.fileName).join(", ")}`,
      "",
      "-".repeat(60),
      "TASK & MODELS",
      "-".repeat(60),
      `Task: ${result.task.replace(/_/g, " ").toUpperCase()}`,
      `Input Type: ${result.inputType.replace(/_/g, " ")}`,
      `Models Used: ${result.modelsUsed.join(", ")}`,
      `Confidence: ${Math.round(result.confidence * 100)}%`,
      "",
      "-".repeat(60),
      "ANSWER",
      "-".repeat(60),
      result.answer.replace(/\*\*/g, ""),
      "",
      "-".repeat(60),
      "EXECUTION TRACE",
      "-".repeat(60),
      trace,
      "",
      "=".repeat(60),
      "Generated by SatQuery AI — Remote Sensing Vision-Language Agent",
      "=".repeat(60),
    ].join("\n");
    const blob = new Blob([report], { type: "text/plain" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `satquery-report-${Date.now()}.txt`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const canAnalyze = query.trim() && !isAnalyzing;

  return (
    <div className="min-h-screen bg-[#F5F0E8] text-[#1A1A2E] font-mono flex flex-col">
      {/* ── HEADER ─────────────────────────────────────────────── */}
      <header className="border-b-[3px] border-[#1A1A2E] bg-white px-6 py-3">
        <div className="mx-auto flex max-w-7xl items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="flex size-9 items-center justify-center border-[2.5px] border-[#1A1A2E] bg-[#FFD166] font-bold">
              <Satellite className="size-4.5" />
            </div>
            <div>
              <span className="font-bold text-base tracking-tight">
                SATQUERY<span className="text-[#EF476F]">AI</span>
              </span>
              <span className="ml-2 nb-badge bg-[#06D6A0] text-white text-[0.6rem]">
                MVP v1
              </span>
              {loadedModels.length > 0 && (
                <span className="nb-badge bg-[#7B68EE] text-white text-[0.55rem] ml-1">
                  {loadedModels.length} ML models loaded
                </span>
              )}
            </div>
          </div>
          <div className="flex items-center gap-3">
            <span className="text-xs text-[#1A1A2E]/50 hidden sm:inline">
              {user?.name || user?.email || "User"}
            </span>
            <Button
              variant="outline"
              size="sm"
              className="nb-btn border-[2px] border-[#1A1A2E] bg-white shadow-[2px_2px_0_#1A1A2E] hover:shadow-[1px_1px_0_#1A1A2E] hover:translate-x-[1px] hover:translate-y-[1px] rounded-none font-bold text-[0.65rem] uppercase tracking-wider"
              onClick={handleSignOut}
            >
              <LogOut className="size-3" />
              Sign Out
            </Button>
          </div>
        </div>
      </header>

      {/* ── MAIN CONTENT ───────────────────────────────────────── */}
      <div className="flex-1 px-4 py-6 sm:px-6">
        <div className="mx-auto max-w-7xl">
          <div className="grid gap-6 lg:grid-cols-[1fr_1fr]">

            {/* ── LEFT: INPUT PANEL ──────────────────────────── */}
            <div className="flex flex-col gap-5">

              {/* Image Upload */}
              <div>
                <div className="mb-2 flex items-center justify-between">
                  <span className="nb-tag bg-[#FFD166]">
                    <ImageIcon className="size-3 inline mr-1" />
                    Images
                  </span>
                  <span className="text-[0.65rem] text-[#1A1A2E]/40 uppercase tracking-wider">
                    {images.length}/4 uploaded
                  </span>
                </div>

                {/* Drop zone */}
                <div
                  onDragOver={(e) => {
                    e.preventDefault();
                    setDragOver(true);
                  }}
                  onDragLeave={() => setDragOver(false)}
                  onDrop={handleDrop}
                  onClick={() => fileInputRef.current?.click()}
                  className={`nb-card cursor-pointer bg-white p-6 text-center transition-all ${
                    dragOver
                      ? "bg-[#FFD166]/20 border-[#FFD166] shadow-[4px_4px_0_#FFD166]"
                      : "hover:bg-[#F5F0E8]"
                  }`}
                >
                  <Upload
                    className={`mx-auto mb-2 size-8 ${
                      dragOver ? "text-[#FFD166]" : "text-[#1A1A2E]/30"
                    }`}
                  />
                  <p className="font-bold text-sm">
                    Drop satellite images here
                  </p>
                  <p className="mt-1 text-[0.65rem] text-[#1A1A2E]/40">
                    GeoTIFF, TIFF, PNG, or JPEG
                  </p>
                  <div className="mt-3 flex justify-center gap-2">
                    <span className="nb-badge bg-[#F5F0E8] text-[#1A1A2E]/50">
                      GeoTIFF
                    </span>
                    <span className="nb-badge bg-[#F5F0E8] text-[#1A1A2E]/50">
                      TIFF
                    </span>
                    <span className="nb-badge bg-[#F5F0E8] text-[#1A1A2E]/50">
                      PNG
                    </span>
                    <span className="nb-badge bg-[#F5F0E8] text-[#1A1A2E]/50">
                      JPEG
                    </span>
                  </div>
                  <input
                    ref={fileInputRef}
                    type="file"
                    multiple
                    accept=".tif,.tiff,.png,.jpg,.jpeg"
                    className="hidden"
                    onChange={(e) => {
                      if (e.target.files) handleFiles(e.target.files);
                      e.target.value = "";
                    }}
                  />
                </div>

                {/* Uploaded image thumbnails */}
                <AnimatePresence>
                  {images.length > 0 && (
                    <motion.div
                      initial={{ opacity: 0, height: 0 }}
                      animate={{ opacity: 1, height: "auto" }}
                      exit={{ opacity: 0, height: 0 }}
                      className="mt-3 grid grid-cols-2 gap-3"
                    >
                      {images.map((img) => (
                        <div
                          key={img.id}
                          className="nb-card-sm bg-white p-3 relative group"
                        >
                          <button
                            onClick={() => removeImage(img.id)}
                            className="absolute top-1 right-1 size-5 flex items-center justify-center bg-[#EF476F] text-white border border-[#1A1A2E] opacity-0 group-hover:opacity-100 transition-opacity z-10"
                          >
                            <X className="size-3" />
                          </button>
                          <div className="aspect-square bg-[#1A1A2E]/5 mb-2 overflow-hidden border-[1.5px] border-[#1A1A2E]">
                            {img.dataUrl ? (
                              <img
                                src={img.dataUrl}
                                alt={img.fileName}
                                className="w-full h-full object-cover"
                              />
                            ) : (
                              <div className="w-full h-full flex items-center justify-center text-[#1A1A2E]/20">
                                <Satellite className="size-8" />
                              </div>
                            )}
                          </div>
                          <p className="font-bold text-[0.6rem] truncate mb-1">
                            {img.fileName}
                          </p>
                          <div className="flex flex-wrap gap-1">
                            <span
                              className={`nb-badge text-[0.55rem] ${
                                img.modality === "sar"
                                  ? "bg-[#118AB2] text-white"
                                  : img.modality === "optical"
                                    ? "bg-[#06D6A0] text-white"
                                    : "bg-[#F5F0E8] text-[#1A1A2E]/50"
                              }`}
                            >
                              {img.modality.toUpperCase() || "UNK"}
                            </span>
                            <span className="nb-badge bg-[#F5F0E8] text-[#1A1A2E]/50 text-[0.55rem]">
                              {img.bands}B
                            </span>
                            {img.acquisitionDate && (
                              <span className="nb-badge bg-[#FFD166] text-[#1A1A2E] text-[0.55rem]">
                                {img.acquisitionDate}
                              </span>
                            )}
                            {img.sensor && (
                              <span className="nb-badge bg-[#7B68EE] text-white text-[0.55rem]">
                                {img.sensor}
                              </span>
                            )}
                          </div>
                        </div>
                      ))}
                    </motion.div>
                  )}
                </AnimatePresence>
              </div>

              {/* Validation Warnings */}
              <AnimatePresence>
                {validationWarnings.length > 0 && (
                  <motion.div
                    initial={{ opacity: 0, height: 0 }}
                    animate={{ opacity: 1, height: "auto" }}
                    exit={{ opacity: 0, height: 0 }}
                    className="border-[2px] border-[#FFD166] bg-[#FFD166]/10 p-3"
                  >
                    <p className="font-bold text-xs mb-1">Input Warnings</p>
                    {validationWarnings.map((w, i) => (
                      <p key={i} className="text-[0.6rem] text-[#1A1A2E]/60">• {w}</p>
                    ))}
                  </motion.div>
                )}
              </AnimatePresence>

              {/* Quick-load demo */}
              {images.length === 0 && (
                <Button
                  variant="outline"
                  className="nb-btn border-[2px] border-[#1A1A2E] bg-white shadow-[2px_2px_0_#1A1A2E] hover:shadow-[1px_1px_0_#1A1A2E] hover:translate-x-[1px] hover:translate-y-[1px] rounded-none font-bold text-[0.65rem] uppercase tracking-wider"
                  onClick={loadDemoImages}
                >
                  <Info className="size-3" />
                  Load demo images
                </Button>
              )}

              {/* Query Input */}
              <div>
                <div className="mb-2">
                  <span className="nb-tag bg-[#118AB2] text-white">
                    <Search className="size-3 inline mr-1" />
                    Query
                  </span>
                </div>
                <div className="nb-card bg-white p-1.5">
                  <textarea
                    value={query}
                    onChange={(e) => setQuery(e.target.value)}
                    placeholder='Ask about satellite images or any location... e.g. "Is there a river in this image?" or "What is the land cover of Mumbai?"'
                    rows={3}
                    className="w-full resize-none bg-transparent p-3 text-sm font-mono outline-none placeholder:text-[#1A1A2E]/30"
                    onKeyDown={(e) => {
                      if (e.key === "Enter" && !e.shiftKey && canAnalyze) {
                        e.preventDefault();
                        handleAnalyze();
                      }
                    }}
                  />
                  <div className="flex items-center justify-between border-t-[2px] border-[#1A1A2E]/10 px-3 py-2">
                    <span className="text-[0.6rem] text-[#1A1A2E]/30 uppercase tracking-wider">
                      Enter to analyze
                    </span>
                    <Button
                      className={`nb-btn rounded-none px-6 py-2 font-bold text-xs uppercase tracking-wider ${
                        canAnalyze
                          ? "bg-[#EF476F] text-white border-[2px] border-[#1A1A2E] shadow-[3px_3px_0_#1A1A2E] hover:shadow-[1px_1px_0_#1A1A2E] hover:translate-x-[1px] hover:translate-y-[1px] hover:bg-[#EF476F]"
                          : "bg-[#1A1A2E]/10 text-[#1A1A2E]/30 border-[2px] border-[#1A1A2E]/20 shadow-none cursor-not-allowed"
                      }`}
                      disabled={!canAnalyze}
                      onClick={handleAnalyze}
                    >
                      {isAnalyzing ? (
                        <>
                          <Loader2 className="size-3.5 animate-spin" />
                          Analyzing
                        </>
                      ) : (
                        <>
                          <Search className="size-3.5" />
                          Analyze
                        </>
                      )}
                    </Button>
                  </div>
                </div>
              </div>

              {/* Example Queries */}
              <div>
                <div className="mb-2">
                  <span className="nb-tag bg-[#F5F0E8] text-[#1A1A2E]/50">
                    With images
                  </span>
                </div>
                <div className="flex flex-wrap gap-2">
                  {EXAMPLE_QUERIES.map((ex, i) => (
                    <button
                      key={i}
                      onClick={() => handleExampleQuery(ex.text, ex.needsImages)}
                      className="nb-card-sm bg-white px-3 py-2 text-left text-xs hover:bg-[#FFD166]/20 transition-colors cursor-pointer"
                    >
                      <span className="mr-1.5">{ex.icon}</span>
                      <span className="text-[#1A1A2E]/70">{ex.text}</span>
                    </button>
                  ))}
                </div>
              </div>
              <div>
                <div className="mb-2">
                  <span className="nb-tag bg-[#06D6A0] text-white">
                    No images needed
                  </span>
                </div>
                <div className="flex flex-wrap gap-2">
                  {TEXT_ONLY_QUERIES.map((ex, i) => (
                    <button
                      key={i}
                      onClick={() => handleExampleQuery(ex.text, false)}
                      className="nb-card-sm bg-white px-3 py-2 text-left text-xs hover:bg-[#06D6A0]/10 transition-colors cursor-pointer"
                    >
                      <span className="mr-1.5">{ex.icon}</span>
                      <span className="text-[#1A1A2E]/70">{ex.text}</span>
                    </button>
                  ))}
                </div>
              </div>
            </div>

            {/* ML Model Status */}
            <div className="nb-card bg-white p-4">
              <div className="mb-2 flex items-center gap-2">
                <span className="nb-tag bg-[#7B68EE] text-white">
                  ML Models
                </span>
                <span className="text-[0.6rem] text-[#1A1A2E]/40 uppercase tracking-wider">
                  Transformers.js + ONNX Runtime
                </span>
              </div>
              <div className="grid grid-cols-2 gap-2 text-[0.65rem]">
                {[
                  { name: "CLIP", task: "Zero-shot classification", color: "bg-[#06D6A0]" },
                  { name: "ViT-GPT2", task: "Image captioning", color: "bg-[#FFD166]" },
                  { name: "DETR", task: "Object detection", color: "bg-[#EF476F]" },
                  { name: "ViT", task: "Feature extraction", color: "bg-[#118AB2]" },
                ].map((m) => (
                  <div key={m.name} className="flex items-center gap-2 border-[1.5px] border-[#1A1A2E]/10 p-2">
                    <span className={`size-2 ${m.color} shrink-0`} />
                    <span className="font-bold">{m.name}</span>
                    <span className="text-[#1A1A2E]/40">{m.task}</span>
                  </div>
                ))}
              </div>
              <div className="mt-2 border-t-[1.5px] border-[#1A1A2E]/10 pt-2">
                <p className="text-[0.6rem] text-[#1A1A2E]/30">
                  Models load on first analysis. First query may take 10-30s.
                </p>
                <div className="mt-1.5 flex items-center gap-1.5">
                  <span className="nb-badge bg-[#FFD166] text-[#1A1A2E] text-[0.55rem]">
                    RS Domain Adaptation
                  </span>
                  <span className="text-[0.55rem] text-[#1A1A2E]/30">
                    BigEarthNet taxonomy mapping
                  </span>
                </div>
              </div>
            </div>

            {/* ── RIGHT: RESULTS PANEL ───────────────────────── */}
            <div className="flex flex-col gap-5">
              <div className="mb-2">
                <span className="nb-tag bg-[#EF476F] text-white">
                  Results
                </span>
              </div>

              {/* Empty state */}
              {!isAnalyzing && !result && (
                <div className="nb-card bg-white p-12 text-center">
                  <Satellite className="mx-auto mb-4 size-12 text-[#1A1A2E]/10" />
                  <p className="font-bold text-sm text-[#1A1A2E]/30">
                    No analysis results yet
                  </p>
                  <p className="mt-1 text-xs text-[#1A1A2E]/20">
                    Upload images for analysis, or ask about any location directly
                  </p>
                </div>
              )}

              {/* Processing Steps */}
              <AnimatePresence>
                {isAnalyzing && analysisSteps.length > 0 && (
                  <motion.div
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="nb-card bg-white p-5"
                  >
                    <div className="mb-3 flex items-center gap-2">
                      <Loader2 className="size-4 animate-spin text-[#EF476F]" />
                      <span className="font-bold text-sm">
                        Processing Pipeline
                      </span>
                      {currentTaskPlan && (
                        <span className="nb-badge bg-[#FFD166] text-[#1A1A2E] text-[0.55rem] ml-auto">
                          {currentTaskPlan.task.replace("_", " ")}
                        </span>
                      )}
                    </div>
                    <div className="space-y-2">
                      {analysisSteps.map((step, i) => (
                        <div
                          key={`${step.step}-${i}`}
                          className="flex items-center gap-3 border-l-[3px] border-[#1A1A2E]/10 pl-3"
                        >
                          {step.status === "completed" && (
                            <Check className="size-3.5 text-[#06D6A0] shrink-0" />
                          )}
                          {step.status === "processing" && (
                            <Loader2 className="size-3.5 text-[#FFD166] animate-spin shrink-0" />
                          )}
                          {step.status === "error" && (
                            <AlertCircle className="size-3.5 text-[#EF476F] shrink-0" />
                          )}
                          <div className="min-w-0">
                            <span className="font-bold text-xs">
                              {step.step}
                            </span>
                            {step.detail && (
                              <p className="text-[0.6rem] text-[#1A1A2E]/40 truncate">
                                {step.detail}
                              </p>
                            )}
                          </div>
                          {step.duration && (
                            <span className="ml-auto text-[0.55rem] text-[#1A1A2E]/30 shrink-0">
                              {step.duration}ms
                            </span>
                          )}
                        </div>
                      ))}
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>

              {/* Result */}
              <AnimatePresence>
                {result && (
                  <motion.div
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="flex flex-col gap-4"
                  >
                    {/* Answer Card */}
                    <div className="nb-card bg-white p-6">
                      <div className="mb-4 flex flex-wrap items-center gap-2">
                        <span className="nb-badge bg-[#06D6A0] text-white">
                          <Check className="size-3" />
                          Complete
                        </span>
                        <span className="nb-badge bg-[#FFD166] text-[#1A1A2E]">
                          {result.task.replace("_", " ").toUpperCase()}
                        </span>
                        <span className="nb-badge bg-[#1A1A2E] text-white ml-auto">
                          {Math.round(result.confidence * 100)}% confidence
                        </span>
                      </div>
                      <MarkdownText text={result.answer} className="text-sm" />
                    </div>

                    {/* Change Map */}
                    {result.changeMap && (
                      <div className="nb-card bg-white p-5">
                        <div className="mb-3 flex items-center gap-2">
                          <Map className="size-4" />
                          <span className="font-bold text-sm">Change Map</span>
                          <span className="nb-badge bg-[#118AB2] text-white text-[0.55rem] ml-auto">
                            Visual Evidence
                          </span>
                        </div>
                        <div className="border-[2.5px] border-[#1A1A2E] bg-[#1A1A2E] p-2">
                          <img
                            src={result.changeMap}
                            alt="Change detection map"
                            className="w-full"
                          />
                        </div>
                        <div className="mt-2 flex gap-3 text-[0.6rem]">
                          <span className="flex items-center gap-1">
                            <span className="size-2 bg-[#ff4444] inline-block" />
                            Increase
                          </span>
                          <span className="flex items-center gap-1">
                            <span className="size-2 bg-[#44ff44] inline-block" />
                            Decrease
                          </span>
                          <span className="flex items-center gap-1">
                            <span className="size-2 bg-[#44aaff] inline-block" />
                            Water
                          </span>
                          <span className="flex items-center gap-1">
                            <span className="size-2 bg-[#ffaa00] inline-block" />
                            Mixed
                          </span>
                        </div>
                      </div>
                    )}

                    {/* Bounding Boxes */}
                    {result.boundingBoxes && result.boundingBoxes.length > 0 && (
                      <div className="nb-card bg-white p-5">
                        <div className="mb-3 font-bold text-sm">
                          Detected Regions
                        </div>
                        <div className="space-y-2">
                          {result.boundingBoxes.map((box, i) => (
                            <div
                              key={i}
                              className="flex items-center gap-3 border-[2px] border-[#1A1A2E] p-2.5"
                            >
                              <span
                                className="size-3 shrink-0"
                                style={{ backgroundColor: box.color }}
                              />
                              <span className="font-bold text-xs flex-1">
                                {box.label}
                              </span>
                              <span className="text-[0.6rem] text-[#1A1A2E]/40">
                                ({box.x}, {box.y}) {box.width}×{box.height}
                              </span>
                              <span className="nb-badge bg-[#F5F0E8] text-[#1A1A2E]/50 text-[0.55rem]">
                                {Math.round(box.confidence * 100)}%
                              </span>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}

                    {/* Execution Trace */}
                    <div className="nb-card bg-white">
                      <button
                        onClick={() => setShowTrace(!showTrace)}
                        className="flex w-full items-center justify-between p-4 cursor-pointer"
                      >
                        <div className="flex items-center gap-2">
                          <Clock className="size-4" />
                          <span className="font-bold text-sm">
                            Execution Trace
                          </span>
                          <span className="nb-badge bg-[#F5F0E8] text-[#1A1A2E]/50 text-[0.55rem]">
                            {result.executionTrace.length} steps
                          </span>
                        </div>
                        {showTrace ? (
                          <ChevronUp className="size-4" />
                        ) : (
                          <ChevronDown className="size-4" />
                        )}
                      </button>
                      <AnimatePresence>
                        {showTrace && (
                          <motion.div
                            initial={{ height: 0, opacity: 0 }}
                            animate={{ height: "auto", opacity: 1 }}
                            exit={{ height: 0, opacity: 0 }}
                            className="overflow-hidden"
                          >
                            <div className="border-t-[2px] border-[#1A1A2E]/10 p-4 space-y-2">
                              {result.executionTrace.map((step, i) => (
                                <div
                                  key={`${step.step}-${i}`}
                                  className="flex items-center gap-3 text-xs"
                                >
                                  <span className="size-5 flex items-center justify-center border-[1.5px] border-[#1A1A2E] font-bold text-[0.6rem] shrink-0">
                                    {i + 1}
                                  </span>
                                  <span className="font-bold min-w-[120px]">
                                    {step.step}
                                  </span>
                                  {step.status === "completed" && (
                                    <Check className="size-3 text-[#06D6A0] shrink-0" />
                                  )}
                                  {step.status === "error" && (
                                    <AlertCircle className="size-3 text-[#EF476F] shrink-0" />
                                  )}
                                  <span className="text-[#1A1A2E]/40 flex-1 truncate">
                                    {step.detail}
                                  </span>
                                  {step.duration && (
                                    <span className="text-[0.55rem] text-[#1A1A2E]/30 shrink-0">
                                      {step.duration}ms
                                    </span>
                                  )}
                                </div>
                              ))}
                              <div className="mt-3 border-t-[2px] border-[#1A1A2E]/10 pt-2 flex justify-between text-[0.6rem] text-[#1A1A2E]/40">
                                <span>
                                  Models: {result.modelsUsed.join(", ")}
                                </span>
                                <span>
                                  Input: {result.inputType.replace("_", " ")}
                                </span>
                              </div>
                            </div>
                          </motion.div>
                        )}
                      </AnimatePresence>
                    </div>

                    {/* Actions */}
                    <div className="flex gap-3">
                      <Button
                        variant="outline"
                        className="nb-btn border-[2px] border-[#1A1A2E] bg-white shadow-[2px_2px_0_#1A1A2E] hover:shadow-[1px_1px_0_#1A1A2E] hover:translate-x-[1px] hover:translate-y-[1px] rounded-none font-bold text-xs uppercase tracking-wider"
                        onClick={downloadReport}
                      >
                        <Download className="size-3" />
                        Download Report
                      </Button>
                      <Button
                        variant="outline"
                        className="nb-btn border-[2px] border-[#1A1A2E] bg-white shadow-[2px_2px_0_#1A1A2E] hover:shadow-[1px_1px_0_#1A1A2E] hover:translate-x-[1px] hover:translate-y-[1px] rounded-none font-bold text-xs uppercase tracking-wider"
                        onClick={() => {
                          setResult(null);
                          setAnalysisSteps([]);
                          setCurrentTaskPlan(null);
                          setShowTrace(false);
                        }}
                      >
                        New Analysis
                      </Button>
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

/**
 * Generate a synthetic demo satellite image using canvas.
 */
function generateDemoImage(variant: "urban" | "urban-grown"): string {
  const canvas = document.createElement("canvas");
  const size = 512;
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext("2d");
  if (!ctx) return "";

  const isGrown = variant === "urban-grown";

  // ── Base terrain: mix of green/brown ─────────────────────────
  // Seed a simple pseudo-random for deterministic but varied content
  const seed = isGrown ? 42 : 7;
  let rng = seed;
  const rand = () => {
    rng = (rng * 16807) % 2147483647;
    return (rng - 1) / 2147483646;
  };

  // Fill base with vegetation tones
  ctx.fillStyle = isGrown ? "#5a6e42" : "#6b8e5a";
  ctx.fillRect(0, 0, size, size);

  // Vegetation patches (green blobs)
  for (let i = 0; i < 120; i++) {
    const x = rand() * size;
    const y = rand() * size;
    const r = rand() * 25 + 5;
    ctx.globalAlpha = rand() * 0.35 + 0.1;
    ctx.fillStyle = rand() > 0.5 ? `hsl(${100 + rand() * 30}, ${40 + rand() * 20}%, ${25 + rand() * 20}%)` : `hsl(${30 + rand() * 20}, ${20 + rand() * 15}%, ${30 + rand() * 15}%)`;
    ctx.beginPath();
    ctx.arc(x, y, r, 0, Math.PI * 2);
    ctx.fill();
  }
  ctx.globalAlpha = 1;

  // ── Water body (top-left) ────────────────────────────────────
  ctx.fillStyle = "#1a3d6e";
  ctx.globalAlpha = 0.7;
  ctx.beginPath();
  ctx.ellipse(70, 60, 55 + (isGrown ? -8 : 0), 28, 0.2, 0, Math.PI * 2);
  ctx.fill();
  // Water edge highlight
  ctx.fillStyle = "#2a5d9e";
  ctx.globalAlpha = 0.4;
  ctx.beginPath();
  ctx.ellipse(75, 65, 40, 18, 0.2, 0, Math.PI * 2);
  ctx.fill();
  ctx.globalAlpha = 1;

  // ── River/stream (bottom portion) ─────────────────────────────
  ctx.strokeStyle = "#1a3d6e";
  ctx.lineWidth = 6;
  ctx.globalAlpha = 0.5;
  ctx.beginPath();
  ctx.moveTo(0, 400);
  ctx.bezierCurveTo(100, 380, 200, 420, 300, 390);
  ctx.bezierCurveTo(400, 360, 450, 380, 512, 370);
  ctx.stroke();
  ctx.globalAlpha = 1;

  // ── Urban zone (center-right) ────────────────────────────────
  const urbanX = isGrown ? 200 : 250;
  const urbanY = isGrown ? 160 : 200;
  const urbanW = isGrown ? 220 : 160;
  const urbanH = isGrown ? 200 : 140;

  // Urban base (gray)
  ctx.fillStyle = "#7a7a7a";
  ctx.globalAlpha = 0.5;
  ctx.fillRect(urbanX, urbanY, urbanW, urbanH);
  ctx.globalAlpha = 1;

  // Road grid
  ctx.strokeStyle = "#999";
  ctx.lineWidth = 1.5;
  ctx.globalAlpha = 0.5;
  for (let x = urbanX; x < urbanX + urbanW; x += 18) {
    ctx.beginPath();
    ctx.moveTo(x, urbanY);
    ctx.lineTo(x, urbanY + urbanH);
    ctx.stroke();
  }
  for (let y = urbanY; y < urbanY + urbanH; y += 18) {
    ctx.beginPath();
    ctx.moveTo(urbanX, y);
    ctx.lineTo(urbanX + urbanW, y);
    ctx.stroke();
  }
  ctx.globalAlpha = 1;

  // Buildings (dark rectangles)
  ctx.fillStyle = "#4a4a4a";
  ctx.globalAlpha = 0.6;
  const buildingCount = isGrown ? 80 : 40;
  for (let i = 0; i < buildingCount; i++) {
    const bx = urbanX + 5 + rand() * (urbanW - 15);
    const by = urbanY + 5 + rand() * (urbanH - 15);
    const bw = 6 + rand() * 8;
    const bh = 6 + rand() * 8;
    ctx.fillRect(bx, by, bw, bh);
  }
  ctx.globalAlpha = 1;

  // ── New development in "grown" variant ───────────────────────
  if (isGrown) {
    // New buildings in NE corner
    ctx.fillStyle = "#5a5a5a";
    ctx.globalAlpha = 0.55;
    for (let i = 0; i < 25; i++) {
      const bx = 370 + rand() * 120;
      const by = 20 + rand() * 100;
      ctx.fillRect(bx, by, 8 + rand() * 10, 8 + rand() * 10);
    }
    ctx.globalAlpha = 1;

    // Cleared land (brown patches where vegetation was)
    ctx.fillStyle = "#8b7355";
    ctx.globalAlpha = 0.5;
    for (let i = 0; i < 8; i++) {
      const bx = 380 + rand() * 100;
      const by = 130 + rand() * 80;
      ctx.fillRect(bx, by, 20 + rand() * 30, 15 + rand() * 20);
    }
    ctx.globalAlpha = 1;
  }

  // ── Soil/agriculture (bottom-left) ───────────────────────────
  ctx.fillStyle = "#a08050";
  ctx.globalAlpha = 0.35;
  const soilW = isGrown ? 90 : 120;
  ctx.fillRect(20, 300, soilW, 80);
  ctx.globalAlpha = 1;

  // Label
  ctx.fillStyle = "rgba(0,0,0,0.6)";
  ctx.fillRect(0, size - 28, size, 28);
  ctx.fillStyle = "#ffffff";
  ctx.font = "bold 11px monospace";
  const label = isGrown
    ? "DEMO — 2026 Post-Development Sentinel-2"
    : "DEMO — 2025 Pre-Development Sentinel-2";
  ctx.fillText(label, 10, size - 10);

  return canvas.toDataURL("image/png");
}
