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
} from "lucide-react";
import type { ImageMetadata, AnalysisResult, ExecutionStep, TaskPlan } from "@/lib/agent/types";
import { extractImageMetadata } from "@/lib/image/processing";
import { routeQuery } from "@/lib/agent/router";
import { executeAnalysis } from "@/lib/analysis/pipeline";

const EXAMPLE_QUERIES = [
  { text: "Is there a river in this image?", icon: "🌊" },
  { text: "Describe the land cover in this image.", icon: "🗺️" },
  { text: "Has the built-up area increased?", icon: "🏗️" },
  { text: "Where are the urban structures?", icon: "📍" },
  { text: "What changed between these two images?", icon: "🔄" },
  { text: "Use optical and SAR to identify built-up areas.", icon: "🛰️" },
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

  const handleAnalyze = async () => {
    if (images.length === 0 || !query.trim()) return;

    setIsAnalyzing(true);
    setResult(null);
    setAnalysisSteps([]);
    setCurrentTaskPlan(null);

    try {
      // Step 1: Route the query
      const taskPlan = routeQuery(query, images);
      setCurrentTaskPlan(taskPlan);

      // Step 2: Execute the analysis pipeline
      const analysisResult = await executeAnalysis(
        taskPlan,
        images,
        query,
        (steps) => setAnalysisSteps([...steps]),
      );

      setResult(analysisResult);
    } catch (error) {
      console.error("Analysis failed:", error);
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

  const handleExampleQuery = (q: string) => {
    setQuery(q);
    if (images.length === 0) {
      // Auto-load demo images if none uploaded
      loadDemoImages();
    }
  };

  const loadDemoImages = async () => {
    // Create synthetic demo image metadata
    const demoImages: ImageMetadata[] = [
      {
        id: "demo_1",
        fileName: "sentinel2_rgb_2025-01-15.tif",
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
        dataUrl: generateDemoImage("#4a7c59", "#2d5a3f", "#8fbc8f"),
      },
      {
        id: "demo_2",
        fileName: "sentinel2_rgb_2026-01-20.tif",
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
        dataUrl: generateDemoImage("#8b7355", "#c4a882", "#6b8e6b"),
      },
    ];
    setImages(demoImages);
  };

  const canAnalyze = images.length > 0 && query.trim() && !isAnalyzing;

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
                    placeholder='Ask a question about your satellite images... e.g. "Is there a river in this image?"'
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
                    Try these
                  </span>
                </div>
                <div className="flex flex-wrap gap-2">
                  {EXAMPLE_QUERIES.map((ex, i) => (
                    <button
                      key={i}
                      onClick={() => handleExampleQuery(ex.text)}
                      className="nb-card-sm bg-white px-3 py-2 text-left text-xs hover:bg-[#FFD166]/20 transition-colors cursor-pointer"
                    >
                      <span className="mr-1.5">{ex.icon}</span>
                      <span className="text-[#1A1A2E]/70">{ex.text}</span>
                    </button>
                  ))}
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
                    Upload images and ask a question to begin
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
                      <p className="text-sm leading-relaxed whitespace-pre-line">
                        {result.answer}
                      </p>
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

                    {/* Reset */}
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
function generateDemoImage(
  color1: string,
  color2: string,
  accent: string,
): string {
  const canvas = document.createElement("canvas");
  const size = 512;
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext("2d");
  if (!ctx) return "";

  // Base terrain
  ctx.fillStyle = color1;
  ctx.fillRect(0, 0, size, size);

  // Terrain texture
  for (let i = 0; i < 200; i++) {
    const x = Math.random() * size;
    const y = Math.random() * size;
    const w = Math.random() * 40 + 5;
    const h = Math.random() * 40 + 5;
    ctx.globalAlpha = Math.random() * 0.4 + 0.1;
    ctx.fillStyle = Math.random() > 0.5 ? color2 : accent;
    ctx.fillRect(x, y, w, h);
  }
  ctx.globalAlpha = 1;

  // "Urban" grid pattern
  ctx.strokeStyle = "#888";
  ctx.lineWidth = 1;
  ctx.globalAlpha = 0.3;
  for (let x = 200; x < 400; x += 20) {
    ctx.beginPath();
    ctx.moveTo(x, 150);
    ctx.lineTo(x, 350);
    ctx.stroke();
  }
  for (let y = 150; y < 350; y += 20) {
    ctx.beginPath();
    ctx.moveTo(200, y);
    ctx.lineTo(400, y);
    ctx.stroke();
  }
  ctx.globalAlpha = 1;

  // "Buildings"
  ctx.fillStyle = "#555";
  ctx.globalAlpha = 0.5;
  for (let i = 0; i < 30; i++) {
    const x = 210 + Math.random() * 170;
    const y = 160 + Math.random() * 170;
    ctx.fillRect(x, y, 8, 8);
  }
  ctx.globalAlpha = 1;

  // Water body
  ctx.fillStyle = "#2255aa";
  ctx.globalAlpha = 0.6;
  ctx.beginPath();
  ctx.ellipse(80, 80, 50, 30, 0, 0, Math.PI * 2);
  ctx.fill();
  ctx.globalAlpha = 1;

  // Label
  ctx.fillStyle = "rgba(0,0,0,0.5)";
  ctx.fillRect(0, size - 24, size, 24);
  ctx.fillStyle = "#fff";
  ctx.font = "bold 10px monospace";
  ctx.fillText("DEMO — Simulated Satellite Imagery", 8, size - 8);

  return canvas.toDataURL("image/png");
}
