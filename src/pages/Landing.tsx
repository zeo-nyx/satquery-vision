import { motion } from "framer-motion";
import { useNavigate } from "react-router";
import { Button } from "@/components/ui/button";
import {
  Satellite,
  Brain,
  Map,
  Layers,
  ArrowRight,
  Zap,
  Shield,
  Eye,
  ChevronRight,
} from "lucide-react";

const fadeUp = {
  hidden: { opacity: 0, y: 20 },
  visible: (i: number) => ({
    opacity: 1,
    y: 0,
    transition: { delay: i * 0.1, duration: 0.5, ease: "easeOut" as const },
  }),
};

const features = [
  {
    icon: Brain,
    title: "Agentic Query Router",
    desc: "Intelligent agent classifies your query and routes it to specialized remote-sensing models. No generic LLM — purpose-built pipelines for every task.",
    color: "bg-[#FFD166]",
  },
  {
    icon: Eye,
    title: "Single-Image VQA",
    desc: "Ask questions about any satellite image. Adapted vision-language models trained on remote sensing data deliver accurate, domain-specific answers.",
    color: "bg-[#06D6A0]",
  },
  {
    icon: Map,
    title: "Change Detection",
    desc: "Upload before/after image pairs. The system detects urban growth, deforestation, water level changes, and infrastructure development with pixel-level precision.",
    color: "bg-[#118AB2]",
  },
  {
    icon: Layers,
    title: "Optical + SAR Fusion",
    desc: "Combine optical and synthetic aperture radar data for all-weather analysis. Structural and spectral information fused for higher classification accuracy.",
    color: "bg-[#EF476F]",
  },
  {
    icon: Satellite,
    title: "GeoTIFF Native",
    desc: "Ingests GeoTIFF, TIFF, PNG, and JPEG. Extracts band info, CRS, acquisition dates, and sensor metadata automatically.",
    color: "bg-[#7B68EE]",
  },
  {
    icon: Shield,
    title: "Auditable Traces",
    desc: "Every analysis produces a full execution trace — task selection, model routing, processing steps, confidence scores — for reproducibility and trust.",
    color: "bg-[#FF9F1C]",
  },
];

const steps = [
  {
    num: "01",
    title: "Upload",
    desc: "Drop satellite images — GeoTIFF, TIFF, PNG, or JPEG. The system detects modality, bands, dates, and sensor metadata.",
  },
  {
    num: "02",
    title: "Ask",
    desc: "Ask any question: 'Is there water?', 'What changed?', 'Describe this scene.' The agent understands your intent.",
  },
  {
    num: "03",
    title: "Route",
    desc: "The query router selects specialized remote-sensing models — VQA, captioning, change detection, or optical-SAR fusion.",
  },
  {
    num: "04",
    title: "Analyze",
    desc: "Models process your images through domain-adapted pipelines. Results are fused with confidence scoring.",
  },
  {
    num: "05",
    title: "Discover",
    desc: "Get answers with visual evidence — change maps, bounding boxes, and a complete execution trace for every result.",
  },
];

export default function Landing() {
  const navigate = useNavigate();

  return (
    <div className="min-h-screen bg-[#F5F0E8] text-[#1A1A2E] font-mono">
      {/* ── NAV ──────────────────────────────────────────────────── */}
      <nav className="sticky top-0 z-50 border-b-[3px] border-[#1A1A2E] bg-[#F5F0E8] px-6 py-4">
        <div className="mx-auto flex max-w-6xl items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="flex size-10 items-center justify-center border-[2.5px] border-[#1A1A2E] bg-[#FFD166] font-bold">
              <Satellite className="size-5" />
            </div>
            <span className="text-lg font-bold tracking-tight">
              SATQUERY<span className="text-[#EF476F]">AI</span>
            </span>
          </div>
          <div className="flex items-center gap-3">
            <Button
              variant="outline"
              className="nb-btn border-[2.5px] border-[#1A1A2E] bg-white shadow-[3px_3px_0_#1A1A2E] hover:shadow-[1px_1px_0_#1A1A2E] hover:translate-x-[2px] hover:translate-y-[2px] rounded-none font-bold uppercase text-xs tracking-wider"
              onClick={() => navigate("/auth")}
            >
              Sign In
            </Button>
            <Button
              className="nb-btn bg-[#1A1A2E] text-white border-[2.5px] border-[#1A1A2E] shadow-[3px_3px_0_#1A1A2E] hover:shadow-[1px_1px_0_#1A1A2E] hover:translate-x-[2px] hover:translate-y-[2px] hover:bg-[#1A1A2E] rounded-none font-bold uppercase text-xs tracking-wider"
              onClick={() => navigate("/auth")}
            >
              Get Started
              <ArrowRight className="size-3.5" />
            </Button>
          </div>
        </div>
      </nav>

      {/* ── HERO ─────────────────────────────────────────────────── */}
      <section className="border-b-[3px] border-[#1A1A2E] px-6 py-20 md:py-28">
        <div className="mx-auto max-w-6xl">
          <motion.div
            initial="hidden"
            animate="visible"
            variants={fadeUp}
            custom={0}
            className="mb-6 inline-flex items-center gap-2 border-[2.5px] border-[#1A1A2E] bg-[#FFD166] px-4 py-2 font-bold text-xs uppercase tracking-widest shadow-[3px_3px_0_#1A1A2E]"
          >
            <Zap className="size-3.5" />
            Remote Sensing Vision-Language Agent
          </motion.div>

          <motion.h1
            initial="hidden"
            animate="visible"
            variants={fadeUp}
            custom={1}
            className="text-4xl font-black leading-[1.1] tracking-tight md:text-6xl lg:text-7xl"
          >
            Ask questions.
            <br />
            <span className="bg-[#1A1A2E] px-3 py-1 text-white inline-block mt-2">
              Get satellite answers.
            </span>
          </motion.h1>

          <motion.p
            initial="hidden"
            animate="visible"
            variants={fadeUp}
            custom={2}
            className="mt-8 max-w-2xl text-lg leading-relaxed text-[#1A1A2E]/70"
          >
            SatQuery AI is an agentic system that routes your queries to
            specialized remote-sensing models — VQA, captioning, change detection,
            and optical-SAR fusion — then combines their results with auditable
            evidence.
          </motion.p>

          <motion.div
            initial="hidden"
            animate="visible"
            variants={fadeUp}
            custom={3}
            className="mt-10 flex flex-wrap gap-4"
          >
            <Button
              className="nb-btn bg-[#EF476F] text-white border-[2.5px] border-[#1A1A2E] shadow-[4px_4px_0_#1A1A2E] hover:shadow-[2px_2px_0_#1A1A2E] hover:translate-x-[2px] hover:translate-y-[2px] hover:bg-[#EF476F] rounded-none px-8 py-6 text-base font-bold"
              onClick={() => navigate("/auth")}
            >
              Launch SatQuery AI
              <ArrowRight className="size-4" />
            </Button>
            <Button
              variant="outline"
              className="nb-btn bg-white border-[2.5px] border-[#1A1A2E] shadow-[4px_4px_0_#1A1A2E] hover:shadow-[2px_2px_0_#1A1A2E] hover:translate-x-[2px] hover:translate-y-[2px] rounded-none px-8 py-6 text-base font-bold"
              onClick={() => document.getElementById("architecture")?.scrollIntoView({ behavior: "smooth" })}
            >
              View Architecture
            </Button>
          </motion.div>

          {/* Architecture preview */}
          <motion.div
            initial="hidden"
            animate="visible"
            variants={fadeUp}
            custom={4}
            className="mt-16 border-[3px] border-[#1A1A2E] bg-white p-6 shadow-[6px_6px_0_#1A1A2E]"
          >
            <div className="font-bold text-xs uppercase tracking-widest text-[#1A1A2E]/50 mb-4">
              System Architecture
            </div>
            <div className="flex flex-col items-center gap-3 font-mono text-sm">
              <div className="border-[2.5px] border-[#1A1A2E] bg-[#FFD166] px-6 py-3 font-bold shadow-[3px_3px_0_#1A1A2E]">
                USER — Image(s) + Question
              </div>
              <ChevronRight className="size-5 rotate-90 text-[#1A1A2E]" />
              <div className="border-[2.5px] border-[#1A1A2E] bg-[#1A1A2E] px-8 py-3 font-bold text-white shadow-[3px_3px_0_#1A1A2E]">
                SATQUERY AGENT — Query Router
              </div>
              <ChevronRight className="size-5 rotate-90 text-[#1A1A2E]" />
              <div className="flex flex-wrap justify-center gap-3">
                <div className="border-[2px] border-[#1A1A2E] bg-[#06D6A0] px-4 py-2 font-bold text-xs shadow-[2px_2px_0_#1A1A2E]">
                  VQA Models
                </div>
                <div className="border-[2px] border-[#1A1A2E] bg-[#118AB2] px-4 py-2 font-bold text-xs shadow-[2px_2px_0_#1A1A2E] text-white">
                  Change Models
                </div>
                <div className="border-[2px] border-[#1A1A2E] bg-[#EF476F] px-4 py-2 font-bold text-xs shadow-[2px_2px_0_#1A1A2E] text-white">
                  Optical+SAR
                </div>
              </div>
              <ChevronRight className="size-5 rotate-90 text-[#1A1A2E]" />
              <div className="border-[2.5px] border-[#1A1A2E] bg-[#7B68EE] px-8 py-3 font-bold text-white shadow-[3px_3px_0_#1A1A2E]">
                Evidence Fusion → Answer + Confidence + Trace
              </div>
            </div>
          </motion.div>
        </div>
      </section>

      {/* ── FEATURES ────────────────────────────────────────────── */}
      <section className="border-b-[3px] border-[#1A1A2E] px-6 py-20">
        <div className="mx-auto max-w-6xl">
          <motion.div
            initial="hidden"
            whileInView="visible"
            viewport={{ once: true, margin: "-100px" }}
            variants={fadeUp}
            custom={0}
            className="mb-4"
          >
            <span className="nb-tag bg-[#FFD166]">Capabilities</span>
          </motion.div>
          <motion.h2
            initial="hidden"
            whileInView="visible"
            viewport={{ once: true, margin: "-100px" }}
            variants={fadeUp}
            custom={1}
            className="mb-12 text-3xl font-black md:text-4xl"
          >
            Built for remote sensing.
            <br />
            Not a generic chatbot.
          </motion.h2>

          <div className="grid gap-5 md:grid-cols-2 lg:grid-cols-3">
            {features.map((f, i) => (
              <motion.div
                key={f.title}
                initial="hidden"
                whileInView="visible"
                viewport={{ once: true, margin: "-50px" }}
                variants={fadeUp}
                custom={i + 2}
                className="nb-card bg-white p-6 hover:translate-x-[2px] hover:translate-y-[2px] hover:shadow-[2px_2px_0_#1A1A2E] transition-all"
              >
                <div
                  className={`mb-4 flex size-11 items-center justify-center border-[2px] border-[#1A1A2E] ${f.color}`}
                >
                  <f.icon className="size-5" />
                </div>
                <h3 className="mb-2 font-bold text-base">{f.title}</h3>
                <p className="text-sm leading-relaxed text-[#1A1A2E]/65">
                  {f.desc}
                </p>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* ── HOW IT WORKS ────────────────────────────────────────── */}
      <section className="border-b-[3px] border-[#1A1A2E] bg-white px-6 py-20">
        <div className="mx-auto max-w-6xl">
          <motion.div
            initial="hidden"
            whileInView="visible"
            viewport={{ once: true, margin: "-100px" }}
            variants={fadeUp}
            custom={0}
            className="mb-4"
          >
            <span className="nb-tag bg-[#06D6A0]">Process</span>
          </motion.div>
          <motion.h2
            initial="hidden"
            whileInView="visible"
            viewport={{ once: true, margin: "-100px" }}
            variants={fadeUp}
            custom={1}
            className="mb-12 text-3xl font-black md:text-4xl"
          >
            Five steps from image to insight.
          </motion.h2>

          <div className="grid gap-0 md:grid-cols-5">
            {steps.map((s, i) => (
              <motion.div
                key={s.num}
                initial="hidden"
                whileInView="visible"
                viewport={{ once: true, margin: "-50px" }}
                variants={fadeUp}
                custom={i + 2}
                className={`border-[3px] border-[#1A1A2E] p-6 ${
                  i < 4 ? "border-r-0 md:border-r-[3px]" : ""
                } ${i < steps.length - 1 ? "border-b-[3px] md:border-b-0" : ""}`}
              >
                <div className="mb-3 font-black text-3xl text-[#EF476F]">
                  {s.num}
                </div>
                <h3 className="mb-2 font-bold text-base">{s.title}</h3>
                <p className="text-xs leading-relaxed text-[#1A1A2E]/60">
                  {s.desc}
                </p>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* ── EXAMPLE QUERIES ─────────────────────────────────────── */}
      <section className="border-b-[3px] border-[#1A1A2E] px-6 py-20">
        <div className="mx-auto max-w-6xl">
          <motion.div
            initial="hidden"
            whileInView="visible"
            viewport={{ once: true, margin: "-100px" }}
            variants={fadeUp}
            custom={0}
            className="mb-4"
          >
            <span className="nb-tag bg-[#118AB2] text-white">Examples</span>
          </motion.div>
          <motion.h2
            initial="hidden"
            whileInView="visible"
            viewport={{ once: true, margin: "-100px" }}
            variants={fadeUp}
            custom={1}
            className="mb-12 text-3xl font-black md:text-4xl"
          >
            One system, many tasks.
          </motion.h2>

          <div className="grid gap-5 md:grid-cols-2">
            {[
              {
                query: '"Is there a river in this image?"',
                task: "VQA",
                color: "bg-[#06D6A0]",
                result:
                  "Yes — a permanent water body is visible in the northern sector with low NIR reflectance consistent with a river.",
              },
              {
                query: '"Describe what you see in this satellite scene."',
                task: "Captioning",
                color: "bg-[#FFD166]",
                result:
                  "Mixed-use landscape with urban zones, agricultural parcels, drainage network, and moderate vegetation cover.",
              },
              {
                query: '"Has the built-up area increased?"',
                task: "Change Detection",
                color: "bg-[#118AB2]",
                result:
                  "Yes. Built-up area increased by 18.5%, primarily in the northeastern portion, with correlated vegetation loss.",
              },
              {
                query: '"Use optical and SAR together to identify water."',
                task: "Optical+SAR Fusion",
                color: "bg-[#EF476F]",
                result:
                  "Cross-modal fusion identifies water bodies at 93.4% accuracy — low optical reflectance + specular SAR return.",
              },
            ].map((ex, i) => (
              <motion.div
                key={i}
                initial="hidden"
                whileInView="visible"
                viewport={{ once: true, margin: "-50px" }}
                variants={fadeUp}
                custom={i + 2}
                className="nb-card bg-white p-6"
              >
                <div className="mb-3 flex items-center justify-between">
                  <span className={`nb-badge ${ex.color} text-white`}>
                    {ex.task}
                  </span>
                </div>
                <div className="mb-3 border-l-[3px] border-[#1A1A2E] pl-3 font-bold text-sm italic">
                  {ex.query}
                </div>
                <div className="text-sm leading-relaxed text-[#1A1A2E]/70">
                  → {ex.result}
                </div>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* ── CTA ─────────────────────────────────────────────────── */}
      <section className="px-6 py-20">
        <div className="mx-auto max-w-6xl">
          <motion.div
            initial="hidden"
            whileInView="visible"
            viewport={{ once: true }}
            variants={fadeUp}
            custom={0}
            className="border-[3px] border-[#1A1A2E] bg-[#1A1A2E] p-12 text-center shadow-[8px_8px_0_#EF476F]"
          >
            <h2 className="mb-4 text-3xl font-black text-white md:text-4xl">
              Ready to analyze satellite imagery?
            </h2>
            <p className="mx-auto mb-8 max-w-lg text-[#F5F0E8]/70">
              Upload images, ask questions, get evidence-backed answers with
              visual change maps and execution traces.
            </p>
            <Button
              className="nb-btn bg-[#EF476F] text-white border-[2.5px] border-white shadow-[4px_4px_0_white] hover:shadow-[2px_2px_0_white] hover:translate-x-[2px] hover:translate-y-[2px] hover:bg-[#EF476F] rounded-none px-10 py-6 text-base font-bold"
              onClick={() => navigate("/auth")}
            >
              Launch SatQuery AI
              <ArrowRight className="size-4" />
            </Button>
          </motion.div>
        </div>
      </section>

      {/* ── FOOTER ──────────────────────────────────────────────── */}
      <footer className="border-t-[3px] border-[#1A1A2E] px-6 py-8">
        <div className="mx-auto flex max-w-6xl flex-col items-center justify-between gap-4 md:flex-row">
          <div className="flex items-center gap-2 font-bold text-sm">
            <div className="flex size-7 items-center justify-center border-[2px] border-[#1A1A2E] bg-[#FFD166]">
              <Satellite className="size-3.5" />
            </div>
            SATQUERY<span className="text-[#EF476F]">AI</span>
          </div>
          <p className="text-xs text-[#1A1A2E]/50">
            Remote Sensing Vision-Language Agent • Built with domain-adapted models
          </p>
        </div>
      </footer>
    </div>
  );
}
