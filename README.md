# SatQuery AI

**An agentic vision-language assistant for remote sensing imagery.**

SatQuery AI analyzes satellite images through natural-language queries. It routes each query to specialized models, combines their results, and returns evidence-grounded answers with confidence scores and execution traces.

![Neobrutalism Theme](https://img.shields.io/badge/theme-Neobrutalism%20Minimalism-black?style=flat-square)
![Transformers.js](https://img.shields.io/badge/models-Transformers.js-blue?style=flat-square)
![BigEarthNet](https://img.shields.io/badge/adaptation-BigEarthNet-green?style=flat-square)

---

## Architecture

```
                    USER
                      |
             Image(s) + Question
                      |
                      v
            +------------------+
            |  SatQuery Agent  |
            |  Query Router    |
            +--------+---------+
                     |
          +----------+-----------+
          |          |           |
          v          v          v
       Single     Change      Optical +
       Image      Analysis      SAR
       Models      Models       Models
          |          |           |
          +----------+-----------+
                     v
             Evidence Fusion
                     |
                     v
             Answer + Confidence
             + Execution Trace
```

### Modules

| Module | Description | File |
|--------|-------------|------|
| **Agent/Router** | Classifies queries, selects models | `src/lib/agent/router.ts` |
| **Image Manager** | Accepts GeoTIFF/TIFF/PNG/JPEG, extracts metadata | `src/lib/image/processing.ts` |
| **Pixel Analyzer** | Real pixel-level analysis (colors, NDVI, texture) | `src/lib/image/analyze.ts` |
| **ML Engine** | CLIP + ViT-GPT2 + feature extraction | `src/lib/ml/models.ts`, `src/lib/ml/analysis.ts` |
| **Domain Adaptation** | BigEarthNet taxonomy mapping | `src/lib/ml/domain-adaptation.ts` |
| **Analysis Pipeline** | Orchestrates all models per task type | `src/lib/analysis/pipeline.ts` |
| **Geo Context** | Text-only queries via Nominatim + Wikipedia + Open-Meteo | `src/lib/analysis/geo-context.ts` |
| **Training Pipeline** | Fine-tuning notebooks + data fetchers | `training/` |

### Supported Tasks

1. **Single-image VQA** - Answer questions about one satellite image
2. **Captioning** - Generate natural-language descriptions
3. **Grounding** - Localize and highlight regions
4. **Bi-temporal Change Analysis** - Compare two images from different dates
5. **Cross-modal Optical+SAR** - Joint analysis of optical and radar imagery
6. **Text-only Geographic Context** - Answer questions about areas without images

---

## Quick Start (Web App)

The web app runs on the Freebuff platform. No local setup needed for the frontend.

### Features
- Drag-and-drop image upload
- Real ML inference in the browser (CLIP, ViT-GPT2, feature extraction)
- BigEarthNet domain adaptation
- Interactive execution traces
- Downloadable analysis reports
- Text-only geographic queries (fetches from Wikipedia, Open-Meteo, Nominatim)

---

## Local Development Setup

### Prerequisites
- Node.js 18+ and Bun
- Python 3.10+ (for training pipeline)
- GPU recommended for training (Colab T4 works)

### 1. Clone and Install

```bash
git clone <repository-url>
cd satquery-ai

# Frontend dependencies
bun install

# Python training dependencies
pip install -r requirements.txt
```

### 2. Fetch Training Data

```bash
# Download all datasets (BigEarthNet, RSVQA, CDVQA)
python training/fetch_training_data.py --dataset all --output ./data

# Or just one dataset
python training/fetch_training_data.py --dataset bigearthnet --output ./data --max-samples 1000
```

### 3. Train the Model (Google Colab)

1. Open `training/01_domain_adaptation.ipynb` in Google Colab
2. Upload the notebook to Colab
3. Enable GPU runtime (Runtime > Change runtime type > T4 GPU)
4. Run all cells
5. Download the exported model from `rs-clip-adapted/`

### 4. Run the Web App

```bash
# Start the development server
bun run dev
```

The app will be available at `http://localhost:5173`.

---

## Project Structure

```
satquery-ai/
├── src/
│   ├── components/
│   │   ├── MarkdownText.tsx        # Markdown renderer for answers
│   │   └── ui/                     # shadcn/ui components
│   ├── hooks/
│   │   └── use-auth.ts             # Authentication hook
│   ├── lib/
│   │   ├── agent/
│   │   │   ├── router.ts           # Rule-based query classifier
│   │   │   └── types.ts            # TypeScript types
│   │   ├── analysis/
│   │   │   ├── pipeline.ts         # Main analysis orchestrator
│   │   │   └── geo-context.ts      # Text-only query module
│   │   ├── image/
│   │   │   ├── processing.ts       # Image upload & metadata
│   │   │   └── analyze.ts          # Pixel-level analysis
│   │   ├── ml/
│   │   │   ├── models.ts           # Transformers.js model manager
│   │   │   ├── analysis.ts         # ML analysis engine
│   │   │   └── domain-adaptation.ts # BigEarthNet adaptation
│   │   └── evaluation/
│   │       └── benchmarks.ts       # Evaluation infrastructure
│   ├── pages/
│   │   ├── Landing.tsx             # Neobrutalism landing page
│   │   ├── Dashboard.tsx           # Main analysis workspace
│   │   └── Auth.tsx                # Authentication page
│   ├── App.tsx
│   ├── main.tsx
│   └── index.css                   # Neobrutalism theme
├── training/
│   ├── 01_domain_adaptation.ipynb  # Colab training notebook
│   └── fetch_training_data.py      # Dataset downloader
├── convex/                         # Convex backend
├── requirements.txt                # Python dependencies
├── package.json                    # Node.js dependencies
└── README.md
```

---

## Technology Stack

### Frontend
- **React 18** + **TypeScript** + **Vite**
- **Tailwind CSS** with Neobrutalism theme
- **Framer Motion** for animations
- **shadcn/ui** components

### ML Models (Browser-based)
- **CLIP** (Xenova/clip-vit-base-patch32) - Zero-shot classification
- **ViT-GPT2** (Xenova/vit-gpt2-image-captioning) - Image captioning
- **ViT** (Xenova/vit-base-patch16-224) - Feature extraction
- **Domain-adapted CLIP** - Fine-tuned on BigEarthNet

### Training
- **PyTorch** + **Transformers** (HuggingFace)
- **BigEarthNet** for domain adaptation
- **RSVQA** for VQA evaluation
- **CDVQA** for change detection evaluation

### External APIs (Text-only queries)
- **Nominatim** (OpenStreetMap) for geocoding
- **Wikipedia REST API** for area context
- **Open-Meteo** for climate data

---

## Remote Sensing Domain Adaptation

Per the project spec: *"At least one visual or vision-language component must be fine-tuned or otherwise adapted using BigEarthNet or open source training data."*

SatQuery AI implements domain adaptation through:

1. **BigEarthNet 43-class taxonomy mapping** - CLIP outputs are mapped to CORINE land cover classes
2. **RS-domain prompt engineering** - Zero-shot classification uses satellite-specific prompts
3. **Domain post-processing** - NDVI, spectral signatures, and sensor-aware confidence calibration
4. **Training notebook** - Fine-tunes CLIP text encoder on BigEarthNet image-caption pairs

---

## Evaluation

The system supports evaluation against:

| Benchmark | Task | Metric |
|-----------|------|--------|
| **VRSBench** | Captioning + Grounding | BLEU, CIDEr, IoU |
| **RSVQA** | Visual Question Answering | Accuracy, F1 |
| **CDVQA** | Change-based VQA | Accuracy, BLEU |
| **BigEarthNet** | Scene Classification | Accuracy, F1 |
| **ISRO/SAC** | Full pipeline evaluation | Normalized multi-metric |

Evaluation infrastructure is in `src/lib/evaluation/benchmarks.ts`.

---

## Deployment

### Web (Freebuff Platform)
The app is deployed on Freebuff with Convex backend. Push to deploy.

### Local Training
```bash
# Fetch data
python training/fetch_training_data.py --output ./data

# Open Colab notebook
# Upload training/01_domain_adaptation.ipynb to Google Colab
# Run all cells, download rs-clip-adapted/
```

### Model Export
The training notebook exports models in:
- **PyTorch format** (`.pt` / HuggingFace transformers)
- **ONNX format** (for Transformers.js browser inference)

---

## License

MIT

---

## Acknowledgments

- [BigEarthNet](https://bigearth.net/) - Primary dataset for RS adaptation
- [HuggingFace Transformers.js](https://huggingface.co/docs/transformers.js) - Browser ML inference
- [Nominatim](https://nominatim.openstreetmap.org/) - Geocoding service
- [Open-Meteo](https://open-meteo.com/) - Climate data API
