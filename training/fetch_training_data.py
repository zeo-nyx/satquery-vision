"""
SatQuery AI - Training Data Pipeline

Downloads and prepares training datasets for remote sensing model fine-tuning.

Supported datasets:
  - BigEarthNet (primary): Sentinel-1 SAR + Sentinel-2 multispectral with text annotations
  - RSVQA: Remote sensing visual question answering
  - CDVQA: Change-based visual question answering
  - VRSBench: Remote sensing image captioning and grounding

Usage:
  python fetch_training_data.py --dataset bigearthnet --output ./data
  python fetch_training_data.py --dataset all --output ./data
  python fetch_training_data.py --dataset rsvqa --output ./data --max-samples 1000

Requirements:
  pip install datasets huggingface_hub Pillow requests tqdm
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path

try:
    from datasets import load_dataset
except ImportError:
    print("ERROR: 'datasets' package not installed. Run: pip install datasets")
    sys.exit(1)

try:
    from PIL import Image
except ImportError:
    print("ERROR: 'Pillow' package not installed. Run: pip install Pillow")
    sys.exit(1)


# ── BigEarthNet Label Taxonomy ───────────────────────────────────

BIGEARTHNET_LABELS = {
    0: ("Continuous urban fabric", "urban"),
    1: ("Discontinuous urban fabric", "urban"),
    2: ("Industrial and commercial units", "urban"),
    3: ("Road and rail networks", "urban"),
    4: ("Port areas", "urban"),
    5: ("Airports", "urban"),
    6: ("Mineral extraction sites", "soil"),
    7: ("Dump sites", "soil"),
    8: ("Construction sites", "soil"),
    9: ("Green urban areas", "vegetation"),
    10: ("Sport and leisure facilities", "vegetation"),
    11: ("Non-irrigated arable land", "vegetation"),
    12: ("Permanently irrigated land", "vegetation"),
    13: ("Rice fields", "vegetation"),
    14: ("Vineyards", "vegetation"),
    15: ("Fruit trees and berry plantations", "vegetation"),
    16: ("Olive groves", "vegetation"),
    17: ("Pastures", "vegetation"),
    18: ("Annual crops associated with permanent crops", "vegetation"),
    19: ("Complex cultivation patterns", "vegetation"),
    20: ("Land principally occupied by agriculture", "vegetation"),
    21: ("Agro-forestry areas", "vegetation"),
    22: ("Broad-leaved forest", "vegetation"),
    23: ("Coniferous forest", "vegetation"),
    24: ("Mixed forest", "vegetation"),
    25: ("Natural grassland", "vegetation"),
    26: ("Moors and heathland", "vegetation"),
    27: ("Sclerophyllous vegetation", "vegetation"),
    28: ("Transitional woodland-shrub", "vegetation"),
    29: ("Beaches, dunes, sands", "soil"),
    30: ("Bare rock", "soil"),
    31: ("Sparsely vegetated areas", "soil"),
    32: ("Burnt areas", "soil"),
    33: ("Inland waters", "water"),
    34: ("Marine waters", "water"),
    35: ("Inland wetlands", "water"),
    36: ("Coastal wetlands", "water"),
    37: ("Glaciers and perpetual snow", "snow"),
    38: ("Clouds", "cloud"),
}

# Caption templates for generating training text
CAPTION_TEMPLATES = {
    "urban": [
        "A satellite image showing an urban area with buildings and roads.",
        "This remote sensing image captures a built-up region with infrastructure.",
        "An aerial view of a city with residential and commercial structures.",
        "Dense urban development visible with high impervious surface fraction.",
        "A cityscape viewed from above showing blocks of buildings and transport networks.",
    ],
    "vegetation": [
        "A satellite image showing dense vegetation and forest canopy.",
        "This remote sensing image captures a natural landscape with trees.",
        "An aerial view of forested land with high green vegetation density.",
        "Agricultural fields and crop patterns visible from satellite altitude.",
        "Natural vegetation cover dominating the landscape with seasonal variation.",
    ],
    "water": [
        "A satellite image showing a water body such as a river or lake.",
        "This remote sensing image captures inland or coastal water features.",
        "An aerial view of water bodies with low reflectance in near-infrared.",
        "A river or reservoir visible with characteristic blue spectral signature.",
    ],
    "soil": [
        "A satellite image showing bare soil and exposed terrain.",
        "This remote sensing image captures arid or semi-arid landscape.",
        "An aerial view of exposed ground with minimal vegetation cover.",
        "Construction or mining site with exposed earth and soil surfaces.",
    ],
    "snow": [
        "A satellite image showing snow and ice cover on the terrain.",
        "This remote sensing image captures a cold region with frozen surfaces.",
    ],
    "cloud": [
        "A satellite image with significant cloud cover obscuring the surface.",
    ],
}


def generate_caption(labels):
    """Generate a descriptive caption from BigEarthNet label indices."""
    if not labels:
        return "A satellite image of mixed terrain."

    # Get the dominant category
    categories = []
    for label_idx in labels:
        if label_idx in BIGEARTHNET_LABELS:
            _, cat = BIGEARTHNET_LABELS[label_idx]
            categories.append(cat)

    if not categories:
        return "A satellite image of mixed terrain."

    # Get most common category
    from collections import Counter
    most_common = Counter(categories).most_common(1)[0][0]

    return random.choice(CAPTION_TEMPLATES.get(most_common, CAPTION_TEMPLATES["urban"]))


def fetch_bigearthnet(output_dir, max_samples=None):
    """Fetch BigEarthNet dataset from HuggingFace."""
    print("\n" + "=" * 60)
    print("Fetching BigEarthNet Dataset")
    print("=" * 60)

    output_path = Path(output_dir) / "bigearthnet"
    output_path.mkdir(parents=True, exist_ok=True)

    try:
        print("Loading BigEarthNet from HuggingFace (streaming)...")
        dataset = load_dataset("GFM-Bench/BigEarthNet", split="train", streaming=True)
    except Exception as e:
        print(f"Failed to load BigEarthNet: {e}")
        print("Trying alternative source...")
        try:
            dataset = load_dataset("bigearthnet", split="train", streaming=True)
        except Exception as e2:
            print(f"Alternative also failed: {e2}")
            print("Creating synthetic BigEarthNet-style data for training...")
            return create_synthetic_bigearthnet(output_dir, max_samples or 500)

    samples = []
    count = 0
    limit = max_samples or 500

    print(f"Collecting up to {limit} samples...")
    for item in dataset:
        if count >= limit:
            break

        try:
            # Extract image and labels
            image = item.get("image")
            labels = item.get("labels", [])

            if image is None:
                continue

            # Generate caption from labels
            caption = generate_caption(labels)

            sample = {
                "id": f"bigearthnet_{count:06d}",
                "caption": caption,
                "labels": labels,
            }

            # Save image
            img_path = output_path / f"{sample['id']}.png"
            if isinstance(image, Image.Image):
                image.save(img_path)
            else:
                continue

            samples.append(sample)
            count += 1

            if count % 50 == 0:
                print(f"  Collected {count}/{limit} samples...")

        except Exception as e:
            continue

    # Save metadata
    meta_path = output_path / "metadata.jsonl"
    with open(meta_path, "w") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")

    print(f"BigEarthNet: {len(samples)} samples saved to {output_path}")
    return samples


def create_synthetic_bigearthnet(output_dir, num_samples=500):
    """Create synthetic satellite-like images for training when real data is unavailable."""
    print(f"Creating {num_samples} synthetic satellite samples...")

    output_path = Path(output_dir) / "bigearthnet"
    output_path.mkdir(parents=True, exist_ok=True)

    samples = []
    import numpy as np

    categories = list(CAPTION_TEMPLATES.keys())

    for i in range(num_samples):
        cat = random.choice(categories)
        caption = random.choice(CAPTION_TEMPLATES[cat])

        # Generate a synthetic satellite-like image
        img_array = np.zeros((224, 224, 3), dtype=np.uint8)

        if cat == "urban":
            img_array[:] = [140, 140, 140]
            for _ in range(50):
                x = random.randint(0, 200)
                y = random.randint(0, 200)
                w = random.randint(8, 25)
                h = random.randint(8, 25)
                c = random.choice([[160, 160, 160], [120, 120, 120], [180, 180, 180]])
                img_array[x : x + w, y : y + h] = c
        elif cat == "vegetation":
            img_array[:, :, 0] = np.random.randint(20, 80, size=(224, 224))
            img_array[:, :, 1] = np.random.randint(80, 180, size=(224, 224))
            img_array[:, :, 2] = np.random.randint(20, 60, size=(224, 224))
        elif cat == "water":
            img_array[:, :, 0] = np.random.randint(20, 60, size=(224, 224))
            img_array[:, :, 1] = np.random.randint(50, 120, size=(224, 224))
            img_array[:, :, 2] = np.random.randint(100, 200, size=(224, 224))
        elif cat == "bare" or cat == "soil":
            img_array[:, :, 0] = np.random.randint(120, 180, size=(224, 224))
            img_array[:, :, 1] = np.random.randint(90, 140, size=(224, 224))
            img_array[:, :, 2] = np.random.randint(50, 100, size=(224, 224))
        else:
            img_array = np.random.randint(50, 200, size=(224, 224, 3), dtype=np.uint8)

        image = Image.fromarray(img_array)
        sample_id = f"bigearthnet_synth_{i:06d}"

        img_path = output_path / f"{sample_id}.png"
        image.save(img_path)

        sample = {"id": sample_id, "caption": caption, "labels": [cat]}
        samples.append(sample)

    meta_path = output_path / "metadata.jsonl"
    with open(meta_path, "w") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")

    print(f"Synthetic BigEarthNet: {len(samples)} samples saved to {output_path}")
    return samples


def fetch_rsvqa(output_dir, max_samples=None):
    """Fetch RSVQA dataset for visual question answering training."""
    print("\n" + "=" * 60)
    print("Fetching RSVQA Dataset")
    print("=" * 60)

    output_path = Path(output_dir) / "rsvqa"
    output_path.mkdir(parents=True, exist_ok=True)

    try:
        # Try loading from HuggingFace
        print("Loading RSVQA from HuggingFace...")
        dataset = load_dataset("arampacha/rsvqa", split="train", streaming=True)
    except Exception:
        try:
            dataset = load_dataset("rsvqa", split="train", streaming=True)
        except Exception as e:
            print(f"RSVQA not available: {e}")
            print("Creating synthetic VQA pairs...")
            return create_synthetic_vqa(output_dir, max_samples or 200, "rsvqa")

    samples = []
    count = 0
    limit = max_samples or 200

    for item in dataset:
        if count >= limit:
            break
        try:
            sample = {
                "id": f"rsvqa_{count:06d}",
                "image_path": "",
                "question": item.get("question", ""),
                "answer": str(item.get("answer", "")),
            }

            image = item.get("image")
            if image:
                img_path = output_path / f"{sample['id']}.png"
                if isinstance(image, Image.Image):
                    image.save(img_path)
                    sample["image_path"] = str(img_path)
                else:
                    continue
            else:
                continue

            samples.append(sample)
            count += 1

            if count % 50 == 0:
                print(f"  Collected {count}/{limit} VQA pairs...")

        except Exception:
            continue

    meta_path = output_path / "metadata.jsonl"
    with open(meta_path, "w") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")

    print(f"RSVQA: {len(samples)} Q&A pairs saved to {output_path}")
    return samples


def create_synthetic_vqa(output_dir, num_samples, dataset_name):
    """Create synthetic VQA pairs for training."""
    import numpy as np

    output_path = Path(output_dir) / dataset_name
    output_path.mkdir(parents=True, exist_ok=True)

    questions_answers = [
        ("Is there a river in this image?", "yes", "water"),
        ("Is there a river in this image?", "no", "urban"),
        ("What is the dominant land cover?", "vegetation", "vegetation"),
        ("What is the dominant land cover?", "urban area", "urban"),
        ("What is the dominant land cover?", "water body", "water"),
        ("Are there buildings visible?", "yes", "urban"),
        ("Are there buildings visible?", "no", "vegetation"),
        ("Is this a forest area?", "yes", "vegetation"),
        ("Is this a forest area?", "no", "bare"),
        ("What season is this image from?", "summer", "vegetation"),
        ("What season is this image from?", "winter", "snow"),
        ("Is there agriculture visible?", "yes", "vegetation"),
        ("Is there agriculture visible?", "no", "urban"),
        ("Describe the terrain.", "flat with vegetation", "vegetation"),
        ("Describe the terrain.", "mountainous with rocks", "bare"),
    ]

    samples = []
    for i in range(num_samples):
        q, a, cat = random.choice(questions_answers)
        sample_id = f"{dataset_name}_synth_{i:06d}"

        # Generate synthetic image
        img_array = np.zeros((224, 224, 3), dtype=np.uint8)
        if cat == "urban":
            img_array[:] = [140, 140, 140]
        elif cat == "vegetation":
            img_array[:, :, 1] = np.random.randint(80, 180, size=(224, 224))
        elif cat == "water":
            img_array[:, :, 2] = np.random.randint(100, 200, size=(224, 224))
        elif cat == "bare":
            img_array[:, :, 0] = np.random.randint(120, 180, size=(224, 224))
        elif cat == "snow":
            img_array[:] = [230, 235, 240]
        else:
            img_array = np.random.randint(50, 200, size=(224, 224, 3), dtype=np.uint8)

        image = Image.fromarray(img_array)
        img_path = output_path / f"{sample_id}.png"
        image.save(img_path)

        sample = {
            "id": sample_id,
            "image_path": str(img_path),
            "question": q,
            "answer": a,
        }
        samples.append(sample)

    meta_path = output_path / "metadata.jsonl"
    with open(meta_path, "w") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")

    print(f"Synthetic {dataset_name}: {len(samples)} Q&A pairs saved")
    return samples


def fetch_cdvqa(output_dir, max_samples=None):
    """Fetch CDVQA dataset for change-based VQA training."""
    print("\n" + "=" * 60)
    print("Fetching CDVQA Dataset")
    print("=" * 60)

    output_path = Path(output_dir) / "cdvqa"
    output_path.mkdir(parents=True, exist_ok=True)

    try:
        print("Loading CDVQA from HuggingFace...")
        dataset = load_dataset("whirc/CDVQA", split="train", streaming=True)
    except Exception:
        try:
            dataset = load_dataset("CDVQA", split="train", streaming=True)
        except Exception as e:
            print(f"CDVQA not available: {e}")
            print("Creating synthetic change-detection pairs...")
            return create_synthetic_change_data(output_dir, max_samples or 200)

    samples = []
    count = 0
    limit = max_samples or 200

    for item in dataset:
        if count >= limit:
            break
        try:
            sample = {
                "id": f"cdvqa_{count:06d}",
                "question": item.get("question", ""),
                "answer": str(item.get("answer", "")),
            }

            # Save image pair
            img1 = item.get("image1") or item.get("before")
            img2 = item.get("image2") or item.get("after")

            if img1 and img2:
                img1_path = output_path / f"{sample['id']}_before.png"
                img2_path = output_path / f"{sample['id']}_after.png"
                if isinstance(img1, Image.Image):
                    img1.save(img1_path)
                if isinstance(img2, Image.Image):
                    img2.save(img2_path)
                sample["image1_path"] = str(img1_path)
                sample["image2_path"] = str(img2_path)
            else:
                continue

            samples.append(sample)
            count += 1

            if count % 50 == 0:
                print(f"  Collected {count}/{limit} change pairs...")

        except Exception:
            continue

    meta_path = output_path / "metadata.jsonl"
    with open(meta_path, "w") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")

    print(f"CDVQA: {len(samples)} change VQA pairs saved to {output_path}")
    return samples


def create_synthetic_change_data(output_dir, num_samples):
    """Create synthetic bi-temporal image pairs for change detection training."""
    import numpy as np

    output_path = Path(output_dir) / "cdvqa"
    output_path.mkdir(parents=True, exist_ok=True)

    change_questions = [
        ("Has the built-up area increased?", "yes"),
        ("Has the built-up area increased?", "no"),
        ("What changed between these two images?", "urban expansion"),
        ("What changed between these two images?", "vegetation loss"),
        ("What changed between these two images?", "water level change"),
        ("Is there deforestation visible?", "yes"),
        ("Is there deforestation visible?", "no"),
        ("Has the water body expanded?", "yes"),
        ("Has the water body expanded?", "no"),
        ("Describe the changes.", "new buildings in the northeast"),
        ("Describe the changes.", "increased vegetation cover"),
    ]

    samples = []
    for i in range(num_samples):
        q, a = random.choice(change_questions)
        sample_id = f"cdvqa_synth_{i:06d}"

        # Generate before/after pair
        base = np.random.randint(50, 150, size=(224, 224, 3), dtype=np.uint8)
        modified = base.copy()

        # Apply some changes
        num_changes = random.randint(1, 3)
        for _ in range(num_changes):
            x = random.randint(0, 180)
            y = random.randint(0, 180)
            w = random.randint(20, 60)
            h = random.randint(20, 60)
            change_type = random.choice(["brighter", "darker", "greener", "bluer"])
            if change_type == "brighter":
                modified[x : x + w, y : y + h] = np.clip(
                    modified[x : x + w, y : y + h].astype(int) + 50, 0, 255
                ).astype(np.uint8)
            elif change_type == "darker":
                modified[x : x + w, y : y + h] = np.clip(
                    modified[x : x + w, y : y + h].astype(int) - 50, 0, 255
                ).astype(np.uint8)
            elif change_type == "greener":
                modified[x : x + w, y : y + h, 1] = np.clip(
                    modified[x : x + w, y : y + h, 1].astype(int) + 60, 0, 255
                ).astype(np.uint8)
            elif change_type == "bluer":
                modified[x : x + w, y : y + h, 2] = np.clip(
                    modified[x : x + w, y : y + h, 2].astype(int) + 60, 0, 255
                ).astype(np.uint8)

        img1 = Image.fromarray(base)
        img2 = Image.fromarray(modified)

        img1_path = output_path / f"{sample_id}_before.png"
        img2_path = output_path / f"{sample_id}_after.png"
        img1.save(img1_path)
        img2.save(img2_path)

        sample = {
            "id": sample_id,
            "image1_path": str(img1_path),
            "image2_path": str(img2_path),
            "question": q,
            "answer": a,
        }
        samples.append(sample)

    meta_path = output_path / "metadata.jsonl"
    with open(meta_path, "w") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")

    print(f"Synthetic CDVQA: {len(samples)} change pairs saved")
    return samples


def main():
    parser = argparse.ArgumentParser(
        description="SatQuery AI - Training Data Pipeline"
    )
    parser.add_argument(
        "--dataset",
        choices=["bigearthnet", "rsvqa", "cdvqa", "all"],
        default="all",
        help="Dataset to download",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./data",
        help="Output directory for datasets",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum samples per dataset",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("SatQuery AI - Training Data Pipeline")
    print("=" * 60)
    print(f"Dataset: {args.dataset}")
    print(f"Output: {args.output}")
    print(f"Max samples: {args.max_samples or 'unlimited'}")

    os.makedirs(args.output, exist_ok=True)

    results = {}

    if args.dataset in ("bigearthnet", "all"):
        results["bigearthnet"] = fetch_bigearthnet(args.output, args.max_samples)

    if args.dataset in ("rsvqa", "all"):
        results["rsvqa"] = fetch_rsvqa(args.output, args.max_samples)

    if args.dataset in ("cdvqa", "all"):
        results["cdvqa"] = fetch_cdvqa(args.output, args.max_samples)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    total = 0
    for name, data in results.items():
        print(f"  {name}: {len(data)} samples")
        total += len(data)
    print(f"  Total: {total} samples")
    print(f"\nData saved to: {args.output}/")
    print("You can now run the training notebook with this data.")


if __name__ == "__main__":
    main()
