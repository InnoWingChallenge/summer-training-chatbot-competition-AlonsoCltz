"""
image_pipeline.py

Unified image processing pipeline for the InnoWings RAG chatbot.

Two image sources, two processing strategies:

  1. Web-scraped poster images  →  ./pictures/  (downloaded by web_scrape.py)
     Strategy : EasyOCR text extraction via photo_processing.py
     Why      : Posters contain printed text (event names, dates, fees, deadlines)
                that OCR reads accurately and cheaply — no API calls needed.
     Scope    : https://innoacademy.engg.hku.hk/pitching/ only (per competition rules)

  2. Phone photos of physical spaces  →  ./pictures/phone/  (added manually)
     Strategy : GPT-5-mini vision captioning via describe_image.py
     Why      : Physical space photos have little readable text — vision models
                produce rich, semantically meaningful descriptions for retrieval.
     Scope    : Makerspace A, Open Event Area, Brainstorming Area, gallery wall
                (public areas of InnoWing 1 only, per competition rules)

Both sources write records to data.json in the same format as web-scraped pages.
This means embed_gen.py and main.py need ZERO changes to handle image data —
the captions and OCR text are retrieved alongside normal page text.

Usage:
    python image_pipeline.py

Run this AFTER web_scrape.py (so ./pictures/ is populated) and BEFORE embed_gen.py.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Set

from dotenv import load_dotenv

# ── Load .env before any imports that read env vars ──────────────────────────
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

from describe_image import describe_image        # GPT-5-mini vision captioning
from photo_processing import PhotoProcessor      # EasyOCR (teammate's implementation)


# ── Paths ─────────────────────────────────────────────────────────────────────
SCRAPED_IMAGE_DIR = BASE_DIR / "pictures"        # web_scrape.py dumps images here
PHONE_IMAGE_DIR   = BASE_DIR / "pictures" / "phone"  # manually added phone photos
DATA_JSON_PATH    = BASE_DIR / "data.json"
PROCESSED_LOG     = BASE_DIR / ".processed_images"  # tracks already-processed filenames

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg"}

# Matches WordPress/CMS resize suffixes: -300x212, -1536x1086, -768x543, etc.
_RESIZE_RE = re.compile(r"-\d+x\d+\.")

# Source URL attributed to all scraped poster images in data.json
PITCHING_URL = "https://innoacademy.engg.hku.hk/pitching/"

# Domain-specific prompt for physical space photos (GPT-5-mini)
_SPACE_PROMPT = (
    "You are documenting the physical spaces of HKU InnoWings (Innovation Wing), "
    "located in the Haking Wong Building at The University of Hong Kong. "
    "Describe what you see in this photo in 3-5 sentences. "
    "Focus on: the name of the area if visible or inferable "
    "(Makerspace A, Open Event Area, Brainstorming Area, or gallery wall), "
    "the equipment and facilities present (e.g. 3D printers, laser cutters, "
    "soldering stations, workbenches, display panels), "
    "the activities the space supports, and anything relevant to students "
    "interested in innovation, making, or entrepreneurship. "
    "Do not comment on image quality or photography technique. "
    "Return only the description — no headings or bullet points."
)


# ── Processed-log helpers ─────────────────────────────────────────────────────

def load_processed_log() -> Set[str]:
    """Return the set of filenames that have already been processed."""
    if not PROCESSED_LOG.exists():
        return set()
    return set(PROCESSED_LOG.read_text(encoding="utf-8").splitlines())


def mark_processed(filename: str) -> None:
    """Append one filename to the processed log."""
    with open(PROCESSED_LOG, "a", encoding="utf-8") as f:
        f.write(filename + "\n")


# ── Image discovery helpers ───────────────────────────────────────────────────

def is_resize_variant(path: Path) -> bool:
    """
    Return True if this file is a CMS-generated resize variant.
    e.g. Pitch-New-Tech-Idea-2026-300x212.png  →  True
         Pitch-New-Tech-Idea-2026.png           →  False
    """
    return bool(_RESIZE_RE.search(path.name))


def find_scraped_images() -> List[Path]:
    """
    Find original (non-thumbnail) images in ./pictures/.
    The web scraper saves multiple resolutions of each poster; we skip
    all resize variants and only process the full-size original.
    """
    if not SCRAPED_IMAGE_DIR.exists():
        return []

    return sorted(
        p for p in SCRAPED_IMAGE_DIR.iterdir()
        if p.is_file()
        and p.suffix.lower() in SUPPORTED_EXTENSIONS
        and not is_resize_variant(p)
    )


def find_phone_images() -> List[Path]:
    """Return all images in ./pictures/phone/."""
    PHONE_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    return sorted(
        p for p in PHONE_IMAGE_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )


# ── data.json writer ──────────────────────────────────────────────────────────

def append_record(record: Dict) -> None:
    """
    Append one record to data.json.
    Loads the full file, appends, and rewrites — safe for the small scale
    of this project (< 200 records total including image data).
    """
    if not DATA_JSON_PATH.exists() or DATA_JSON_PATH.stat().st_size == 0:
        DATA_JSON_PATH.write_text("[]", encoding="utf-8")

    with open(DATA_JSON_PATH, "r", encoding="utf-8") as f:
        data: List[Dict] = json.load(f)

    data.append(record)

    with open(DATA_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# ── Stage 1: Scraped poster images → EasyOCR ─────────────────────────────────

def process_scraped_posters(processed_log: Set[str]) -> int:
    """
    Run EasyOCR on original-resolution poster images from ./pictures/.
    Uses photo_processing.PhotoProcessor (teammate's implementation) for OCR.
    Appends records to data.json.  Returns count of newly processed images.
    """
    images = find_scraped_images()
    new_images = [p for p in images if p.name not in processed_log]

    if not images:
        print("   No scraped poster images found in ./pictures/")
        return 0

    print(f"\n📋 Found {len(images)} original poster image(s) "
          f"({len(images) - len(new_images)} already processed, "
          f"{len(new_images)} new)")

    if not new_images:
        return 0

    # Initialise EasyOCR once — model loading takes ~5 seconds on first run
    print("   Loading EasyOCR model (first run may take a moment)...")
    processor = PhotoProcessor(
        image_dir=SCRAPED_IMAGE_DIR,
        languages=["en"],
        gpu=False,
    )

    count = 0
    for img_path in new_images:
        print(f"   🔍 OCR  →  {img_path.name}")
        try:
            ocr_text = processor.extract_text_from_image(img_path)

            if not ocr_text.strip():
                print("        ⚠️  No text extracted — skipping (image may be decorative).")
                mark_processed(img_path.name)
                continue

            record = {
                "url": PITCHING_URL,
                "text": ocr_text,
                "source": "image",
                "image": img_path.name,
                "image_type": "scraped_poster",
            }
            append_record(record)
            mark_processed(img_path.name)
            count += 1
            print(f"        ✅ Appended ({len(ocr_text)} chars)")

        except Exception as exc:
            print(f"        ❌ Failed: {exc}")

    return count


# ── Stage 2: Phone photos → GPT-5-mini vision ────────────────────────────────

def process_phone_photos(processed_log: Set[str]) -> int:
    """
    Caption phone photos of InnoWing 1 physical spaces using GPT-5-mini vision.
    Scope (per competition rules): Makerspace A, Open Event Area,
    Brainstorming Area, gallery wall outside the office.
    Appends records to data.json.  Returns count of newly processed images.
    """
    images = find_phone_images()
    new_images = [p for p in images if p.name not in processed_log]

    if not images:
        print(f"\n📱 No phone photos found.")
        print(f"   → Drop photos of InnoWing 1 spaces into: {PHONE_IMAGE_DIR}")
        print("     (Makerspace A / Open Event Area / Brainstorming Area / gallery wall)")
        return 0

    print(f"\n📱 Found {len(images)} phone photo(s) "
          f"({len(images) - len(new_images)} already processed, "
          f"{len(new_images)} new)")

    if not new_images:
        return 0

    # GPT-5-mini is used per competition instructions for image inputs
    vision_model = os.getenv("AZURE_OPENAI_VISION_MODEL", "gpt-5-mini")
    print(f"   Using vision model: {vision_model}")

    count = 0
    for img_path in new_images:
        print(f"   🔭 Vision  →  {img_path.name}")
        try:
            caption = describe_image(
                image_path=str(img_path),
                model=vision_model,
                prompt=_SPACE_PROMPT,
            )

            record = {
                "url": "innowings://physical/innowing-1",
                "text": caption,
                "source": "image",
                "image": img_path.name,
                "image_type": "phone_photo",
                "location": "InnoWing 1",
            }
            append_record(record)
            mark_processed(img_path.name)
            count += 1
            print(f"        ✅ Appended ({len(caption)} chars)")

        except Exception as exc:
            print(f"        ❌ Failed: {exc}")

    return count


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    print("🚀 InnoWings Image Pipeline")
    print("=" * 50)
    print(f"   Poster images (OCR)    : {SCRAPED_IMAGE_DIR}")
    print(f"   Phone photos (vision)  : {PHONE_IMAGE_DIR}")
    print(f"   Output                 : {DATA_JSON_PATH}")

    processed_log = load_processed_log()
    if processed_log:
        print(f"\n   Already in log: {len(processed_log)} filename(s) — will be skipped.")

    scraped_count = process_scraped_posters(processed_log)
    phone_count   = process_phone_photos(processed_log)

    total = scraped_count + phone_count
    print("\n" + "=" * 50)
    if total == 0:
        print("   Nothing new to process. All images already in data.json.")
    else:
        print(f"✅ Added {scraped_count} poster record(s) + {phone_count} space photo record(s) to data.json.")
        print("   → Run embed_gen.py next to rebuild ChromaDB with the new data.")


if __name__ == "__main__":
    main()
