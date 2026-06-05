"""
photo_processing.py

Object-oriented image OCR pipeline using EasyOCR.

What it does:
1. Loads multiple images from ./pictures by default (png, jpg, jpeg).
2. Extracts text from each image using EasyOCR.
3. Appends concise OCR results to an existing JSON array file.
4. Keeps the JSON format close to your web scraper format.

Install dependency:
    pip install easyocr

Example:
    python photo_processing.py
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence

# Linux/macOS file locking. This helps when your scraper and OCR script write to
# the same JSON file at the same time. If you run this on Windows, consider using
# the package `portalocker` instead.
try:
    import fcntl
except ImportError:  # pragma: no cover
    fcntl = None

try:
    import easyocr
except ImportError as exc:
    raise ImportError("EasyOCR is not installed. Install it with: pip install easyocr") from exc


class JsonArrayAppender:
    """
    Append dictionary records to a JSON file with a top-level array.

    This avoids loading and rewriting the whole JSON file. It opens the file,
    removes the final `]`, appends the new object, then writes `]` back.

    Expected JSON file format:
        [
          {
            "url": "https://example.com",
            "text": "normal web text"
          }
        ]
    """

    def __init__(self, json_path: Path) -> None:
        self.json_path = Path(json_path)
        self.json_path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, item: Dict[str, str]) -> None:
        self.append_many([item])

    def append_many(self, items: Sequence[Dict[str, str]]) -> None:
        if not items:
            return

        self._ensure_file_exists()

        with open(self.json_path, "r+b") as file:
            self._lock(file)
            try:
                self._append_many_locked(file, items)
            finally:
                self._unlock(file)

    def _ensure_file_exists(self) -> None:
        if not self.json_path.exists() or self.json_path.stat().st_size == 0:
            self.json_path.write_text("[]", encoding="utf-8")

    def _append_many_locked(self, file, items: Sequence[Dict[str, str]]) -> None:
        closing_bracket_pos = self._find_last_non_whitespace_char(file)

        if closing_bracket_pos is None:
            file.seek(0)
            file.write(b"[]")
            file.truncate()
            closing_bracket_pos = 1
        else:
            file.seek(closing_bracket_pos)
            if file.read(1) != b"]":
                raise ValueError(
                    f"{self.json_path} must be a JSON array ending with ']'."
                )

        has_existing_items = self._array_has_items(file, closing_bracket_pos)

        file.seek(closing_bracket_pos)
        file.truncate()

        parts: List[str] = []
        for index, item in enumerate(items):
            if index == 0:
                parts.append(",\n" if has_existing_items else "\n")
            else:
                parts.append(",\n")
            parts.append(json.dumps(item, ensure_ascii=False, indent=2))

        parts.append("\n]")
        file.write("".join(parts).encode("utf-8"))
        file.flush()
        os.fsync(file.fileno())

    def _find_last_non_whitespace_char(self, file) -> Optional[int]:
        file.seek(0, os.SEEK_END)
        pos = file.tell() - 1

        while pos >= 0:
            file.seek(pos)
            char = file.read(1)
            if char not in b" \t\r\n":
                return pos
            pos -= 1

        return None

    def _array_has_items(self, file, closing_bracket_pos: int) -> bool:
        pos = closing_bracket_pos - 1

        while pos >= 0:
            file.seek(pos)
            char = file.read(1)
            if char in b" \t\r\n":
                pos -= 1
                continue
            return char != b"["

        return False

    def _lock(self, file) -> None:
        if fcntl is not None:
            fcntl.flock(file.fileno(), fcntl.LOCK_EX)

    def _unlock(self, file) -> None:
        if fcntl is not None:
            fcntl.flock(file.fileno(), fcntl.LOCK_UN)


class PhotoProcessor:
    """Load images, extract text using EasyOCR, and append concise records to JSON."""

    SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg"}

    def __init__(
        self,
        image_dir: Optional[Path] = None,
        output_json_path: Optional[Path] = None,
        languages: Optional[List[str]] = None,
        gpu: bool = False,
    ) -> None:
        self.BASE_DIR = Path(__file__).resolve().parent
        self.IMAGE_DIR = image_dir or self.BASE_DIR / "pictures"
        self.OUTPUT_JSON_PATH = output_json_path or self.BASE_DIR / "image_data.json"

        self.languages = languages or ["en"]
        self.gpu = gpu

        self.reader = easyocr.Reader(self.languages, gpu=self.gpu)
        self.json_appender = JsonArrayAppender(self.OUTPUT_JSON_PATH)

    def find_images(self) -> List[Path]:
        """Find all png, jpg, and jpeg files inside IMAGE_DIR."""
        if not self.IMAGE_DIR.exists():
            raise FileNotFoundError(f"Image directory does not exist: {self.IMAGE_DIR}")

        return sorted(
            path
            for path in self.IMAGE_DIR.iterdir()
            if path.is_file() and path.suffix.lower() in self.SUPPORTED_EXTENSIONS
        )

    def extract_text_from_image(self, image_path: Path) -> str:
        """Extract text from one image using EasyOCR."""
        results = self.reader.readtext(str(image_path), detail=0)
        lines = [str(text).strip() for text in results if str(text).strip()]
        return "\n".join(lines)

    def build_record(self, image_path: Path, source_url: Optional[str] = None) -> Dict[str, str]:
        """
        Build a concise JSON record.

        To distinguish image OCR text from normal webpage text, this uses:
            "source": "image"

        Normal scraper records can remain like:
            {
              "url": "...",
              "text": "..."
            }

        Image OCR records will be:
            {
              "source": "image",
              "image": "poster.png",
              "url": "...",        # optional, only if provided
              "text": "..."
            }
        """
        text = self.extract_text_from_image(image_path)

        record = {
            "source": "image",
            "image": image_path.name,
            "text": text,
        }

        if source_url:
            record["url"] = source_url

        return record

    def process_with_openai_later(self, text: str) -> Optional[str]:
        """
        Placeholder for future OpenAI/Azure OpenAI logic.

        You can implement this later if you want to summarize, classify,
        or clean the OCR text.
        """
        return None

    def process_one_image(self, image_path: Path, source_url: Optional[str] = None) -> Dict[str, str]:
        """Process one image and append one concise record to JSON."""
        record = self.build_record(image_path=image_path, source_url=source_url)
        self.json_appender.append(record)
        return record

    def process_all_images(self, url_by_filename: Optional[Dict[str, str]] = None) -> List[Dict[str, str]]:
        """
        Process all images inside IMAGE_DIR.

        Optional example if you know which page an image came from:
            processor.process_all_images({
                "poster.png": "https://example.com/page"
            })
        """
        url_by_filename = url_by_filename or {}
        records = []

        for image_path in self.find_images():
            source_url = url_by_filename.get(image_path.name)
            records.append(self.process_one_image(image_path, source_url=source_url))

        return records


if __name__ == "__main__":
    processor = PhotoProcessor(
        image_dir=Path(__file__).resolve().parent / "pictures",
        output_json_path=Path(__file__).resolve().parent / "image_data.json",
        languages=["en"],
        gpu=False,
    )

    records = processor.process_all_images()
    print(f"Processed {len(records)} image(s).")
