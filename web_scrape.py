import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import json
import time
import os
from typing import List, Dict
from pathlib import Path
from dotenv import load_dotenv
from collections import Counter

import re
import mimetypes
from concurrent.futures import ThreadPoolExecutor, as_completed

# Tags that are NEVER meaningful content — strip them unconditionally
NOISE_TAGS = [
    "script", "style", "nav", "header", "footer",
    "aside", "form", "noscript", "iframe",
    "button", "svg", "figure",
]

# CSS class/id selectors that commonly wrap navigation or boilerplate
NOISE_SELECTORS = [
    ".nav", ".navbar", ".menu", ".sidebar",
    ".footer", ".cookie", ".breadcrumb",
    ".social", ".share", ".advertisement",
    "#nav", "#header", "#footer", "#sidebar",
]
# ─────────────────────────────────────────────────────────────────────────────

BLOCKLIST_EXACT = {
    "Skip to content","Main Menu","Menu Toggle","Home","About us","Programmes and activities","Workshop","Sharing","Pitching","Student-initiated courses","Study Tour","Inno Show and Carnivals","Robot Arm Challenge 2026","Student Development Projects","Funding Scheme","Internship Opportunities","Innovation Wing","Previous","Next","Previous image","Next image","Learn more","Learn More","Learn More »","Read More »","Click Here","Contact Us","Be our member!","Register",".",
}

# ── Prefix blocklist — strip lines starting with these strings ───────────────
BLOCKLIST_PREFIXES = (
    "Copyright ©️",
    "For inquiries",
)

class WebScraper:
    def __init__(self):
        self.BASE_DIR = Path(__file__).resolve().parent
        self.SEED_URLS = [
            "https://innowings.engg.hku.hk/",
            "https://innoacademy.engg.hku.hk/",
            "https://innoacademy.engg.hku.hk/pitching/",
        ]
        self.ALLOWED_DOMAINS = {"innowings.engg.hku.hk", "innoacademy.engg.hku.hk"}
        self.HEADERS = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                          "(KHTML, like Gecko) Chrome/133.0 Safari/537.36"
        }
        self.MAX_PAGES = 50          # Safety limit - increase if needed
        self.DELAY = 0.5              # Seconds between requests (polite enough for this small site)
        self.visited: List[str] = []
        self.queue: List[str] = self.SEED_URLS[:]
        self.documents: List[Dict[str, str]] = []

        self.IMAGE_PAGE = "https://innoacademy.engg.hku.hk/pitching/"
        self.IMAGE_DIR = Path(__file__).resolve().parent / "pictures"
        self.IMAGE_DIR.mkdir(parents=True, exist_ok=True)
        self.image_executor = ThreadPoolExecutor(max_workers=5)
        self.image_futures = []

    def safe_filename(self, url: str, index: int) -> str:
        parsed = urlparse(url)
        name = Path(parsed.path).name
        name = re.sub(r"[^A-Za-z0-9._-]", "_", name)
        if not name or "." not in name:
            name = f"image_{index}.jpg"
        return name

    def get_img_urls(self, soup: BeautifulSoup, page_url: str) -> List[str]:
        img_urls = []

        for img in soup.find_all("img"):
            candidates = []

            for attr in ["src", "data-src", "data-lazy-src", "data-original"]:
                if img.get(attr):
                    candidates.append(img.get(attr))

            if img.get("srcset"):
                for item in img.get("srcset").split(","):
                    candidates.append(item.strip().split(" ")[0])

            for candidate in candidates:
                full_url = urljoin(page_url, candidate)
                if full_url not in img_urls:
                    img_urls.append(full_url)

        return img_urls

    def download_image(self, img_url: str, index: int):
        try:
            resp = requests.get(img_url, headers=self.HEADERS, timeout=20)
            resp.raise_for_status()

            content_type = resp.headers.get("Content-Type", "").split(";")[0]
            if not content_type.startswith("image/"):
                return

            filename = self.safe_filename(img_url, index)
            suffix = Path(filename).suffix

            if not suffix:
                suffix = mimetypes.guess_extension(content_type) or ".jpg"
                filename += suffix

            output_path = self.IMAGE_DIR / filename

            # Avoid overwriting files with the same name
            if output_path.exists():
                stem = output_path.stem
                suffix = output_path.suffix
                output_path = self.IMAGE_DIR / f"{stem}_{index}{suffix}"

            with open(output_path, "wb") as f:
                f.write(resp.content)

            print(f"🖼️ Downloaded image → {output_path}")

        except Exception as e:
            print(f"❌ Failed image {img_url}: {e}")

    def queue_image_downloads(self, soup: BeautifulSoup, page_url: str):
        if not page_url.startswith(self.IMAGE_PAGE):
            return

        img_urls = self.get_img_urls(soup, page_url)

        for index, img_url in enumerate(img_urls, start=1):
            self.image_futures.append(
                self.image_executor.submit(self.download_image, img_url, index)
            )



    # Check if a URL belongs to Innowing
    def is_internal_link(self, url: str) -> bool:
        parsed = urlparse(url)
        return parsed.netloc in self.ALLOWED_DOMAINS or not parsed.netloc  # allow relative links

        # ── CHANGED: simple_clean_text() replaced with clean_html() ──────────────
    def clean_html(self, soup: BeautifulSoup) -> str:
        """
        Extract clean text from a BeautifulSoup object.

        These sites do NOT use semantic HTML (<main>, <article>, <nav>),
        so DOM-based isolation doesn't work. Strategy instead:
          1. Strip known noise tags where they exist.
          2. Extract all body text.
          3. Filter lines using an exact blocklist + prefix blocklist + length floor.
        """
        # Step 1 — strip noisy tags where they do exist
        for tag in soup(NOISE_TAGS):
            tag.decompose()
        for selector in NOISE_SELECTORS:
            for el in soup.select(selector):
                el.decompose()

        # Step 2 — fall back to full body (semantic zones don't exist here)
        target = soup.find("body") or soup
        raw = target.get_text(separator="\n", strip=True)

        # Step 3 — filter line by line
        clean_lines = []
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            if line in BLOCKLIST_EXACT:           # exact nav/boilerplate match
                continue
            if line.startswith(BLOCKLIST_PREFIXES):  # copyright / contact footer
                continue
            if len(line) < 20:                    # too short to be real content
                continue
            clean_lines.append(line)

        return "\n".join(clean_lines)

    # ── CHANGED: returns (doc, soup) to avoid second HTTP request ─────────────
    def scrape_page(self, url: str) -> tuple:  # returns (dict, BeautifulSoup | None)
        """
        Fetch a page and return both the cleaned text dict AND the soup object.
        Returning soup lets crawl() extract links without a second HTTP request.
        """
        try:
            resp = requests.get(url, headers=self.HEADERS, timeout=15)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")
            self.queue_image_downloads(soup, url)   # ← image logic unchanged
            text = self.clean_html(soup)             # ← uses new cleaner
            return {"url": url, "text": text}, soup
        except Exception as e:
            print(f"❌ Failed {url}: {e}")
            return {"url": url, "text": f"[ERROR: {e}]"}, None
    # ─────────────────────────────────────────────────────────────────────────

    def deduplicate_across_pages(self, threshold: float = 0.4):
        """
        Remove lines that appear in more than `threshold` fraction of all pages.
        This catches boilerplate the blocklist missed — any line on 40%+ of pages
        is structural noise, not content.

        Call this AFTER crawl() completes, before save_data().
        """
        if not self.documents:
            return

        total_pages = len(self.documents)
        cutoff = max(2, int(total_pages * threshold))  # at least 2 pages

        # Count how many pages each line appears on
        line_page_count = Counter()
        for doc in self.documents:
            # Use a set — count each line once per page, not per occurrence
            for line in set(doc["text"].splitlines()):
                line_page_count[line.strip()] += 1

        # Build the boilerplate set
        boilerplate = {
            line for line, count in line_page_count.items()
            if count >= cutoff
        }

        print(f"\n🧹 Cross-page dedup: found {len(boilerplate)} boilerplate lines "
              f"(appear on {cutoff}+ of {total_pages} pages)")

        # Strip boilerplate lines from every document
        cleaned = 0
        for doc in self.documents:
            original_lines = doc["text"].splitlines()
            filtered = [l for l in original_lines if l.strip() not in boilerplate]
            if len(filtered) < len(original_lines):
                cleaned += 1
            doc["text"] = "\n".join(filtered)

        print(f"   Applied to {cleaned} pages.")    

    def crawl(self):
        print("🚀 Starting crawling...\n")

        while self.queue and len(self.documents) < self.MAX_PAGES:
            url = self.queue.pop(0)
            if url in self.visited:
                continue

            print(f"📄 [{len(self.documents)+1}/{self.MAX_PAGES}] Scraping → {url}")
            doc, soup = self.scrape_page(url)

            if doc["text"].strip():
                self.documents.append(doc)

            self.visited.append(url)

            if soup is not None:
                try:
                    for a in soup.find_all("a", href=True):
                        full_url = urljoin(url, a["href"])
                        if self.is_internal_link(full_url) and full_url not in self.visited:
                            self.queue.append(full_url)
                except Exception:
                    pass

            time.sleep(self.DELAY)

        if self.image_futures:
            print("\n⏳ Waiting for image downloads to finish...")
            for future in as_completed(self.image_futures):
                future.result()
            self.image_executor.shutdown(wait=True)

        # ── NEW: strip cross-page boilerplate after all pages are collected ──
        self.deduplicate_across_pages(threshold=0.4)

        print(f"\n✅ Crawl finished! Collected {len(self.documents)} pages.")

    def save_data(self):
        load_dotenv("../.env")
        dataset = self.BASE_DIR / (os.getenv("DATASET") or "data.json")

        # Save to data.json
        with open(dataset, "w", encoding="utf-8") as f:
            json.dump(self.documents, f, ensure_ascii=False, indent=2)

        print(f"💾 Saved to {dataset}.")


def test_scraper():
    scraper = WebScraper()
    scraper.crawl()
    scraper.save_data()
    #print(scraper.documents)

if __name__ == "__main__":
    #scraper = WebScraper()
    #scraper.crawl()
    #scraper.save_data()
    test_scraper()