import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import json
import time
import os
from typing import List, Dict
from pathlib import Path
from dotenv import load_dotenv

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

class WebScraper:
    def __init__(self):
        self.BASE_DIR = Path.cwd().parent
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
        Strategy:
          1. Remove universally noisy tags (nav, footer, header, etc.)
          2. Remove elements matching known noise CSS selectors.
          3. Isolate the main content zone using semantic HTML.
          4. Fall back to <body> if no semantic container is found.
          5. Drop lines shorter than 20 chars (likely menu remnants).
        """
        # Step 1 — strip noisy tags in-place
        for tag in soup(NOISE_TAGS):
            tag.decompose()

        # Step 2 — strip elements identified by CSS class/id
        for selector in NOISE_SELECTORS:
            for el in soup.select(selector):
                el.decompose()

        # Step 3 — target the main content zone
        content = (
            soup.find("main")
            or soup.find("article")
            or soup.find(attrs={"role": "main"})
            or soup.find("div", class_=lambda c: c and any(
                kw in c.lower() for kw in
                ["content", "post-body", "entry", "article-body", "page-content"]
            ))
        )

        # Step 4 — fall back to body if no semantic zone found
        target = content or soup.find("body") or soup

        # Step 5 — extract and filter text
        raw = target.get_text(separator="\n", strip=True)
        lines = [line.strip() for line in raw.splitlines() if line.strip()]
        lines = [l for l in lines if len(l) >= 20]

        return "\n".join(lines)
    # ─────────────────────────────────────────────────────────────────────────

    # ── CHANGED: returns (doc, soup) to avoid second HTTP request ─────────────
    def scrape_page(self, url: str) -> tuple[Dict[str, str]]:
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

    # ── CHANGED: reuses soup from scrape_page() — single request per page ─────
    def crawl(self):
        print("🚀 Starting crawling...\n")

        while self.queue and len(self.documents) < self.MAX_PAGES:
            url = self.queue.pop(0)
            if url in self.visited:
                continue

            print(f"📄 [{len(self.documents)+1}/{self.MAX_PAGES}] Scraping → {url}")

            # One request per page — soup reused for link extraction below
            doc, soup = self.scrape_page(url)

            if doc["text"].strip():
                self.documents.append(doc)

            self.visited.append(url)

            # Reuse the soup we already have — no second HTTP request needed
            if soup is not None:
                try:
                    for a in soup.find_all("a", href=True):
                        full_url = urljoin(url, a["href"])
                        if self.is_internal_link(full_url) and full_url not in self.visited:
                            self.queue.append(full_url)
                except Exception:
                    pass

            time.sleep(self.DELAY)

        # ── UNCHANGED: image download wait logic ─────────────────────────────
        if self.image_futures:
            print("\n⏳ Waiting for image downloads to finish...")
            for future in as_completed(self.image_futures):
                future.result()
            self.image_executor.shutdown(wait=True)
        # ─────────────────────────────────────────────────────────────────────

        print(f"\n✅ Crawl finished! Collected {len(self.documents)} pages.")
    # ─────────────────────────────────────────────────────────────────────────

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