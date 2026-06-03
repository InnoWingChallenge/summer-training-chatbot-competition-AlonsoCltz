import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import json
import time
import os
from typing import List, Dict
from pathlib import Path
from dotenv import load_dotenv

class WebScraper:
    def __init__(self):
        self.BASE_DIR = Path.cwd().parent
        self.SEED_URLS = [
            "https://innowings.engg.hku.hk/",
            "https://innoacademy.engg.hku.hk/",   
        ]
        self.ALLOWED_DOMAINS = {"innowings.engg.hku.hk", "innoacademy.engg.hku.hk"}
        self.HEADERS = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                          "(KHTML, like Gecko) Chrome/133.0 Safari/537.36"
        }
        self.MAX_PAGES = 20          # Safety limit - increase if needed
        self.DELAY = 1.0              # Seconds between requests (polite enough for this small site)
        self.visited: List[str] = []
        self.queue: List[str] = self.SEED_URLS[:]
        self.documents: List[Dict[str, str]] = []

    # Check if a URL belongs to Innowing
    def is_internal_link(self, url: str) -> bool:
        parsed = urlparse(url)
        return parsed.netloc in self.ALLOWED_DOMAINS or not parsed.netloc  # allow relative links

    # Performs the most basic cleaning: removes only JavaScript and CSS, then extracts readable text while reducing extra blank lines.
    def simple_clean_text(self, soup: BeautifulSoup) -> str:
        # Very basic cleaning - remove script/style only
        for tag in soup(["script", "style"]):
            tag.decompose()
        text = soup.get_text(separator="\n", strip=True)
        # Remove excessive blank lines
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        return "\n".join(lines)

    # Downloads a single page and returns a dictionary containing the URL and its cleaned text (or an error message)
    def scrape_page(self, url: str) -> Dict[str, str]:
        try:
            resp = requests.get(url, headers=self.HEADERS, timeout=15)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")
            text = self.simple_clean_text(soup)
            return {"url": url, "text": text}
        except Exception as e:
            print(f"❌ Failed {url}: {e}")
            return {"url": url, "text": f"[ERROR: {e}]"}

    def crawl(self):
        print("🚀 Starting crawling...\n")
        
        while self.queue and len(self.documents) < self.MAX_PAGES:
            url = self.queue.pop(0)
            if url in self.visited:
                continue

            print(f"📄 [{len(self.documents)+1}/{self.MAX_PAGES}] Scraping → {url}")
            doc = self.scrape_page(url)
            
            if doc["text"].strip():
                self.documents.append(doc)
            
            self.visited.append(url)

            try:
                resp = requests.get(url, headers=self.HEADERS, timeout=10)
                soup = BeautifulSoup(resp.text, "html.parser")
                for a in soup.find_all("a", href=True):
                    full_url = urljoin(url, a["href"])
                    if self.is_internal_link(full_url) and full_url not in self.visited:
                        self.queue.append(full_url)
            except:
                pass  # ignore link extraction errors

            time.sleep(self.DELAY)

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
    print(scraper.documents)

if __name__ == "__main__":
    #scraper = WebScraper()
    #scraper.crawl()
    #scraper.save_data()
    test_scraper()

