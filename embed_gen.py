from dotenv import load_dotenv
import os
import json
import hashlib
from pathlib import Path
from openai import AzureOpenAI
from typing import List, Dict
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb
from pathlib import Path
BASE_DIR = Path.cwd().parent

# Load .env from parent folder
load_dotenv(BASE_DIR / ".env")

DATASET = BASE_DIR / (os.getenv("DATASET") or "data.json")  # Changed to data.json as requested
EMBEDDING_MODEL = str(BASE_DIR / (os.getenv("EMBEDDING_MODEL") or "text-embedding-3-small"))
CHROMA_PATH = BASE_DIR / (os.getenv("CHROMA_PATH") or "chroma_db/chroma_db")

if os.path.exists(DATASET):
    print(f"📂 Loading documents from {DATASET}...")
    with open(DATASET, "r", encoding="utf-8") as f:
        documents: List[Dict] = json.load(f)
    print(f"   Loaded {len(documents)} raw pages.")
else:
    raise FileNotFoundError(
        f"❌ {DATASET} not found. Please run your scraper first to generate data.json (or set OUTPUT_FILE env var)."
    )

print("\n🔪 Chunking documents using RecursiveCharacterTextSplitter (best-practice settings)...")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=3000,
    length_function=len,
    separators=[""],  # split only on character count
)

all_texts: List[str] = []
all_metadatas: List[Dict] = []
all_ids: List[str] = []

for doc in documents:
    url = doc["url"]
    text = doc.get("text", "").strip()
    if not text:
        continue

    chunks = text_splitter.split_text(text)

    for chunk_idx, chunk in enumerate(chunks):
        # Deterministic ID (hash of URL + chunk index) → safe re-ingest / upsert
        id_str = f"{url}:{chunk_idx}"
        chunk_id = hashlib.md5(id_str.encode("utf-8")).hexdigest()

        all_texts.append(chunk)
        all_metadatas.append({
            "url": url,
            "source": "HKU InnoWings / InnoAcademy",
            "chunk_index": chunk_idx,
            "total_chunks_on_page": len(chunks),   # useful for debugging / future hierarchical RAG
        })
        all_ids.append(chunk_id)

print(f"Created {len(all_texts)} chunks from {len(documents)} documents.")

API_Key = os.getenv("AZURE_OPENAI_API_KEY_2")
if not API_Key:
    raise RuntimeError("Missing Azure OpenAI credentials.")

client = AzureOpenAI(
    azure_endpoint="https://api-iw.azure-api.net/sig-embedding/openai/deployments/text-embedding-3-small/embeddings?api-version=2024-10-21",
    api_key=API_Key,
    api_version="2024-10-21",
)

def get_embedding(text: str) -> List[float]:
    response = client.embeddings.create(
        input=[text.replace("\n", " ")],
        model=EMBEDDING_MODEL,
    )
    return response.data[0].embedding

