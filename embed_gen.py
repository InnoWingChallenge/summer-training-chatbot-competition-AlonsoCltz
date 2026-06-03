from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import chromadb
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import AzureOpenAI


@dataclass(frozen=True)
class EmbeddingConfig:
    """Stores all configuration needed for embedding generation and ChromaDB ingestion."""

    base_dir: Path
    dataset: Path
    chroma_path: Path
    embedding_model: str
    azure_endpoint: str
    azure_api_version: str
    collection_name: str
    source_name: str
    chunk_size: int
    batch_size: int
    fresh_ingest: bool

    @classmethod
    def from_env(cls) -> "EmbeddingConfig":
        """Create config from environment variables, loading .env from the parent folder."""
        base_dir = Path.cwd().parent
        load_dotenv(base_dir / ".env")

        return cls(
            base_dir=base_dir,
            dataset=cls._path_from_env(base_dir, "DATASET", "data.json"),
            chroma_path=cls._path_from_env(base_dir, "CHROMA_PATH", "chroma_db/chroma_db"),
            embedding_model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
            azure_endpoint=os.getenv(
                "AZURE_OPENAI_ENDPOINT",
                "https://api-iw.azure-api.net/sig-embedding/openai/deployments/"
                "text-embedding-3-small/embeddings?api-version=2024-10-21",
            ),
            azure_api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21"),
            collection_name=os.getenv("CHROMA_COLLECTION_NAME", "Innowing_db"),
            source_name=os.getenv("SOURCE_NAME", "HKU InnoWings / InnoAcademy"),
            chunk_size=int(os.getenv("CHUNK_SIZE", "3000")),
            batch_size=int(os.getenv("BATCH_SIZE", "50")),
            fresh_ingest=os.getenv("FRESH_INGEST", "true").lower() in {"1", "true", "yes"},
        )

    @staticmethod
    def _path_from_env(base_dir: Path, env_name: str, default_value: str) -> Path:
        """Read a path from env and make it absolute if needed."""
        raw_path = Path(os.getenv(env_name, default_value))
        return raw_path if raw_path.is_absolute() else base_dir / raw_path


class DocumentLoader:
    """Loads scraped documents from a JSON dataset file."""

    def __init__(self, dataset: Path):
        self.dataset = dataset

    def load(self) -> List[Dict[str, Any]]:
        """Load and validate the dataset."""
        if not self.dataset.exists():
            raise FileNotFoundError(
                f"❌ {self.dataset} not found. Please run your scraper first to generate data.json "
                "or set the DATASET environment variable."
            )

        print(f"📂 Loading documents from {self.dataset}...")
        with self.dataset.open("r", encoding="utf-8") as file:
            documents: List[Dict[str, Any]] = json.load(file)

        print(f"   Loaded {len(documents)} raw pages.")
        return documents


class DocumentChunker:
    """Splits page text into smaller chunks and prepares ChromaDB metadata."""

    def __init__(self, chunk_size: int, source_name: str):
        self.source_name = source_name
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            length_function=len,
            separators=[""],  # split only by character count
        )

    def chunk_documents(
        self,
        documents: List[Dict[str, Any]],
    ) -> Tuple[List[str], List[Dict[str, Any]], List[str]]:
        """Return chunk texts, metadata, and deterministic IDs."""
        print("\n🔪 Chunking documents using RecursiveCharacterTextSplitter...")

        all_texts: List[str] = []
        all_metadatas: List[Dict[str, Any]] = []
        all_ids: List[str] = []

        for doc in documents:
            url = doc.get("url", "")
            text = doc.get("text", "").strip()

            if not url or not text:
                continue

            chunks = self.text_splitter.split_text(text)
            self._append_chunks(url, chunks, all_texts, all_metadatas, all_ids)

        print(f"Created {len(all_texts)} chunks from {len(documents)} documents.")
        return all_texts, all_metadatas, all_ids

    def _append_chunks(
        self,
        url: str,
        chunks: List[str],
        all_texts: List[str],
        all_metadatas: List[Dict[str, Any]],
        all_ids: List[str],
    ) -> None:
        """Append chunks from one URL to the shared ingestion lists."""
        for chunk_index, chunk in enumerate(chunks):
            all_texts.append(chunk)
            all_metadatas.append(
                {
                    "url": url,
                    "source": self.source_name,
                    "chunk_index": chunk_index,
                    "total_chunks_on_page": len(chunks),
                }
            )
            all_ids.append(self._make_chunk_id(url, chunk_index))

    @staticmethod
    def _make_chunk_id(url: str, chunk_index: int) -> str:
        """Create a stable ID so repeated ingestion updates the same chunk."""
        raw_id = f"{url}:{chunk_index}"
        return hashlib.md5(raw_id.encode("utf-8")).hexdigest()


class AzureEmbeddingService:
    """Handles communication with Azure OpenAI for embeddings."""

    def __init__(self, config: EmbeddingConfig):
        self.config = config
        self.client = self._create_client()

    def _create_client(self) -> AzureOpenAI:
        """Create an Azure OpenAI client after checking credentials."""
        api_key = os.getenv("AZURE_OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("Missing AZURE_OPENAI_API_KEY in environment variables.")

        return AzureOpenAI(
            azure_endpoint=self.config.azure_endpoint,
            api_key=api_key,
            api_version=self.config.azure_api_version,
        )

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a batch of texts."""
        cleaned_texts = [text.replace("\n", " ") for text in texts]
        response = self.client.embeddings.create(
            input=cleaned_texts,
            model=self.config.embedding_model,
        )
        return [item.embedding for item in response.data]


class ChromaVectorStore:
    """Handles ChromaDB collection setup and vector upserts."""

    def __init__(self, config: EmbeddingConfig):
        self.config = config
        self.client = chromadb.PersistentClient(path=str(config.chroma_path))
        self.collection = self._prepare_collection()

    def _prepare_collection(self):
        """Reset the collection if requested, then create or load it."""
        if self.config.fresh_ingest:
            self._delete_existing_collection()

        return self.client.get_or_create_collection(
            name=self.config.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def _delete_existing_collection(self) -> None:
        """Delete the existing collection for a clean rebuild."""
        try:
            self.client.delete_collection(name=self.config.collection_name)
            print(f"🗑️  Deleted existing collection '{self.config.collection_name}' for fresh ingest.")
        except Exception:
            # Collection may not exist yet, which is fine.
            pass

    def upsert_batch(
        self,
        texts: List[str],
        metadatas: List[Dict[str, Any]],
        ids: List[str],
        embeddings: List[List[float]],
    ) -> None:
        """Insert or update one batch of chunks in ChromaDB."""
        self.collection.upsert(
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
            ids=ids,
        )


class EmbeddingIngestor:
    """Coordinates document loading, chunking, embedding generation, and vector storage."""

    def __init__(self, config: EmbeddingConfig):
        self.config = config
        self.loader = DocumentLoader(config.dataset)
        self.chunker = DocumentChunker(config.chunk_size, config.source_name)
        self.embedding_service = AzureEmbeddingService(config)
        self.vector_store = ChromaVectorStore(config)

    def run(self) -> None:
        """Run the full embedding ingestion pipeline."""
        documents = self.loader.load()
        texts, metadatas, ids = self.chunker.chunk_documents(documents)

        if not texts:
            print("⚠️  No valid text chunks found. Nothing was added to ChromaDB.")
            return

        print("🧬 Generating embeddings and storing in ChromaDB...")
        self._ingest_batches(texts, metadatas, ids)
        self._print_success_message()

    def _ingest_batches(
        self,
        texts: List[str],
        metadatas: List[Dict[str, Any]],
        ids: List[str],
    ) -> None:
        """Embed and upsert all chunks in batches."""
        total = len(texts)

        for start in range(0, total, self.config.batch_size):
            end = start + self.config.batch_size
            batch_texts = texts[start:end]
            batch_metadatas = metadatas[start:end]
            batch_ids = ids[start:end]

            embeddings = self.embedding_service.embed_texts(batch_texts)
            self.vector_store.upsert_batch(batch_texts, batch_metadatas, batch_ids, embeddings)

            print(f"  [{min(end, total)}/{total}] Added batch to ChromaDB")

    def _print_success_message(self) -> None:
        """Print final status details."""
        print("\n🎉 Ingestion complete!")
        print(f"   ChromaDB collection '{self.config.collection_name}' saved to: {self.config.chroma_path}")
        print("   ✅ Ready for RAG! Use the same embedding model for queries.")


def main() -> None:
    """Program entry point."""
    config = EmbeddingConfig.from_env()
    ingestor = EmbeddingIngestor(config)
    ingestor.run()


if __name__ == "__main__":
    main()
