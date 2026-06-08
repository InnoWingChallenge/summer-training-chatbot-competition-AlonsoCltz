from __future__ import annotations

import argparse
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import chromadb
import numpy as np
from dotenv import load_dotenv
from openai import AzureOpenAI
from rank_bm25 import BM25Okapi                  # Added for BM25 keyword stream
from sentence_transformers import CrossEncoder    # Added for Re-ranking
import json


@dataclass(frozen=True)
class RAGConfig:
    """Configuration for the RAG question-answering application."""

    base_dir: Path
    chroma_path: Path
    collection_name: str

    azure_openai_api_key: str

    chat_endpoint: str
    chat_api_version: str
    chat_model: str

    embedding_endpoint: str
    embedding_api_version: str
    embedding_model: str

    top_k: int
    max_context_chars: int
    temperature: float
    
    # --- New Retrieve-Wide Rerank-Narrow Hyperparameters ---
    retrieval_top_k: int = 10      # Retrieve a wider candidate net (e.g., Top-10)
    rrf_k: int = 60                # Smoothing constant for RRF list merging
    rerank_threshold: float = 0.0  # Logit cutoff boundary for Cross-Encoder filtering
    final_top_k: int = 3           # Chunks ultimately delivered to GPT
    vector_threshold: float = 0

    @classmethod
    def from_env(cls) -> "RAGConfig":
        base_dir = Path(__file__).resolve().parent
        load_dotenv(base_dir / ".env")

        api_key = os.getenv("AZURE_OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "Missing Azure OpenAI credentials. Set AZURE_OPENAI_API_KEY "
                "in .env or in your environment."
            )

        return cls(
            base_dir=base_dir,
            chroma_path=cls._path_from_env(base_dir, "CHROMA_PATH", "chroma_db/chroma_db"),
            collection_name=os.getenv("CHROMA_COLLECTION_NAME", "Innowing_db"),
            azure_openai_api_key=api_key,
            chat_endpoint=os.getenv(
                "AZURE_OPENAI_CHAT_ENDPOINT",
                "https://api-iw.azure-api.net/sig-shared-jpeast",
            ),
            chat_api_version=os.getenv("AZURE_OPENAI_CHAT_API_VERSION", "2025-01-01-preview"),
            chat_model=os.getenv("AZURE_OPENAI_CHAT_MODEL", "gpt-4o-mini"),
            embedding_endpoint=os.getenv(
                "AZURE_OPENAI_EMBEDDING_ENDPOINT",
                "https://api-iw.azure-api.net/sig-embedding",
            ),
            embedding_api_version=os.getenv("AZURE_OPENAI_EMBEDDING_API_VERSION", "2024-10-21"),
            embedding_model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
            top_k=int(os.getenv("RAG_TOP_K", "5")),
            max_context_chars=int(os.getenv("RAG_MAX_CONTEXT_CHARS", "12000")),
            temperature=float(os.getenv("RAG_TEMPERATURE", "0.2")),
        )

    @staticmethod
    def _path_from_env(base_dir: Path, env_name: str, default_value: str) -> Path:
        raw_path = Path(os.getenv(env_name, default_value))
        return raw_path if raw_path.is_absolute() else base_dir / raw_path


@dataclass(frozen=True)
class RetrievedDocument:
    """One document chunk returned by the hybrid retrieval streams."""

    text: str
    metadata: Dict[str, Any]
    vector_score: float = 0.0
    bm25_score: float = 0.0
    rrf_score: float = 0.0
    rerank_score: Optional[float] = None

    @property
    def source_url(self) -> str:
        return str(self.metadata.get("url", "Unknown source"))


class AzureOpenAIService:
    """Handles both query embeddings and final answer generation."""

    def __init__(self, config: RAGConfig):
        self.config = config
        self.chat_client = AzureOpenAI(
            base_url=f"{config.chat_endpoint}/deployments/{config.chat_model}",
            api_key=config.azure_openai_api_key,
            api_version=config.chat_api_version,
        )

        self.embedding_client = AzureOpenAI(
            base_url=f"{config.embedding_endpoint}/openai/deployments/{config.embedding_model}",
            api_key=config.azure_openai_api_key,
            api_version=config.embedding_api_version,
        )

    def embed_query(self, question: str) -> List[float]:
        response = self.embedding_client.embeddings.create(
            input=[question.replace("\n", " ")],
            model=self.config.embedding_model,
        )
        return response.data[0].embedding

    def generate_answer(
        self,
        question: str,
        context: str,
        previous_questions: Optional[Sequence[str]] = None,
    ) -> str:
        system_prompt = (
            "You are a helpful assistant for HKU InnoWing and Innovation Academy. "
            "Answer only using the provided context. The previous questions are only "
            "there to help you understand follow-up questions such as 'it', 'they', or "
            "'that programme'. If the context does not contain the answer, say you do "
            "not have enough information. Keep the answer clear and concise."
            "for image sources, there might be errors during OCR, fix the spelling if necessary."
        )

        history_text = self._format_previous_questions(previous_questions)
        user_prompt = (
            f"Previous questions:\n{history_text}\n\n"
            f"Context:\n{context}\n\n"
            f"Current question:\n{question}\n\n"
            "Answer:"
        )

        response = self.chat_client.chat.completions.create(
            model=self.config.chat_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=self.config.temperature,
        )
        return response.choices[0].message.content or ""

    @staticmethod
    def _format_previous_questions(previous_questions: Optional[Sequence[str]]) -> str:
        if not previous_questions:
            return "None"
        return "\n".join(
            f"{index}. {previous_question}"
            for index, previous_question in enumerate(previous_questions, start=1)
        )


class ChromaRetriever:
    """Reads ChromaDB data, manages a localized BM25 index, and handles RRF pooling."""

    def __init__(self, config: RAGConfig):
        self.config = config
        self.client = chromadb.PersistentClient(path=str(config.chroma_path))
        self.collection = self._load_collection()
        
        # --- Build Accompanying BM25 Corpus Index ---
        print("Loading all database chunks to construct the BM25 Index...")
        all_data = self.collection.get(include=["documents", "metadatas"])
        self.all_documents = all_data["documents"] or []
        self.all_metadatas = all_data["metadatas"] or []
        
        tokenized_corpus = [self._tokenize(doc) for doc in self.all_documents]
        self.bm25_index = BM25Okapi(tokenized_corpus)
        print(f"BM25 baseline set up over {len(self.all_documents)} chunks.")

    def _load_collection(self):
        try:
            return self.client.get_collection(name=self.config.collection_name)
        except Exception as exc:
            raise RuntimeError(
                f"Could not load ChromaDB collection '{self.config.collection_name}'"
            ) from exc

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """Simple regex alphanumeric tokenizer conforming to guidelines."""
        return re.findall(r'[a-z0-9]+', text.lower())

    def retrieve_hybrid_rrf(self, raw_question: str, query_embedding: List[float]) -> List[RetrievedDocument]:
        """Runs Vector + BM25 parallel wide-searches, blending with Reciprocal Rank Fusion."""
        top_k = self.config.retrieval_top_k

        # Stream 1: Wide Dense Vector Search
        vector_results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        
        v_docs = vector_results.get("documents", [[]])[0]
        v_metas = vector_results.get("metadatas", [[]])[0]
        v_dists = vector_results.get("distances", [[]])[0]

        vector_ranked = {}
        for rank, (doc, meta, dist) in enumerate(zip(v_docs, v_metas, v_dists)):
            # Establish a reliable dictionary identifier mapping unique chunks
            chunk_id = meta.get("url", "unknown") + ":" + str(meta.get("chunk_index", rank))
            vector_ranked[chunk_id] = {
                "rank": rank + 1,
                "text": doc,
                "meta": meta,
                "vector_score": round(1 - dist, 4)
            }

        # Stream 2: Wide Sparse BM25 Keyword Search
        query_tokens = self._tokenize(raw_question)
        bm25_scores = self.bm25_index.get_scores(query_tokens)
        top_bm25_indices = np.argsort(bm25_scores)[::-1][:top_k]

        bm25_ranked = {}
        for rank, idx in enumerate(top_bm25_indices):
            meta = self.all_metadatas[idx]
            chunk_id = meta.get("url", "unknown") + ":" + str(meta.get("chunk_index", idx))
            bm25_ranked[chunk_id] = {
                "rank": rank + 1,
                "text": self.all_documents[idx],
                "meta": meta,
                "bm25_score": round(float(bm25_scores[idx]), 4)
            }

        # Step 3: Pool lists using Reciprocal Rank Fusion (RRF)
        all_chunk_ids = set(vector_ranked.keys()) | set(bm25_ranked.keys())
        fused_documents: List[RetrievedDocument] = []
        
        for cid in all_chunk_ids:
            v_rank = vector_ranked.get(cid, {}).get("rank", top_k + 1)
            b_rank = bm25_ranked.get(cid, {}).get("rank", top_k + 1)
            
            # Reciprocal Formula: 1 / (k + rank)
            rrf_score = (1 / (self.config.rrf_k + v_rank)) + (1 / (self.config.rrf_k + b_rank))
            source_data = vector_ranked.get(cid) or bm25_ranked.get(cid)

            fused_documents.append(
                RetrievedDocument(
                    text=source_data["text"],
                    metadata=source_data["meta"],
                    vector_score=vector_ranked.get(cid, {}).get("vector_score", 0.0),
                    bm25_score=bm25_ranked.get(cid, {}).get("bm25_score", 0.0),
                    rrf_score=round(rrf_score, 6)
                )
            )

        # Sort combined results descending by calculated RRF score
        fused_documents.sort(key=lambda x: x.rrf_score, reverse=True)
        
        return fused_documents[:top_k]


class JsonHandler:
    @staticmethod
    def load_questions_from_json(path: Path) -> List[str]:
        if not path.exists():
            raise FileNotFoundError(f"Questions file not found: {path}")
        with path.open("r", encoding="utf-8") as file:
            questions = json.load(file)
        return [q.strip() for q in questions if isinstance(q, str) and q.strip()]


class ContextBuilder:
    """Formats retrieved chunks into a compact context block for the LLM."""

    def __init__(self, max_chars: int):
        self.max_chars = max_chars

    def build(self, documents: Sequence[RetrievedDocument]) -> str:
        context_parts: List[str] = []
        used_chars = 0

        for index, document in enumerate(documents, start=1):
            part = (
                f"[Source {index}]\n"
                f"URL: {document.source_url}\n"
                f"Content:\n{document.text.strip()}\n"
            )

            if used_chars + len(part) > self.max_chars:
                remaining_chars = self.max_chars - used_chars
                if remaining_chars <= 0:
                    break
                part = part[:remaining_chars]

            context_parts.append(part)
            used_chars += len(part)

        return "\n---\n".join(context_parts)


class RAGApplication:
    """Coordinates wide hybrid retrieval, Cross-Encoder re-ranking, and final narrow generation."""

    def __init__(self, config: RAGConfig):
        self.config = config
        self.openai_service = AzureOpenAIService(config)
        self.retriever = ChromaRetriever(config)
        self.context_builder = ContextBuilder(config.max_context_chars)
        self.previous_questions: List[str] = []
        
        # --- Instantiate Local Transformer Reranker ---
        print("Loading local CPU Cross-Encoder ('ms-marco-MiniLM-L-6-v2')...")
        self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        print("Reranker ready.")

    def answer_question(
        self,
        question: str,
        use_history: bool = True,
        save_to_history: bool = True,
    ) -> str:
        if not question or not question.strip():
            raise ValueError("Question cannot be empty.")

        history = self.previous_questions if use_history else []
        retrieval_query = self._build_retrieval_query(question, history)

        # 1. Wide Retrieval: Embed and query via the Vector + BM25 Hybrid flow
        query_embedding = self.openai_service.embed_query(retrieval_query)
        candidates = self.retriever.retrieve_hybrid_rrf(question, query_embedding)
        candidates = [
            doc for doc in candidates
            if doc.vector_score >= self.config.vector_threshold or doc.bm25_score > 0
        ]

        if not candidates:
            return "I do not have enough information to answer this question."

        # 2. Narrow Re-ranking: Feed text pairs directly into Cross-Encoder
        pairs = [(question, doc.text) for doc in candidates]
        rerank_scores = self.reranker.predict(pairs)

        reranked_docs = []
        for doc, score in zip(candidates, rerank_scores):
            # Map raw logits back to updated objects
            updated_doc = RetrievedDocument(
                text=doc.text,
                metadata=doc.metadata,
                vector_score=doc.vector_score,
                bm25_score=doc.bm25_score,
                rrf_score=doc.rrf_score,
                rerank_score=float(score)
            )
            reranked_docs.append(updated_doc)

        # Sort candidates cleanly by Cross-Encoder logit scores
        reranked_docs.sort(key=lambda x: x.rerank_score, reverse=True)

        # 3. Confidence Level Guardrail Filter
        # Logit values above 0.0 indicate semantic query satisfaction
        filtered_docs = [
            d for d in reranked_docs 
            if d.rerank_score is not None and d.rerank_score >= self.config.rerank_threshold
        ]

        # Take strictly the targeted high-accuracy payload
        final_context_docs = filtered_docs[:self.config.final_top_k]
        for top_chunk in final_context_docs:
            print(top_chunk)
            print("="*20)
        # Early execution exit if zero context items pass the logit check
        if not final_context_docs:
            print("\n[GUARD] Warning: No context passed the validation threshold.")
            return "I do not have enough information in my knowledge base to answer that question accurately."

        # 4. Final Clean Context Synthesis
        context = self.context_builder.build(final_context_docs)
        
        answer = self.openai_service.generate_answer(
            question=question,
            context=context,
            previous_questions=history,
        )

        if save_to_history:
            self.previous_questions.append(question)

        return answer

    def answer_questions(self, questions: Sequence[str]) -> List[Tuple[str, str]]:
        answers: List[Tuple[str, str]] = []
        for question in questions:
            answers.append((question, self.answer_question(question)))
        return answers

    def reset_history(self) -> None:
        self.previous_questions.clear()

    @staticmethod
    def _build_retrieval_query(question: str, previous_questions: Sequence[str]) -> str:
        if not previous_questions:
            return question
        history_text = "\n".join(
            f"Previous question {index}: {previous_question}"
            for index, previous_question in enumerate(previous_questions, start=1)
        )
        return f"{history_text}\nCurrent question: {question}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ask questions against upgraded hybrid RAG index.")
    parser.add_argument("questions", nargs="*", help="Question(s) to execute.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = RAGConfig.from_env()
    app = RAGApplication(config)

    questions = args.questions
    if not questions:
        questions_path = config.base_dir / "questions.json"
        questions = JsonHandler.load_questions_from_json(questions_path)

    for question, answer in app.answer_questions(questions):
        print(f"\nQ: {question}")
        print(f"A: {answer}")


if __name__ == "__main__":
    main()