from __future__ import annotations

import argparse
import json
import logging
import os
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Hide Hugging Face progress bars
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

# Hide Python warnings
warnings.filterwarnings("ignore")

# Hide Hugging Face / Transformers logs
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)

try:
    from huggingface_hub.utils import disable_progress_bars
    from huggingface_hub import logging as hf_logging
    from transformers.utils import logging as transformers_logging

    disable_progress_bars()
    hf_logging.set_verbosity_error()
    transformers_logging.set_verbosity_error()
except Exception:
    pass

import chromadb
import numpy as np
from dotenv import load_dotenv
from openai import AzureOpenAI
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder


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

    # Hybrid retrieval / reranking parameters
    retrieval_top_k: int = 10
    rrf_k: int = 60
    rerank_threshold: float = 0.0
    final_top_k: int = 3
    vector_threshold: float = 0.0

    # Iterative planning parameters
    max_sub_questions: int = 4
    max_rounds: int = 3
    max_round_notes_chars: int = 9000
    debug: bool = True

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
            retrieval_top_k=int(os.getenv("RAG_RETRIEVAL_TOP_K", "10")),
            rrf_k=int(os.getenv("RAG_RRF_K", "60")),
            rerank_threshold=float(os.getenv("RAG_RERANK_THRESHOLD", "0.0")),
            final_top_k=int(os.getenv("RAG_FINAL_TOP_K", "3")),
            vector_threshold=float(os.getenv("RAG_VECTOR_THRESHOLD", "0.0")),
            max_sub_questions=int(os.getenv("RAG_MAX_SUB_QUESTIONS", "3")),
            max_rounds=int(os.getenv("RAG_MAX_ROUNDS", "2")),
            max_round_notes_chars=int(os.getenv("RAG_MAX_ROUND_NOTES_CHARS", "9000")),
            debug=os.getenv("RAG_DEBUG", "1").strip().lower() not in {"0", "false", "no", "off"},
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

    @property
    def chunk_id(self) -> str:
        return f"{self.metadata.get('url', 'unknown')}:{self.metadata.get('chunk_index', 'unknown')}"


class AzureOpenAIService:
    """Azure OpenAI wrapper for embedding, planning, observation extraction, and answer generation."""

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

    def decompose_question(
        self,
        question: str,
        previous_questions: Optional[Sequence[str]] = None,
    ) -> List[str]:
        """
        Split a complex question into retrieval-friendly sub-questions.

        This is still used as round 1. Later rounds use observations from earlier
        rounds, so the output of round 1 can be pipelined into round 2.
        """
        system_prompt = (
            "You are a query planner for a RAG chatbot. Split the user's question into the "
            "smallest helpful standalone sub-questions needed for first-round retrieval. "
            "Do not answer. Return only a JSON array of strings.\n\n"
            "Important planning rules:\n"
            "- If the question asks 'which X won the most prizes/awards', first retrieve the list of X "
            "and any broad prize/award records. Do not try to decide the final winner in this step.\n"
            "- If the question likely needs candidate entities first, create a candidate-list sub-question.\n"
            "- If it asks about multiple entities/events/people, create one sub-question for each.\n"
            "- Keep each sub-question self-contained."
        )
        history_text = self._format_previous_questions(previous_questions)
        user_prompt = (
            f"Previous questions:\n{history_text}\n\n"
            f"Current question:\n{question}\n\n"
            f"Return at most {self.config.max_sub_questions} first-round sub-questions as JSON:"
        )

        try:
            response = self.chat_client.chat.completions.create(
                model=self.config.chat_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.0,
            )
            content = response.choices[0].message.content or ""
            sub_questions = self._parse_json_string_list(content)
        except Exception:
            sub_questions = []

        if not sub_questions:
            sub_questions = self._fallback_split_question(question)

        return self._clean_string_list(sub_questions, self.config.max_sub_questions) or [question.strip()]

    def plan_next_round(
        self,
        original_question: str,
        previous_questions: Optional[Sequence[str]] = None,
        round_number: int = 1,
        previous_round_notes: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> List[str]:
        """Plan retrieval sub-questions for the current iterative round.

        This is the core pipeline logic:
        - round 1 creates broad/candidate-list retrieval queries;
        - after each round, observations/entities are stored in round_notes;
        - round 2+ uses those notes to create entity-specific follow-up queries.

        Example:
        User: "Which SIG teams won most prizes?"
        Round 1: retrieve candidate SIG team list / broad prize records.
        Round 2: retrieve prizes won by the discovered teams.
        Round 3: retrieve missing details only if needed.
        """
        if round_number <= 1 or not previous_round_notes:
            return self.decompose_question(original_question, previous_questions)

        system_prompt = (
            "You are an iterative query planner for a RAG chatbot. You must decide what to retrieve next "
            "for the original user question. You are given structured observations from earlier retrieval rounds. "
            "Use newly discovered entities such as team names, project names, people, programmes, events, dates, "
            "competitions, and prizes to write concrete follow-up retrieval questions.\n\n"
            "Return only a JSON array of strings. Return [] only when the previous observations already contain "
            "enough evidence to answer the original question.\n\n"
            "Rules:\n"
            "- If candidate teams/entities were found and the original question asks who won the most prizes/awards, "
            "ask for the prizes/awards won by those exact candidates.\n"
            "- Prefer one concise query covering all candidate entities if there are many entities.\n"
            "- If a previous observation says information is missing, ask a targeted query for that missing information.\n"
            "- Do not repeat a previous sub-question.\n"
            "- Do not answer the original question here."
        )
        history_text = self._format_previous_questions(previous_questions)
        notes_text = self._format_round_notes(previous_round_notes)
        user_prompt = (
            f"Previous conversation questions:\n{history_text}\n\n"
            f"Original question:\n{original_question}\n\n"
            f"Current round number: {round_number}\n\n"
            f"Previous retrieval observations:\n{notes_text}\n\n"
            f"Return at most {self.config.max_sub_questions} next sub-questions as JSON. "
            "Return [] only if no more retrieval is needed:"
        )

        try:
            response = self.chat_client.chat.completions.create(
                model=self.config.chat_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.0,
            )
            content = response.choices[0].message.content or ""
            sub_questions = self._parse_json_string_list(content)
        except Exception:
            sub_questions = []

        cleaned = self._clean_string_list(sub_questions, self.config.max_sub_questions)
        if cleaned:
            return cleaned

        # Guardrail: if the LLM planner is too conservative, create a deterministic
        # follow-up when the original question obviously needs aggregation over entities.
        return self._fallback_follow_up_questions(original_question, previous_round_notes)

    def summarize_round_observation(
        self,
        original_question: str,
        sub_question: str,
        context: str,
        previous_round_notes: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Convert retrieved context for one sub-question into structured notes for the next round."""
        system_prompt = (
            "You summarize retrieved context for an iterative RAG pipeline. Use only the provided context. "
            "Return only a JSON object with these keys:\n"
            "{\n"
            "  \"observation\": string,\n"
            "  \"entities\": [string],\n"
            "  \"prizes_or_awards\": [string],\n"
            "  \"missing_information\": [string],\n"
            "  \"enough_to_answer_original\": boolean,\n"
            "  \"suggested_follow_up_questions\": [string]\n"
            "}\n\n"
            "Entity extraction rules:\n"
            "- Include discovered team names, project names, competition names, people, programmes, and awards.\n"
            "- If the context contains a list of teams, put each team name in entities.\n"
            "- If the original question asks which team/entity won the most prizes, suggest a follow-up question "
            "asking what prizes were won by the discovered teams unless the context already gives complete counts.\n"
            "- Do not invent facts. If uncertain, add missing_information instead."
        )
        notes_text = self._format_round_notes(previous_round_notes)
        user_prompt = (
            f"Original question:\n{original_question}\n\n"
            f"Current sub-question:\n{sub_question}\n\n"
            f"Previous round observations:\n{notes_text}\n\n"
            f"Retrieved context:\n{context}\n\n"
            "JSON object:"
        )

        default_note = {
            "observation": "The retrieved context did not provide enough clear information for this sub-question.",
            "entities": [],
            "prizes_or_awards": [],
            "missing_information": [sub_question],
            "enough_to_answer_original": False,
            "suggested_follow_up_questions": [],
        }

        try:
            response = self.chat_client.chat.completions.create(
                model=self.config.chat_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.0,
            )
            content = response.choices[0].message.content or ""
            parsed = self._parse_json_object(content)
        except Exception:
            parsed = {}

        note = dict(default_note)
        if isinstance(parsed, dict):
            for key in note.keys():
                if key in parsed:
                    note[key] = parsed[key]

        note["observation"] = str(note.get("observation") or default_note["observation"]).strip()
        for key in ["entities", "prizes_or_awards", "missing_information", "suggested_follow_up_questions"]:
            note[key] = self._clean_string_list(note.get(key, []), max_items=30)
        note["enough_to_answer_original"] = bool(note.get("enough_to_answer_original", False))
        return note

    def generate_answer(
        self,
        question: str,
        context: str,
        previous_questions: Optional[Sequence[str]] = None,
        sub_questions: Optional[Sequence[str]] = None,
    ) -> str:
        system_prompt = (
            "You are a helpful assistant for HKU InnoWing and Innovation Academy. "
            "Answer only using the provided context. The previous questions are only there "
            "to help you understand follow-up questions such as 'it', 'they', or 'that programme'. "
            "The sub-questions show how the original question was broken down for retrieval. "
            "Use the context from all rounds and sub-questions to answer the original question completely. "
            "If the context contains intermediate observations, treat them as extracted notes from earlier retrieved chunks. "
            "If the question requires counting, ranking, or comparing prizes/awards, show the count or evidence used. "
            "If the context answers only part of the question, answer that part and say which part does not have enough information. "
            "If the context does not contain the answer at all, say you do not have enough information. "
            "Keep the answer clear and concise. For image sources, OCR may contain spelling mistakes, so fix obvious spelling if needed."
        )

        history_text = self._format_previous_questions(previous_questions)
        sub_question_text = self._format_previous_questions(sub_questions) if sub_questions else "None"
        user_prompt = (
            f"Previous questions:\n{history_text}\n\n"
            f"Retrieval sub-questions:\n{sub_question_text}\n\n"
            f"Context:\n{context}\n\n"
            f"Original question:\n{question}\n\n"
            "Final answer:"
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
    def _parse_json_string_list(content: str) -> List[str]:
        """Parse a JSON array of strings, including when wrapped in markdown fences."""
        value = AzureOpenAIService._parse_json_value(content, expected="array")
        if not isinstance(value, list):
            return []
        return [item for item in value if isinstance(item, str) and item.strip()]

    @staticmethod
    def _parse_json_object(content: str) -> Dict[str, Any]:
        """Parse a JSON object, including when wrapped in markdown fences."""
        value = AzureOpenAIService._parse_json_value(content, expected="object")
        return value if isinstance(value, dict) else {}

    @staticmethod
    def _parse_json_value(content: str, expected: str) -> Any:
        content = (content or "").strip()
        content = re.sub(r"^```(?:json)?\s*", "", content, flags=re.IGNORECASE)
        content = re.sub(r"\s*```$", "", content)

        if expected == "array":
            match = re.search(r"\[.*\]", content, flags=re.DOTALL)
        else:
            match = re.search(r"\{.*\}", content, flags=re.DOTALL)
        if match:
            content = match.group(0)

        return json.loads(content)

    @staticmethod
    def _clean_string_list(value: Any, max_items: int = 10) -> List[str]:
        if value is None:
            return []
        if isinstance(value, str):
            value = [value]
        if not isinstance(value, Sequence):
            return []

        cleaned: List[str] = []
        seen = set()
        for item in value:
            if not isinstance(item, str):
                continue
            item = re.sub(r"\s+", " ", item).strip(" \t\n\r-•*")
            if not item:
                continue
            key = item.lower()
            if key in seen:
                continue
            cleaned.append(item)
            seen.add(key)
            if len(cleaned) >= max_items:
                break
        return cleaned

    @staticmethod
    def _fallback_split_question(question: str) -> List[str]:
        """Small rule-based backup splitter for common 'and' / aggregation questions."""
        q = question.strip()
        lower = q.lower()

        # For aggregation questions, retrieve candidates first and then let later rounds follow up.
        aggregation_words = ["most", "highest", "largest", "winner", "won most", "more prizes", "most prizes", "most awards"]
        if any(word in lower for word in aggregation_words):
            return [
                f"What candidate entities, teams, projects, prizes, or awards are relevant to this question: {q}",
                q,
            ]

        # Split patterns like: "who is X and who is Y", "what is X and what is Y".
        repeated_wh = re.search(
            r"\b(and|also)\s+(who|what|when|where|which|how|why)\b",
            lower,
        )
        if repeated_wh:
            parts = re.split(
                r"\s+(?:and|also)\s+(?=(?:who|what|when|where|which|how|why)\b)",
                q,
                flags=re.IGNORECASE,
            )
            return [part.strip(" ?.!") + "?" for part in parts if part.strip()]

        # For explicit comparisons, keep the original query because the LLM planner is better.
        comparison_words = ["compare", "difference", "different", "similar", "versus", " vs "]
        if any(word in lower for word in comparison_words):
            return [q]

        return [q]

    def _fallback_follow_up_questions(
        self,
        original_question: str,
        previous_round_notes: Optional[Sequence[Dict[str, Any]]],
    ) -> List[str]:
        """Deterministic backup for the common 'find candidates, then count prizes' pattern."""
        if not previous_round_notes:
            return []

        lower = original_question.lower()
        needs_prize_aggregation = any(
            phrase in lower
            for phrase in ["most prize", "most award", "won most", "highest number of prize", "highest number of award"]
        )
        if not needs_prize_aggregation:
            return []

        entities = self._collect_entities(previous_round_notes, max_items=20)
        if not entities:
            return []

        entity_text = "; ".join(entities)
        return [
            f"For these candidate entities/teams: {entity_text}, what prizes or awards did each one win?"
        ][: self.config.max_sub_questions]

    @staticmethod
    def _collect_entities(notes: Optional[Sequence[Dict[str, Any]]], max_items: int = 20) -> List[str]:
        if not notes:
            return []
        entities: List[str] = []
        for note in notes:
            for entity in note.get("entities", []) or []:
                if isinstance(entity, str):
                    entities.append(entity)
        return AzureOpenAIService._clean_string_list(entities, max_items=max_items)

    def _format_round_notes(self, notes: Optional[Sequence[Dict[str, Any]]]) -> str:
        if not notes:
            return "None"
        lines: List[str] = []
        used_chars = 0
        for note in notes:
            round_no = note.get("round", "?")
            sub_q = str(note.get("sub_question", "")).strip()
            observation = str(note.get("observation") or note.get("answer") or "").strip()
            entities = self._clean_string_list(note.get("entities", []), max_items=30)
            awards = self._clean_string_list(note.get("prizes_or_awards", []), max_items=20)
            missing = self._clean_string_list(note.get("missing_information", []), max_items=10)
            followups = self._clean_string_list(note.get("suggested_follow_up_questions", []), max_items=10)
            sources = self._clean_string_list(note.get("sources", []), max_items=5)
            enough = note.get("enough_to_answer_original", False)

            block = [f"Round {round_no} sub-question: {sub_q}"]
            if observation:
                block.append(f"Observation: {observation}")
            if entities:
                block.append("Entities: " + "; ".join(entities))
            if awards:
                block.append("Prizes/Awards: " + "; ".join(awards))
            if missing:
                block.append("Missing information: " + "; ".join(missing))
            if followups:
                block.append("Suggested follow-up questions: " + "; ".join(followups))
            if sources:
                block.append("Sources: " + "; ".join(sources))
            block.append(f"Enough to answer original: {bool(enough)}")
            block_text = "\n".join(block)

            if used_chars + len(block_text) > self.config.max_round_notes_chars:
                remaining = self.config.max_round_notes_chars - used_chars
                if remaining <= 0:
                    break
                block_text = block_text[:remaining]
            lines.append(block_text)
            used_chars += len(block_text)
        return "\n\n".join(lines) if lines else "None"

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

        all_data = self.collection.get(include=["documents", "metadatas"])
        self.all_documents = all_data["documents"] or []
        self.all_metadatas = all_data["metadatas"] or []

        tokenized_corpus = [self._tokenize(doc) for doc in self.all_documents]
        self.bm25_index = BM25Okapi(tokenized_corpus) if tokenized_corpus else None

    def _load_collection(self):
        try:
            return self.client.get_collection(name=self.config.collection_name)
        except Exception as exc:
            raise RuntimeError(
                f"Could not load ChromaDB collection '{self.config.collection_name}'"
            ) from exc

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return re.findall(r"[a-z0-9]+", text.lower())

    def retrieve_hybrid_rrf(self, raw_question: str, query_embedding: List[float]) -> List[RetrievedDocument]:
        """Runs Vector + BM25 wide searches, then blends them with Reciprocal Rank Fusion."""
        top_k = self.config.retrieval_top_k
        if top_k <= 0:
            return []

        vector_results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        v_docs = vector_results.get("documents", [[]])[0] or []
        v_metas = vector_results.get("metadatas", [[]])[0] or []
        v_dists = vector_results.get("distances", [[]])[0] or []

        vector_ranked: Dict[str, Dict[str, Any]] = {}
        for rank, (doc, meta, dist) in enumerate(zip(v_docs, v_metas, v_dists)):
            meta = meta or {}
            chunk_id = f"{meta.get('url', 'unknown')}:{meta.get('chunk_index', rank)}"
            try:
                vector_score = round(1 - float(dist), 4)
            except Exception:
                vector_score = 0.0
            vector_ranked[chunk_id] = {
                "rank": rank + 1,
                "text": doc,
                "meta": meta,
                "vector_score": vector_score,
            }

        bm25_ranked: Dict[str, Dict[str, Any]] = {}
        if self.bm25_index is not None and self.all_documents:
            query_tokens = self._tokenize(raw_question)
            bm25_scores = self.bm25_index.get_scores(query_tokens)
            top_bm25_indices = np.argsort(bm25_scores)[::-1][:top_k]

            for rank, idx in enumerate(top_bm25_indices):
                if idx >= len(self.all_documents) or idx >= len(self.all_metadatas):
                    continue
                meta = self.all_metadatas[idx] or {}
                chunk_id = f"{meta.get('url', 'unknown')}:{meta.get('chunk_index', idx)}"
                bm25_ranked[chunk_id] = {
                    "rank": rank + 1,
                    "text": self.all_documents[idx],
                    "meta": meta,
                    "bm25_score": round(float(bm25_scores[idx]), 4),
                }

        all_chunk_ids = set(vector_ranked.keys()) | set(bm25_ranked.keys())
        fused_documents: List[RetrievedDocument] = []

        for cid in all_chunk_ids:
            v_rank = vector_ranked.get(cid, {}).get("rank", top_k + 1)
            b_rank = bm25_ranked.get(cid, {}).get("rank", top_k + 1)
            rrf_score = (1 / (self.config.rrf_k + v_rank)) + (1 / (self.config.rrf_k + b_rank))
            source_data = vector_ranked.get(cid) or bm25_ranked.get(cid)
            if not source_data:
                continue

            fused_documents.append(
                RetrievedDocument(
                    text=source_data["text"],
                    metadata=source_data["meta"],
                    vector_score=vector_ranked.get(cid, {}).get("vector_score", 0.0),
                    bm25_score=bm25_ranked.get(cid, {}).get("bm25_score", 0.0),
                    rrf_score=round(rrf_score, 6),
                )
            )

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
            sub_question = document.metadata.get("retrieval_sub_question")
            sub_question_line = f"Retrieved for: {sub_question}\n" if sub_question else ""
            source_title = document.metadata.get("title") or document.metadata.get("source") or document.metadata.get("filename")
            title_line = f"Title/Source name: {source_title}\n" if source_title else ""
            score_line = ""
            if document.rerank_score is not None:
                score_line = f"Rerank score: {document.rerank_score:.4f}\n"

            part = (
                f"[Source {index}]\n"
                f"{sub_question_line}"
                f"URL: {document.source_url}\n"
                f"{title_line}"
                f"{score_line}"
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
    """Coordinates question planning, hybrid retrieval, reranking, iterative notes, and final generation."""

    def __init__(self, config: RAGConfig):
        self.config = config
        self.openai_service = AzureOpenAIService(config)
        self.retriever = ChromaRetriever(config)
        self.context_builder = ContextBuilder(config.max_context_chars)
        self.previous_questions: List[str] = []

        self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    def answer_question(
        self,
        question: str,
        use_history: bool = True,
        save_to_history: bool = True,
    ) -> str:
        if not question or not question.strip():
            raise ValueError("Question cannot be empty.")

        question = question.strip()
        history = self.previous_questions if use_history else []
        max_rounds = max(1, self.config.max_rounds)

        all_context_docs: List[RetrievedDocument] = []
        all_sub_questions: List[str] = []
        round_notes: List[Dict[str, Any]] = []
        seen_sub_questions = set()

        for round_no in range(1, max_rounds + 1):
            planned_sub_questions = self.openai_service.plan_next_round(
                original_question=question,
                previous_questions=history,
                round_number=round_no,
                previous_round_notes=round_notes,
            )

            # Use suggested follow-ups from observations before stopping. This makes the pipeline
            # less dependent on the planner call returning perfect JSON.
            if round_no > 1 and not planned_sub_questions:
                planned_sub_questions = self._collect_suggested_followups(round_notes)

            fresh_sub_questions: List[str] = []
            for sub_q in planned_sub_questions:
                key = re.sub(r"\s+", " ", sub_q.strip().lower())
                if key and key not in seen_sub_questions:
                    fresh_sub_questions.append(sub_q.strip())
                    seen_sub_questions.add(key)

            #if self.config.debug:
                #print(f"\n[ROUND {round_no}/{max_rounds}] PLAN")
                #if fresh_sub_questions:
                    #print("[PLAN] Sub-questions:")
                    #for index, sub_q in enumerate(fresh_sub_questions, start=1):
                    #    print(f"  {index}. {sub_q}")
                #else:
                    #print("[PLAN] No new sub-questions. Stopping iterative retrieval.")

            if not fresh_sub_questions:
                break

            round_docs: List[RetrievedDocument] = []
            for sub_question in fresh_sub_questions:
                retrieval_query = self._build_retrieval_query(
                    question=sub_question,
                    previous_questions=history,
                    round_notes=round_notes,
                    original_question=question,
                )

                docs_for_sub_question = self._retrieve_and_rerank(
                    retrieval_question=sub_question,
                    retrieval_query=retrieval_query,
                )
                all_context_docs.extend(docs_for_sub_question)
                round_docs.extend(docs_for_sub_question)
                all_sub_questions.append(sub_question)

                sources = self._extract_sources(docs_for_sub_question)
                if docs_for_sub_question:
                    sub_context = self.context_builder.build(docs_for_sub_question)
                    note = self.openai_service.summarize_round_observation(
                        original_question=question,
                        sub_question=sub_question,
                        context=sub_context,
                        previous_round_notes=round_notes,
                    )
                else:
                    note = {
                        "observation": "No relevant context was retrieved for this sub-question.",
                        "entities": [],
                        "prizes_or_awards": [],
                        "missing_information": [sub_question],
                        "enough_to_answer_original": False,
                        "suggested_follow_up_questions": [],
                    }

                note.update(
                    {
                        "round": round_no,
                        "sub_question": sub_question,
                        "sources": sources,
                        "retrieved_chunk_count": len(docs_for_sub_question),
                    }
                )
                round_notes.append(note)

                #if self.config.debug:
                    #print(f"[ROUND {round_no}] Observation for: {sub_question}")
                    #print(f"  {note.get('observation', '')}")
                    #entities = note.get("entities") or []
                    #if entities:
                    #    print("  Entities: " + "; ".join(entities[:10]))
                    #followups = note.get("suggested_follow_up_questions") or []
                    #if followups:
                        #print("  Suggested follow-up: " + "; ".join(followups[:3]))

            #if self.config.debug:
                #print(f"[ROUND {round_no}] Retrieved chunks: {len(round_docs)}")

            # If the structured notes say we have enough information, we can stop early.
            # We only do this after at least round 2 so candidate-list questions still get a follow-up chance.
            if round_no >= 2 and self._notes_indicate_enough(round_notes):
                #if self.config.debug:
                    #print(f"[ROUND {round_no}] Notes indicate enough evidence. Stopping early.")
                break

        final_context_docs = self._deduplicate_documents(all_context_docs)

        if not final_context_docs and not round_notes:
            if save_to_history:
                self.previous_questions.append(question)
            return "I do not have enough information in my knowledge base to answer that question accurately."

        retrieved_context = self.context_builder.build(final_context_docs) if final_context_docs else "No retrieved context chunks."
        observation_context = self.openai_service._format_round_notes(round_notes)
        context = (
            f"Intermediate observations from retrieval rounds:\n{observation_context}\n\n"
            f"Retrieved context chunks:\n{retrieved_context}"
        )

        answer = self.openai_service.generate_answer(
            question=question,
            context=context,
            previous_questions=history,
            sub_questions=all_sub_questions,
        )

        if save_to_history:
            self.previous_questions.append(question)

        return answer

    def _retrieve_and_rerank(
        self,
        retrieval_question: str,
        retrieval_query: str,
    ) -> List[RetrievedDocument]:
        """Retrieve documents for one sub-question, then rerank and filter them."""
        query_embedding = self.openai_service.embed_query(retrieval_query)

        # Important change: use the enriched retrieval_query for both embedding and BM25.
        # This lets previous-round entities/observations affect lexical search too.
        candidates = self.retriever.retrieve_hybrid_rrf(retrieval_query, query_embedding)
        candidates = [
            doc for doc in candidates
            if doc.vector_score >= self.config.vector_threshold or doc.bm25_score > 0
        ]

        if not candidates:
            return []

        pairs = [(retrieval_question, doc.text) for doc in candidates]
        rerank_scores = self.reranker.predict(pairs)

        reranked_docs: List[RetrievedDocument] = []
        for doc, score in zip(candidates, rerank_scores):
            metadata = dict(doc.metadata)
            metadata["retrieval_sub_question"] = retrieval_question
            reranked_docs.append(
                RetrievedDocument(
                    text=doc.text,
                    metadata=metadata,
                    vector_score=doc.vector_score,
                    bm25_score=doc.bm25_score,
                    rrf_score=doc.rrf_score,
                    rerank_score=float(score),
                )
            )

        reranked_docs.sort(key=lambda x: x.rerank_score if x.rerank_score is not None else -999, reverse=True)
        filtered_docs = [
            doc for doc in reranked_docs
            if doc.rerank_score is not None and doc.rerank_score >= self.config.rerank_threshold
        ]

        if not filtered_docs:
            # Fallback: keep the single best candidate, otherwise multi-round questions can fail too easily.
            return reranked_docs[:1]

        return filtered_docs[:self.config.final_top_k]

    def _deduplicate_documents(self, documents: Sequence[RetrievedDocument]) -> List[RetrievedDocument]:
        """Keep only one copy of each chunk, preferring the highest rerank score."""
        best_by_id: Dict[str, RetrievedDocument] = {}
        for doc in documents:
            current = best_by_id.get(doc.chunk_id)
            doc_score = doc.rerank_score if doc.rerank_score is not None else -999.0
            current_score = current.rerank_score if current and current.rerank_score is not None else -999.0
            if current is None or doc_score > current_score:
                best_by_id[doc.chunk_id] = doc

        deduped = list(best_by_id.values())
        deduped.sort(key=lambda x: x.rerank_score if x.rerank_score is not None else -999, reverse=True)

        # Allow enough chunks for all rounds, but keep the final context bounded.
        max_docs = max(
            self.config.final_top_k,
            self.config.final_top_k * self.config.max_sub_questions * max(1, self.config.max_rounds),
        )
        return deduped[:max_docs]

    def answer_questions(self, questions: Sequence[str]) -> List[Tuple[str, str]]:
        answers: List[Tuple[str, str]] = []
        for question in questions:
            answers.append((question, self.answer_question(question)))
        return answers

    def reset_history(self) -> None:
        self.previous_questions.clear()

    def _build_retrieval_query(
        self,
        question: str,
        previous_questions: Sequence[str],
        round_notes: Optional[Sequence[Dict[str, Any]]] = None,
        original_question: Optional[str] = None,
    ) -> str:
        parts: List[str] = []
        if previous_questions:
            history_text = "\n".join(
                f"Previous question {index}: {previous_question}"
                for index, previous_question in enumerate(previous_questions, start=1)
            )
            parts.append(history_text)

        if original_question and original_question.strip() != question.strip():
            parts.append(f"Original user question: {original_question}")

        if round_notes:
            parts.append("Known observations from previous retrieval rounds:")
            parts.append(self.openai_service._format_round_notes(round_notes))

            entities = self.openai_service._collect_entities(round_notes, max_items=30)
            if entities:
                parts.append("Known candidate entities: " + "; ".join(entities))

        parts.append(f"Current retrieval question: {question}")
        return "\n".join(parts)

    @staticmethod
    def _extract_sources(documents: Sequence[RetrievedDocument]) -> List[str]:
        sources: List[str] = []
        seen = set()
        for doc in documents:
            source = doc.source_url
            key = source.lower()
            if source and key not in seen:
                sources.append(source)
                seen.add(key)
        return sources

    @staticmethod
    def _collect_suggested_followups(round_notes: Sequence[Dict[str, Any]]) -> List[str]:
        followups: List[str] = []
        seen = set()
        for note in round_notes:
            for followup in note.get("suggested_follow_up_questions", []) or []:
                if not isinstance(followup, str):
                    continue
                item = re.sub(r"\s+", " ", followup).strip()
                key = item.lower()
                if item and key not in seen:
                    followups.append(item)
                    seen.add(key)
        return followups

    @staticmethod
    def _notes_indicate_enough(round_notes: Sequence[Dict[str, Any]]) -> bool:
        if not round_notes:
            return False
        # Stop early only when at least one latest note says enough and none of the latest
        # notes report missing information. This avoids stopping after a partial candidate list.
        latest_round = max(int(note.get("round", 0) or 0) for note in round_notes)
        latest_notes = [note for note in round_notes if int(note.get("round", 0) or 0) == latest_round]
        if not latest_notes:
            return False
        any_enough = any(bool(note.get("enough_to_answer_original", False)) for note in latest_notes)
        any_missing = any(bool(note.get("missing_information")) for note in latest_notes)
        return any_enough and not any_missing


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ask questions against upgraded hybrid iterative RAG index.")
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


def generate_rag_answers(questions):
    """Compatibility helper for callers that already import generate_rag_answers."""
    config = RAGConfig.from_env()
    app = RAGApplication(config)

    if not questions:
        print("not funny jason")
        return []
    results = []
    for question, answer in app.answer_questions(questions):
        results.append(answer)
    return results
def rag_answer(questions):
    """Compatibility helper for callers that already import generate_rag_answers."""
    config = RAGConfig.from_env()
    app = RAGApplication(config)

    if not questions:
        print("not funny jason")
        return []
    results = []
    for question, answer in app.answer_questions([questions]):
        results.append(answer)
    return results

if __name__ == "__main__":
    main()
