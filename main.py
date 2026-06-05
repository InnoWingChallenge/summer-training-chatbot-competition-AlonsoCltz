from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import chromadb
from dotenv import load_dotenv
from openai import AzureOpenAI

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

    @classmethod
    def from_env(cls) -> "RAGConfig":
        """
        Load all runtime settings from .env / environment variables.
        This file only reads the existing ChromaDB collection. It does not run
        web_scrape.py or embed_gen.py. Run those files separately when you need
        to rebuild the dataset or embeddings.
        """
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
    """One document chunk returned by the vector database."""

    text: str
    metadata: Dict[str, Any]
    distance: Optional[float] = None

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
        """Create an embedding for the user question."""
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
        """Generate a grounded answer using retrieved context and recent question history."""
        system_prompt = (
            "You are a helpful assistant for HKU InnoWing and Innovation Academy. "
            "Answer only using the provided context. The previous questions are only "
            "there to help you understand follow-up questions such as 'it', 'they', or "
            "'that programme'. If the context does not contain the answer, say you do "
            "not have enough information. Keep the answer clear and concise."
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
        """Format previous questions for the answer prompt."""
        if not previous_questions:
            return "None"
        return "\n".join(
            f"{index}. {previous_question}"
            for index, previous_question in enumerate(previous_questions, start=1)
        )


class ChromaRetriever:
    """Reads the existing ChromaDB collection and retrieves relevant chunks."""

    def __init__(self, config: RAGConfig):
        self.config = config
        self.client = chromadb.PersistentClient(path=str(config.chroma_path))
        self.collection = self._load_collection()

    def _load_collection(self):
        try:
            return self.client.get_collection(name=self.config.collection_name)
        except Exception as exc:
            raise RuntimeError(
                f"Could not load ChromaDB collection '{self.config.collection_name}' "
                f"from {self.config.chroma_path}. Make sure embeddings have already "
                "been generated by running embed_gen.py separately."
            ) from exc

    def retrieve(self, query_embedding: List[float], top_k: Optional[int] = None) -> List[RetrievedDocument]:
        """Return the most relevant document chunks for the query embedding."""
        n_results = top_k or self.config.top_k
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )

        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        retrieved: List[RetrievedDocument] = []
        for text, metadata, distance in zip(documents, metadatas, distances):
            if text:
                retrieved.append(
                    RetrievedDocument(
                        text=text,
                        metadata=metadata or {},
                        distance=distance,
                    )
                )
        return retrieved

class JsonHandler:
    @staticmethod
    def load_questions_from_json(path: Path) -> List[str]:
        """Load questions from a JSON file.

        Expected format:
            [
                "Question 1?",
                "Question 2?"
            ]
        """
        if not path.exists():
            raise FileNotFoundError(f"Questions file not found: {path}")

        with path.open("r", encoding="utf-8") as file:
            questions = json.load(file)

        if not isinstance(questions, list):
            raise ValueError("questions.json must contain a JSON list of strings.")

        cleaned_questions = [
            question.strip()
            for question in questions
            if isinstance(question, str) and question.strip()
        ]

        if not cleaned_questions:
            raise ValueError("questions.json does not contain any valid questions.")

        return cleaned_questions

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
    """Coordinates query embedding, retrieval, context building, and answer generation."""

    def __init__(self, config: RAGConfig):
        self.config = config
        self.openai_service = AzureOpenAIService(config)
        self.retriever = ChromaRetriever(config)
        self.context_builder = ContextBuilder(config.max_context_chars)
        self.previous_questions: List[str] = []

    def answer_question(
        self,
        question: str,
        use_history: bool = True,
        save_to_history: bool = True,
    ) -> str:
        """
        Answer one question using retrieval-augmented generation.

        If use_history=True, previous questions are included directly in the
        retrieval query and final LLM prompt. This keeps the multi-question flow
        simple without adding a separate question-rewriting step.
        """
        if not question or not question.strip():
            raise ValueError("Question cannot be empty.")

        history = self.previous_questions if use_history else []
        retrieval_query = self._build_retrieval_query(question, history)

        query_embedding = self.openai_service.embed_query(retrieval_query)
        retrieved_documents = self.retriever.retrieve(query_embedding)
        context = self.context_builder.build(retrieved_documents)

        if not context.strip():
            answer = "I do not have enough information to answer this question."
        else:
            answer = self.openai_service.generate_answer(
                question=question,
                context=context,
                previous_questions=history,
            )

        if save_to_history:
            self.previous_questions.append(question)

        return answer

    def answer_questions(self, questions: Sequence[str]) -> List[Tuple[str, str]]:
        """
        Answer multiple related questions in order.

        Each new question can use the earlier questions as simple conversation
        history, so follow-up wording like "it" or "that programme" is easier
        to understand.
        """
        answers: List[Tuple[str, str]] = []
        for question in questions:
            answers.append((question, self.answer_question(question)))
        return answers

    def reset_history(self) -> None:
        """Clear saved question history for a new conversation."""
        self.previous_questions.clear()

    @staticmethod
    def _build_retrieval_query(question: str, previous_questions: Sequence[str]) -> str:
        """Combine previous questions with the current question for retrieval."""
        if not previous_questions:
            return question

        history_text = "\n".join(
            f"Previous question {index}: {previous_question}"
            for index, previous_question in enumerate(previous_questions, start=1)
        )
        return f"{history_text}\nCurrent question: {question}"


def rag_answer(question: str) -> str:
    """
    Public API function for answering one question.

    Example:
        answer = rag_answer("What is the Innovation Academy?")
    """
    app = RAGApplication(RAGConfig.from_env())
    return app.answer_question(question)


def generate_rag_answers(questions: List[str]) -> List[Tuple[str, str]]:
    """
    Public API function for answering multiple related questions.

    Questions are answered in order. Each question can use previous questions as
    simple conversation history.

    Returns:
        A list of (question, answer) pairs.
    """
    app = RAGApplication(RAGConfig.from_env())
    return app.answer_questions(questions)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Ask questions against the existing ChromaDB RAG index. "
            "This script does not run the web scraper or embedding generator."
        )
    )
    parser.add_argument(
        "questions",
        nargs="*",
        help="Question(s) to ask. If omitted, an example question is used.",
    )
    return parser.parse_args()


def main() -> None:
    """Command-line entry point."""
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
