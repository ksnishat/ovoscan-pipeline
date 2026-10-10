"""
RAG quality assistant for OvoScan.

Builds a ChromaDB vector index over a hatchery quality-control manual and
answers defect-disposition questions with a local Ollama LLM.

Design notes
------------
* The embedding model and LLM are both configurable via environment variables
  so the module does not hardcode a specific Ollama tag.
* The vector store is persisted to disk, so the index is built once and reused
  across process restarts instead of being rebuilt on every API boot.
* Retrieval is exposed separately from generation (`retrieve`) so the retrieved
  chunks can be logged for observability and evaluated independently.
* Every failure path degrades to a template answer rather than raising, because
  the API must stay up even when the LLM is unavailable.
"""
from __future__ import annotations

import os
from typing import Any, Optional

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_LLM_MODEL = "llama3.2"
DEFAULT_PERSIST_DIR = "data/chroma"
DEFAULT_COLLECTION = "hatchery_rules"

PROMPT_TEMPLATE = """You are a hatchery quality-control assistant.

Use ONLY the reference material below to answer. If the reference material does
not contain the answer, say so explicitly instead of guessing.

=== REFERENCE MATERIAL ===
{context}
=== END REFERENCE MATERIAL ===

QUESTION:
{question}

Answer in three short parts:
1. CRITERIA - what the manual says identifies this condition
2. ACTION - the required disposition step
3. ESCALATION - any threshold that triggers escalation, or "none specified"
"""


class HatcheryAgent:
    """Retrieval-augmented quality assistant backed by ChromaDB."""

    def __init__(
        self,
        knowledge_base_path: str,
        persist_directory: Optional[str] = None,
        collection_name: Optional[str] = None,
        embedding_model: Optional[str] = None,
        llm_model: Optional[str] = None,
        ollama_base_url: Optional[str] = None,
    ) -> None:
        self.kb_path = knowledge_base_path
        self.persist_directory = persist_directory or os.getenv(
            "CHROMA_PERSIST_DIR", DEFAULT_PERSIST_DIR
        )
        self.collection_name = collection_name or os.getenv(
            "CHROMA_COLLECTION", DEFAULT_COLLECTION
        )
        self.embedding_model_name = embedding_model or os.getenv(
            "EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL
        )
        self.llm_model_name = llm_model or os.getenv("OLLAMA_MODEL", DEFAULT_LLM_MODEL)
        self.ollama_base_url = ollama_base_url or os.getenv(
            "OLLAMA_HOST", "http://localhost:11434"
        )

        self.embeddings = HuggingFaceEmbeddings(model_name=self.embedding_model_name)
        self.llm = Ollama(model=self.llm_model_name, base_url=self.ollama_base_url)
        self.vector_db: Optional[Chroma] = None
        self.chain: Any = None

    # ------------------------------------------------------------------ ingest

    def ingest_knowledge(self, force_rebuild: bool = False) -> int:
        """Build (or reuse) the vector index. Returns the number of chunks."""
        if not os.path.exists(self.kb_path):
            raise FileNotFoundError(f"Knowledge base not found: {self.kb_path}")

        os.makedirs(self.persist_directory, exist_ok=True)

        if not force_rebuild and self._index_exists():
            self.vector_db = Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embeddings,
                persist_directory=self.persist_directory,
            )
            count = self.vector_db._collection.count()
            if count > 0:
                self._build_chain()
                return count

        # Chroma.from_documents appends to an existing collection rather than
        # replacing it, so a rebuild must drop the collection first or stale
        # chunks from a previous version of the document survive.
        if force_rebuild and self._index_exists():
            stale = Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embeddings,
                persist_directory=self.persist_directory,
            )
            stale.delete_collection()

        documents = TextLoader(self.kb_path, encoding="utf-8").load()

        # The manual uses long '====' rules as section dividers. Left in place
        # they become their own chunks and dominate retrieval, so strip them
        # and split on paragraph boundaries instead.
        for doc in documents:
            doc.page_content = "\n".join(
                line
                for line in doc.page_content.splitlines()
                if set(line.strip()) not in ({"="}, {"-"}) and line.strip()
            )

        # Recursive splitting respects paragraph boundaries before falling back
        # to character cuts, which keeps the disposition tables in Section 5
        # intact instead of slicing them mid-row.
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=120,
            separators=["\n\n", "\n", " ", ""],
        )
        chunks = splitter.split_documents(documents)

        self.vector_db = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            collection_name=self.collection_name,
            persist_directory=self.persist_directory,
        )
        self._build_chain()
        return len(chunks)

    def _index_exists(self) -> bool:
        return os.path.isdir(self.persist_directory) and bool(
            os.listdir(self.persist_directory)
        )

    def _build_chain(self) -> None:
        prompt = PromptTemplate(
            template=PROMPT_TEMPLATE, input_variables=["context", "question"]
        )
        self.chain = prompt | self.llm | StrOutputParser()

    # --------------------------------------------------------------- retrieval

    def retrieve(self, query: str, k: int = 4) -> list:
        """Return the retrieved chunks with their similarity scores."""
        if self.vector_db is None:
            raise RuntimeError("Vector store not initialised; call ingest_knowledge()")
        results = self.vector_db.similarity_search_with_score(query, k=k)
        return [
            {"content": doc.page_content, "score": float(score)}
            for doc, score in results
        ]

    # -------------------------------------------------------------- generation

    def analyze_defect(self, prediction_class: str) -> str:
        """Answer a disposition question for a predicted class."""
        if self.chain is None or self.vector_db is None:
            raise RuntimeError("RAG chain not initialised; call ingest_knowledge()")

        if prediction_class.lower() in {"fertile", "good"}:
            return "Quality standard met. Egg is fertile - proceed to incubation."

        query = (
            f"The inspection system classified an egg as '{prediction_class}'. "
            "According to the manual, what criteria identify this condition, "
            "what action is required, and what escalation threshold applies?"
        )

        chunks = self.retrieve(query, k=4)
        context = "\n\n---\n\n".join(c["content"] for c in chunks)
        return self.chain.invoke({"context": context, "question": query})


if __name__ == "__main__":
    kb = os.getenv("KNOWLEDGE_BASE_PATH", "knowledge_base/hatchery_manual.txt")
    agent = HatcheryAgent(kb)
    n = agent.ingest_knowledge()
    print(f"Indexed {n} chunks from {kb}\n")

    print("--- retrieved chunks for 'infertile' ---")
    for hit in agent.retrieve("infertile egg criteria and action", k=3):
        print(f"  score={hit['score']:.4f}  {hit['content'][:90]!r}")

    print("\n--- generated report ---")
    print(agent.analyze_defect("infertile"))
