from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── Vector DB ─────────────────────────────────────────────────────────────
    # mode "local"  → stores data in qdrant_local_path (no Docker needed)
    # mode "server" → connects to a running Qdrant server at qdrant_url
    qdrant_mode: Literal["local", "server"] = "local"
    qdrant_local_path: str = "./qdrant_data"
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "grounded_rag"

    # ── Embeddings ────────────────────────────────────────────────────────────
    # provider "gemini" → text-embedding-004 via Gemini API (uses gemini_api_key)
    # provider "openai" → text-embedding-3-small via OpenAI API (uses openai_api_key)
    # Swap model + dim together when changing providers.
    embedding_provider: Literal["gemini", "openai"] = "gemini"
    embedding_model: str = "models/gemini-embedding-001"
    embedding_dim: int = 3072

    # ── LLM ──────────────────────────────────────────────────────────────────
    llm_provider: Literal["anthropic", "openai", "gemini"] = "gemini"
    anthropic_api_key: str = ""
    openai_api_key: str = ""
    gemini_api_key: str = ""
    llm_model: str = "gemini-2.5-flash"
    llm_max_tokens: int = 1024

    # ── Retrieval ────────────────────────────────────────────────────────────
    retrieval_top_k: int = 10     # candidates from ANN search
    reranker_top_n: int = 5       # kept after cross-encoder reranking (Phase 4)

    # ── Chunking ─────────────────────────────────────────────────────────────
    chunk_size: int = 512
    chunk_overlap: int = 64

    # ── Confidence / Abstention ───────────────────────────────────────────────
    confidence_threshold: float = 0.3

    # ── Agentic re-retrieval (Phase 8) ───────────────────────────────────────
    agentic_retry_enabled: bool = True
    agentic_max_retries: int = 2        # reformulation attempts before final abstention

    # ── Cache (Phase 7) ───────────────────────────────────────────────────────
    cache_enabled: bool = True
    cache_max_size: int = 512

    # ── Observability (Phase 7) ───────────────────────────────────────────────
    trace_log_path: str = "logs/traces.jsonl"
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "https://cloud.langfuse.com"


settings = Settings()
