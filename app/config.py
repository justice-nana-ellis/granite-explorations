from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    claude_api_key: str
    claude_model: str = "claude-opus-4-7"
    claude_chat_model: str = "claude-haiku-4-5-20251001"
    port: int = 8000

    session_ttl: int = 3600
    session_max_messages: int = 10
    session_max_count: int = 200

    cloudinary_cloud_name: str = ""
    cloudinary_api_key: str = ""
    cloudinary_api_secret: str = ""

    # ── RAG: database (pick one) ──────────────────────────────────────
    # Option A — direct PostgreSQL (Neon, Render, Railway, RDS, Aurora …)
    database_url: str = ""
    # Option B — Supabase REST client
    supabase_url: str = ""
    supabase_key: str = ""   # use the service_role (secret) key

    # ── RAG: embeddings (pick one) ────────────────────────────────────
    # Option A — Hugging Face Inference API (free account, cloud-friendly)
    huggingface_api_key: str = ""
    # Option B — local sentence-transformers (no key, ~90 MB download once)
    #   Leave blank → uses all-MiniLM-L6-v2; or set a custom HF model name
    sentence_transformer_model: str = ""

    model_config = SettingsConfigDict(
        env_file=Path(__file__).parents[1] / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
