import logging
import warnings
warnings.filterwarnings("ignore")
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.config import settings
from app.logging_config import configure_logging
from app.services.storage_service import configure_cloudinary

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()
    configure_cloudinary()

    print(f"  ✓  model   {settings.claude_model}")
    print(f"  ✓  chat    {settings.claude_chat_model}")

    # RAG database
    if settings.database_url:
        try:
            from urllib.parse import urlparse
            host = urlparse(settings.database_url).hostname or "postgres"
        except Exception:
            host = "postgres"
        print(f"  ✓  db      postgres  ({host})")
    elif settings.supabase_url and settings.supabase_key:
        project = settings.supabase_url.replace("https://", "").split(".")[0]
        print(f"  ✓  db      supabase  ({project})")
    else:
        print(f"  -  db      not configured  (RAG endpoints disabled)")

    # RAG embeddings
    if settings.sentence_transformer_model:
        print(f"  ✓  embed   local  ({settings.sentence_transformer_model})")
    elif settings.huggingface_api_key:
        print(f"  ✓  embed   huggingface  (all-MiniLM-L6-v2)")
    else:
        print(f"  ✓  embed   local  (all-MiniLM-L6-v2  default)")

    print(f"  ✓  ready   http://0.0.0.0:{settings.port}")
    yield


def create_app() -> FastAPI:
    from app.routes import analyze, chat, rag, sessions, upload, visualize

    app = FastAPI(
        title="Granite Explorations API",
        description="AI-powered financial document analysis and visualization",
        version="1.0.0",
        lifespan=lifespan,
    )

    app.include_router(sessions.router)
    app.include_router(chat.router)
    app.include_router(upload.router)
    app.include_router(analyze.router)
    app.include_router(visualize.router)
    app.include_router(rag.router)

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=settings.port, reload=True, log_level="critical")
