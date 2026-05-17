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

    # ── Warm up slow models in the background so first requests are instant ──
    import asyncio

    async def _warm_embedding():
        try:
            from app.services.rag_service import _embed
            await _embed(["warmup"])
            print("  ✓  embed   model warmed up")
        except Exception as exc:
            print(f"  !  embed   warmup failed: {exc}")

    async def _warm_timesfm():
        try:
            from app.services.timesfm_service import ensure_loaded
            await ensure_loaded()
            print("  ✓  timesfm model warmed up")
        except Exception as exc:
            print(f"  !  timesfm warmup failed: {exc}")

    asyncio.create_task(_warm_embedding())
    if settings.use_timesfm:
        asyncio.create_task(_warm_timesfm())

    yield


def create_app() -> FastAPI:
    from app.routes import analyze, chat, fund_forecast, rag, sessions, timesfm, upload, visualize

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
    app.include_router(timesfm.router)
    app.include_router(fund_forecast.router)

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=settings.port, reload=True, log_level="critical")
