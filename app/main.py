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
    print(f"  ✓  ready   http://0.0.0.0:{settings.port}")
    yield


def create_app() -> FastAPI:
    from app.routes import analyze, chat, sessions, upload, visualize

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

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=settings.port, reload=True, log_level="critical")
