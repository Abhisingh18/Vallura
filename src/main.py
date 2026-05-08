"""
Valura AI — FastAPI application entry point.

Initializes the application with:
  - Classifier (OpenAI or rule-based fallback)
  - Session memory store
  - API routes

Run with: uvicorn src.main:app --reload
"""
from __future__ import annotations

import logging
import os
import sys
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# ---------------------------------------------------------------------------
# Environment & Logging
# ---------------------------------------------------------------------------
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)-25s | %(levelname)-7s | %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lifespan — initialize shared resources
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown lifecycle."""
    from src.classifier.factory import create_classifier
    from src.memory.session_memory import SessionMemory

    # Classifier — factory selects OpenAI or rule-based
    app.state.classifier = create_classifier()
    logger.info("Classifier initialized: %s", type(app.state.classifier).__name__)

    # Session memory — in-memory for the prototype
    app.state.memory = SessionMemory()
    logger.info("Session memory initialized (in-memory)")

    yield  # app runs here

    # Shutdown (nothing to clean up for in-memory state)
    logger.info("Shutting down Valura AI")


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Valura AI Financial Assistant",
    description=(
        "AI-powered financial assistant microservice with safety filtering, "
        "intent classification, agent routing, and streaming responses."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

# CORS — permissive for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routes
from src.api.routes import router as chat_router  # noqa: E402

app.include_router(chat_router)


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    """Simple health check endpoint."""
    return {
        "status": "ok",
        "classifier": type(app.state.classifier).__name__
        if hasattr(app.state, "classifier")
        else "not_initialized",
    }

# ---------------------------------------------------------------------------
# Frontend Serving
# ---------------------------------------------------------------------------
import os
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

frontend_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend")

if os.path.isdir(frontend_path):
    # Mount the frontend directory. html=True automatically serves index.html at "/"
    app.mount("/", StaticFiles(directory=frontend_path, html=True), name="frontend")
else:
    logger.warning("Frontend directory not found at %s", frontend_path)
