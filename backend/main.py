"""
InsureAI — FastAPI Application Entry Point
============================================
Initializes database, loads champion model from Model Registry,
configures structured logging, sets up CORS, and attaches ML engines.
"""

import os
import json
import logging
import joblib
import pandas as pd
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from backend.core.settings import settings
from backend.core.logging_config import setup_logging
from backend.api.routes import router
from backend.database.db import init_db
from backend.ml.registry import ModelRegistry

load_dotenv()
setup_logging()
logger = logging.getLogger("insureai.main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 1. Initialize SQLite Database
    try:
        await init_db()
        logger.info("Database initialized successfully.")
    except Exception as e:
        logger.error("Database initialization failed: %s", e)

    # 2. Model Registry & Champion Loading
    app.state.registry = ModelRegistry(settings.MODELS_DIR)
    app.state.model = None
    app.state.metadata = None
    app.state.model_loaded = False
    app.state.predict_engine = None
    app.state.explain_engine = None

    # Determine model and metadata paths strictly via registry active champion pointer
    try:
        model_path, metadata_path = app.state.registry.get_active_champion_paths()
        app.state.model = joblib.load(model_path)
        app.state.model_loaded = True
        logger.info("Loaded champion model from: %s", model_path)

        with open(metadata_path, "r", encoding="utf-8") as f:
            app.state.metadata = json.load(f)
        logger.info(
            "Loaded metadata: model='%s', version='%s', threshold=%s",
            app.state.metadata.get("model_name"),
            app.state.metadata.get("version"),
            str(app.state.metadata.get("default_threshold")),
        )
    except Exception as e:
        logger.warning("Active Champion model could not be loaded via registry pointer: %s", e)

    # 3. Initialize Prediction & Explainability Engines
    if app.state.model_loaded:
        try:
            from backend.ml.predict import PredictionEngine
            from backend.ml.explain import ExplainabilityEngine

            engine = PredictionEngine()
            engine.load_model(model_path)
            if app.state.metadata:
                engine.metadata = app.state.metadata
            app.state.predict_engine = engine

            # Setup SHAP Explainer
            explain_engine = ExplainabilityEngine()
            feature_names = app.state.metadata.get("feature_names", []) if app.state.metadata else []

            # Load small sample of training data for explainer background
            sample_df = None
            if os.path.exists(settings.REAL_TRAIN_DATA_PATH):
                try:
                    sample_df = pd.read_csv(settings.REAL_TRAIN_DATA_PATH, nrows=60)
                    sample_df = sample_df.drop(columns=["is_claim", "policy_id"], errors="ignore")
                except Exception as ex:
                    logger.warning("Could not load sample train data for SHAP background: %s", ex)

            explain_engine.setup(app.state.model, sample_df, feature_names)
            app.state.explain_engine = explain_engine
            logger.info("Prediction and Explainability engines initialized successfully.")
        except Exception as e:
            logger.error("Error during model setup: %s", e, exc_info=True)

    yield
    logger.info("Shutting down InsureAI API...")


import uuid
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

class RequestIDMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        req_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        request.state.request_id = req_id
        response = await call_next(request)
        response.headers["X-Request-ID"] = req_id
        return response

app = FastAPI(
    title="InsureAI — Intelligent Insurance Claim Prediction",
    description="Full-stack AI/ML vehicle insurance claim prediction with explainability, what-if simulations, and risk analysis.",
    version=settings.APP_VERSION,
    lifespan=lifespan,
)

app.add_middleware(RequestIDMiddleware)

# CORS Configuration
origins = [o.strip() for o in settings.CORS_ORIGINS.split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global 500 handler to protect internal stack traces
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    req_id = getattr(request.state, "request_id", "unknown")
    logger.error("[%s] Internal server error: %s", req_id, exc, exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "detail": "An internal server error occurred. Please contact system support.",
            "request_id": req_id,
        },
    )

# Mount API router
app.include_router(router)


@app.get("/")
async def root():
    return {
        "message": "Welcome to InsureAI API.",
        "docs": "/docs",
        "health": "/health",
        "version": settings.APP_VERSION,
    }
