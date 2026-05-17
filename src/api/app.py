from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException

from src.api.schemas import (
    BatchPrediction,
    FeatureImportance,
    HealthResponse,
    PredictBatchRequest,
    PredictBatchResponse,
    PredictRequest,
    PredictResponse,
    VersionResponse,
)
from src.inference.predictor import NewsViralityPredictor, get_predictor

logger = logging.getLogger("api")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")

_predictor: NewsViralityPredictor | None = None


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global _predictor
    try:
        _predictor = get_predictor()
        logger.info("Модель загружена успешно")
    except FileNotFoundError as exc:
        logger.warning("Модель не загружена: %s", exc)
        _predictor = None
    yield


app = FastAPI(
    title="News Title Virality API",
    description="Предсказание вероятности популярности новостного заголовка (UCI Mashable)",
    version="1.0.0",
    lifespan=lifespan,
)


def _require_predictor() -> NewsViralityPredictor:
    if _predictor is None:
        raise HTTPException(
            status_code=503,
            detail="Модель не загружена. Выполните: make features-cp2 && make final-model",
        )
    return _predictor


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", model_loaded=_predictor is not None)


@app.get("/version", response_model=VersionResponse)
def version() -> VersionResponse:
    predictor = _require_predictor()
    return VersionResponse(
        model="LightGBM",
        popularity_threshold_shares=predictor.popularity_threshold_shares,
        git_sha=os.getenv("GIT_SHA", "local"),
        version="1.0.0",
    )


@app.post("/predict", response_model=PredictResponse)
def predict(body: PredictRequest) -> PredictResponse:
    predictor = _require_predictor()
    try:
        raw: dict[str, Any] = predictor.predict(
            title=body.title.strip(),
            channel=body.channel,
            weekday=body.weekday,
            threshold=body.threshold,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    logger.info(
        "predict | len=%d channel=%s prob=%.3f",
        len(body.title), body.channel, raw["probability"],
    )

    top = [
        FeatureImportance(feature=t["feature"], importance=t["importance"])
        for t in raw.get("top_features_global", [])
    ]
    return PredictResponse(
        probability=raw["probability"],
        is_popular=raw["is_popular"],
        classification_threshold=raw["classification_threshold"],
        popularity_threshold_shares=raw["popularity_threshold_shares"],
        model=raw["model"],
        top_features_global=top,
    )


@app.post("/predict_batch", response_model=PredictBatchResponse)
def predict_batch(body: PredictBatchRequest) -> PredictBatchResponse:
    predictor = _require_predictor()
    predictions: list[BatchPrediction] = []
    for item in body.items:
        try:
            raw = predictor.predict(
                title=item.title.strip(),
                channel=item.channel,
                weekday=item.weekday,
                threshold=item.threshold,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        predictions.append(
            BatchPrediction(
                title=item.title,
                probability=raw["probability"],
                is_popular=raw["is_popular"],
            )
        )

    logger.info("predict_batch | count=%d", len(predictions))
    return PredictBatchResponse(
        predictions=predictions,
        classification_threshold=body.items[0].threshold,
        model="LightGBM",
    )
