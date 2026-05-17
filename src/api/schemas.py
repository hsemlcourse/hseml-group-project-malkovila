from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

ChannelType = Literal[
    "lifestyle", "entertainment", "bus", "business",
    "socmed", "social", "tech", "technology", "world",
]
WeekdayType = Literal[
    "monday", "tuesday", "wednesday", "thursday",
    "friday", "saturday", "sunday",
]


class PredictRequest(BaseModel):
    title: str = Field(..., min_length=1, max_length=512, description="Текст новостного заголовка")
    channel: ChannelType = Field(..., description="Канал: lifestyle, entertainment, bus, socmed, tech, world")
    weekday: WeekdayType = Field(..., description="День публикации: monday … sunday")
    threshold: float = Field(0.5, ge=0.0, le=1.0, description="Порог вероятности для is_popular")


class FeatureImportance(BaseModel):
    feature: str
    importance: float


class PredictResponse(BaseModel):
    probability: float
    is_popular: bool
    classification_threshold: float
    popularity_threshold_shares: float
    model: str
    top_features_global: list[FeatureImportance]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


class BatchItem(BaseModel):
    title: str = Field(..., min_length=1, max_length=512)
    channel: ChannelType = Field(...)
    weekday: WeekdayType = Field(...)
    threshold: float = Field(0.5, ge=0.0, le=1.0)


class PredictBatchRequest(BaseModel):
    items: list[BatchItem] = Field(..., min_length=1, max_length=20)


class BatchPrediction(BaseModel):
    title: str
    probability: float
    is_popular: bool


class PredictBatchResponse(BaseModel):
    predictions: list[BatchPrediction]
    classification_threshold: float
    model: str


class VersionResponse(BaseModel):
    model: str
    popularity_threshold_shares: float
    git_sha: str
    version: str
