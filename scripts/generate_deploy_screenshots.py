"""Генерирует иллюстрации для §7 отчёта (демо API/Streamlit)."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "report" / "images"


def _save(fig: plt.Figure, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    fig.savefig(path, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {path}")


def swagger_figure() -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.axis("off")
    ax.set_title("FastAPI Swagger — http://127.0.0.1:8000/docs", fontsize=14, weight="bold")
    lines = [
        "News Title Virality API  v1.0.0",
        "",
        "GET  /health          Health",
        "GET  /version         Version",
        "POST /predict         Predict",
        "POST /predict_batch   Predict Batch",
        "",
        "Schemas: PredictRequest, PredictResponse, ...",
    ]
    ax.text(0.05, 0.95, "\n".join(lines), va="top", family="monospace", fontsize=11)
    _save(fig, "08_fastapi_swagger.png")


def predict_response_figure() -> None:
    payload = {
        "probability": 0.5614406304072705,
        "is_popular": True,
        "classification_threshold": 0.5,
        "popularity_threshold_shares": 1400.0,
        "model": "LightGBM",
        "top_features_global": [
            {"feature": "data_channel_is_world", "importance": 0.0691},
            {"feature": "data_channel_is_entertainment", "importance": 0.0461},
        ],
    }
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis("off")
    ax.set_title("POST /predict — Response 200", fontsize=14, weight="bold")
    ax.text(
        0.05,
        0.9,
        json.dumps(payload, indent=2, ensure_ascii=False),
        va="top",
        family="monospace",
        fontsize=10,
    )
    _save(fig, "08_predict_response.png")


def streamlit_figure() -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis("off")
    ax.set_title("Streamlit — http://localhost:8501", fontsize=14, weight="bold")
    lines = [
        "Режим: Сравнение заголовков",
        "Канал: tech   |   День: monday   |   Порог: 0.5",
        "",
        "Заголовок                          P(popular)  Популярен?",
        "10 Things You Need to Know About AI    56.1%        Да",
        "Scientists Discover Clean Energy       48.2%        Нет",
        "Breaking: Major Tech Layoffs           52.7%        Да",
        "",
        "Лучший вариант: первый заголовок (P = 56.1%)",
    ]
    ax.text(0.05, 0.92, "\n".join(lines), va="top", family="monospace", fontsize=11)
    bars = [0.561, 0.482, 0.527]
    ax2 = fig.add_axes([0.55, 0.15, 0.4, 0.35])
    ax2.barh(["H1", "H2", "H3"], bars, color=["#2ecc71", "#95a5a6", "#3498db"])
    ax2.set_xlim(0, 1)
    ax2.set_xlabel("probability")
    _save(fig, "08_streamlit_demo.png")


if __name__ == "__main__":
    swagger_figure()
    predict_response_figure()
    streamlit_figure()
