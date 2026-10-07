"""FastAPI backend: serves the trained classifier and the analysis results.

Run: uvicorn api.main:app --reload
Docs: http://127.0.0.1:8000/docs
"""
import json
import random
from contextlib import asynccontextmanager

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.config import ARTIFACTS, MODEL_PATH
from src.explain import predict

state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load everything once at startup rather than on every request.
    state["bundle"] = joblib.load(MODEL_PATH)
    state["names"] = json.loads((ARTIFACTS / "category_names.json").read_text())
    state["analysis"] = json.loads((ARTIFACTS / "analysis.json").read_text())
    state["metrics"] = json.loads((ARTIFACTS / "metrics.json").read_text())
    state["preprocess"] = json.loads((ARTIFACTS / "preprocess_report.json").read_text())
    state["samples"] = json.loads((ARTIFACTS / "samples.json").read_text())
    yield
    state.clear()


app = FastAPI(title="arXiv Abstract Classifier", version="1.0", lifespan=lifespan)


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=30, description="An abstract (or any research text)")
    top_k: int = Field(3, ge=1, le=13)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/categories")
def categories():
    return state["names"]


@app.post("/predict")
def predict_category(req: PredictRequest):
    result = predict(state["bundle"], req.text, req.top_k)
    if not any(seg["score"] for seg in result["segments"]):
        raise HTTPException(422, "None of the words in this text are known to the model.")
    for p in result["predictions"]:
        p["name"] = state["names"].get(p["code"], p["code"])
    return result


@app.get("/sample")
def sample(category: str | None = None):
    """A random held-out abstract the model never saw during training."""
    pool = [s for s in state["samples"] if category in (None, s["category_code"])]
    if not pool:
        raise HTTPException(404, f"No samples for category {category!r}")
    return random.choice(pool)


@app.get("/stats/overview")
def overview():
    return {"preprocess": state["preprocess"],
            "category_counts": state["analysis"]["category_counts"],
            "papers_per_year": state["analysis"]["papers_per_year"]}


@app.get("/stats/association")
def association():
    """How category relates to summary text: chi-square words, similarity, length."""
    a = state["analysis"]
    return {"top_terms": a["top_terms"], "similarity": a["similarity"], "length": a["length"]}


@app.get("/stats/model")
def model_metrics():
    return state["metrics"]
