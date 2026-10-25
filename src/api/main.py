import io
import json
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import StreamingResponse

from src.api.schemas import FEATURE_COLUMNS, Transaction

MODEL_PATH = Path("model/final_model.joblib")
THRESHOLD_PATH = Path("reports/phase4_threshold.json")
TEST_REPORT_PATH = Path("reports/final_test_report.json")
# preprocess.py ke output ka order: log_amount, V1..V28, hour_sin, hour_cos
FEATURE_NAMES = ["log_amount"] + [f"V{i}" for i in range(1, 29)] + ["hour_sin", "hour_cos"]
MAX_BATCH_ROWS = 100_000

state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["model"] = joblib.load(MODEL_PATH)
    state["cfg"] = json.loads(THRESHOLD_PATH.read_text())
    yield
    state.clear()


app = FastAPI(title="Fraud Detection API", version="1.0.0", lifespan=lifespan)


def score(df: pd.DataFrame) -> np.ndarray:
    return state["model"].predict_proba(df[FEATURE_COLUMNS])[:, 1]


def explain(df: pd.DataFrame, top_k: int = 3) -> list:
    """XGBoost ka exact TreeSHAP: har row ke top_k features (impact + = fraud ki taraf)."""
    model = state["model"]
    est = getattr(model, "named_steps", {}).get("model")
    if not isinstance(est, xgb.XGBClassifier):
        return [[] for _ in range(len(df))]
    Xt = model[:-1].transform(df[FEATURE_COLUMNS])
    contrib = est.get_booster().predict(xgb.DMatrix(Xt), pred_contribs=True)[:, :-1]
    out = []
    for row in contrib:
        idx = np.argsort(np.abs(row))[::-1][:top_k]
        out.append([{"feature": FEATURE_NAMES[i], "impact": round(float(row[i]), 3)} for i in idx])
    return out


def resolve_threshold(threshold):
    return state["cfg"]["threshold"] if threshold is None else threshold


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": "model" in state}


@app.get("/model-info")
def model_info():
    cfg = state["cfg"]
    info = {
        "model": cfg["model"],
        "calibration": cfg["calibration"],
        "default_threshold": cfg["threshold"],
        "c_fp": cfg["c_fp"],
        "test_metrics": None,
    }
    if TEST_REPORT_PATH.exists():
        info["test_metrics"] = json.loads(TEST_REPORT_PATH.read_text())["metrics"]
    return info


@app.post("/predict")
def predict(tx: Transaction, threshold: float | None = Query(None, ge=0.0, le=1.0)):
    t = resolve_threshold(threshold)
    df = pd.DataFrame([tx.model_dump()])[FEATURE_COLUMNS]
    p = float(score(df)[0])
    return {
        "fraud_probability": round(p, 6),
        "decision": "FRAUD" if p >= t else "LEGIT",
        "threshold": t,
        "top_reasons": explain(df)[0],
        "model_version": f"{state['cfg']['model']}-{state['cfg']['calibration']}",
    }


@app.post("/predict/batch")
async def predict_batch(
    file: UploadFile = File(...),
    threshold: float | None = Query(None, ge=0.0, le=1.0),
):
    t = resolve_threshold(threshold)
    raw = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(raw))
    except Exception:
        raise HTTPException(400, "Valid CSV upload karo")
    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise HTTPException(422, f"Missing columns: {missing}")
    if len(df) > MAX_BATCH_ROWS:
        raise HTTPException(413, f"Max {MAX_BATCH_ROWS} rows allowed")
    feats = df[FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce")
    if feats.isna().any().any():
        raise HTTPException(422, "Feature columns me non-numeric ya null values hain")

    probs = score(feats)
    out = df.copy()
    out["fraud_probability"] = np.round(probs, 6)
    out["decision"] = np.where(probs >= t, "FRAUD", "LEGIT")
    buf = io.StringIO()
    out.to_csv(buf, index=False)
    return StreamingResponse(
        iter([buf.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=scored.csv"},
    )
