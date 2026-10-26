import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

from src.data.load import load_config

V_COLS = [f"V{i}" for i in range(1, 29)]


def make_features(df, center, scale):
    hour = (df["Time"] // 3600) % 24
    out = pd.DataFrame({"log_amount": (np.log1p(df["Amount"]) - center) / scale})
    for c in V_COLS:
        out[c] = df[c].to_numpy()
    out["hour_sin"] = np.sin(2 * np.pi * hour / 24).to_numpy()
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24).to_numpy()
    return out


def main():
    cfg = load_config()
    target = cfg["columns"]["target"]
    proc = Path(cfg["paths"]["processed_dir"])
    out = Path("deploy")
    out.mkdir(exist_ok=True)

    model = joblib.load("model/final_model.joblib")
    est = model.named_steps["model"]
    scaler = model.named_steps["scale"].named_transformers_["amount"]
    est.get_booster().save_model(str(out / "xgb_model.json"))

    thr = json.loads(Path("reports/phase4_threshold.json").read_text())
    meta = {
        "center": float(scaler.center_[0]),
        "scale": float(scaler.scale_[0]),
        "threshold": float(thr["threshold"]),
        "model": thr["model"],
    }
    (out / "preprocess.json").write_text(json.dumps(meta, indent=2))

    # Sample for the demo: all 52 test fraud + 450 random legit
    test = pd.read_csv(proc / "test.csv")
    fraud = test[test[target] == 1]
    legit = test[test[target] == 0].sample(450, random_state=cfg["seed"])
    sample = pd.concat([fraud, legit]).sample(frac=1, random_state=cfg["seed"]).reset_index(drop=True)
    sample.to_csv(out / "test_sample.csv", index=False)

    # Sanity check: standalone booster must match the original pipeline
    booster = xgb.Booster()
    booster.load_model(str(out / "xgb_model.json"))
    feats = make_features(sample, meta["center"], meta["scale"])
    p_new = booster.predict(xgb.DMatrix(feats.to_numpy(dtype=float)))
    p_old = model.predict_proba(sample.drop(columns=[target]))[:, 1]
    diff = float(np.abs(p_new - p_old).max())
    print(f"max |difference| vs original pipeline: {diff:.2e}")
    assert diff < 1e-4, "Exported model does not match the pipeline"

    for p in sorted(out.iterdir()):
        print(f"  {p.name}: {p.stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    main()
