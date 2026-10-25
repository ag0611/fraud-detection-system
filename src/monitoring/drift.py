from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from src.data.load import load_config

PSI_WARN, PSI_ALERT = 0.10, 0.20


def psi(ref, cur, bins: int = 10, eps: float = 1e-6) -> float:
    """Population Stability Index: reference ke quantile bins par current ka distribution."""
    inner = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1))[1:-1])
    if len(inner) < 2:
        return 0.0
    n = len(inner) + 1
    r = np.bincount(np.searchsorted(inner, ref, side="right"), minlength=n) / len(ref)
    c = np.bincount(np.searchsorted(inner, cur, side="right"), minlength=n) / len(cur)
    r, c = np.clip(r, eps, None), np.clip(c, eps, None)
    return float(np.sum((c - r) * np.log(c / r)))


def monitored_features(df: pd.DataFrame) -> pd.DataFrame:
    # Time ko skip kiya: wo hamesha badhta hai, drift ka matlab nahi
    out = df[[f"V{i}" for i in range(1, 29)]].copy()
    out["log_amount"] = np.log1p(df["Amount"])
    return out


def main():
    cfg = load_config()
    target = cfg["columns"]["target"]
    proc = Path(cfg["paths"]["processed_dir"])
    splits = {n: pd.read_csv(proc / f"{n}.csv") for n in ["train", "val", "test"]}
    ref = monitored_features(splits["train"])

    rows = []
    for name in ["val", "test"]:
        cur = monitored_features(splits[name])
        for col in ref.columns:
            ks = ks_2samp(ref[col], cur[col])
            rows.append({"split": name, "feature": col, "psi": psi(ref[col], cur[col]),
                         "ks_stat": float(ks.statistic), "ks_pvalue": float(ks.pvalue)})
    df = pd.DataFrame(rows)
    Path("reports").mkdir(exist_ok=True)
    df.round(4).to_csv("reports/drift_report.csv", index=False)

    print("=== FRAUD RATE (concept/prior drift) ===")
    for n, d in splits.items():
        print(f"  {n:5s} {d[target].mean() * 100:.3f}%  ({int(d[target].sum())} fraud / {len(d):,})")

    print("\n=== FEATURE DRIFT vs train (PSI) ===")
    for name in ["val", "test"]:
        d = df[df.split == name].sort_values("psi", ascending=False)
        warn = int((d.psi >= PSI_WARN).sum())
        alert = int((d.psi >= PSI_ALERT).sum())
        top = ", ".join(f"{r.feature} ({r.psi:.3f})" for r in d.head(5).itertuples())
        print(f"  {name}: {alert} features PSI>={PSI_ALERT}, {warn} features PSI>={PSI_WARN}")
        print(f"    top: {top}")

    flag = False
    model_path = Path("model/final_model.joblib")
    if model_path.exists():
        model = joblib.load(model_path)
        feats = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
        s_val = model.predict_proba(splits["val"][feats])[:, 1]
        s_test = model.predict_proba(splits["test"][feats])[:, 1]
        # score drift: val reference, test current
        score_psi = psi(s_val, s_test)
        print(f"\n=== SCORE DRIFT (val -> test) ===\n  PSI = {score_psi:.3f}  mean score {s_val.mean():.5f} -> {s_test.mean():.5f}")
        flag = score_psi >= PSI_ALERT

    n_alert = int(((df.split == "test") & (df.psi >= PSI_ALERT)).sum())
    flag = flag or n_alert >= 3
    print(f"\nRETRAIN FLAG: {'YES' if flag else 'no'}  (rule: score PSI>={PSI_ALERT} ya >=3 features PSI>={PSI_ALERT})")

    t = df[df.split == "test"].sort_values("psi", ascending=False).head(10).iloc[::-1]
    plt.figure(figsize=(7, 5))
    plt.barh(t.feature, t.psi)
    plt.axvline(PSI_WARN, color="orange", ls="--", label="warn 0.10")
    plt.axvline(PSI_ALERT, color="red", ls="--", label="alert 0.20")
    plt.title("Top feature drift: train -> test (PSI)")
    plt.legend()
    plt.savefig("reports/figures/drift_psi.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved reports/drift_report.csv, figures/drift_psi.png")


if __name__ == "__main__":
    main()
