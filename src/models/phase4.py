import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss
from sklearn.model_selection import TimeSeriesSplit

from src.data.load import load_config
from src.models.tune import build_pipeline

C_FP = 5.0   # ek legit transaction ko manually review karne ka cost (Amount ki unit me). Apne hisaab se badlo
FIG = Path("reports/figures")


def feature_names():
    # preprocess.py ka output order: log_amount, V1..V28, hour_sin, hour_cos
    return ["log_amount"] + [f"V{i}" for i in range(1, 29)] + ["hour_sin", "hour_cos"]


def main():
    cfg = load_config()
    seed, target = cfg["seed"], cfg["columns"]["target"]
    proc = Path(cfg["paths"]["processed_dir"])
    train = pd.read_csv(proc / "train.csv")
    val = pd.read_csv(proc / "val.csv")
    X_tr, y_tr = train.drop(columns=[target]), train[target]
    X_va, y_va = val.drop(columns=[target]), val[target]
    yv = y_va.to_numpy()
    amount_va = X_va["Amount"].to_numpy()

    with open("reports/phase3_best_params.json") as f:
        best = json.load(f)
    name = best["selected"]
    params = best["params"][name]
    print("Model:", name)

    # 1) Calibration: raw vs sigmoid vs isotonic (train par, time-aware CV ke saath)
    base = build_pipeline(name, params, seed).fit(X_tr, y_tr)
    variants = {"raw": base}
    for method in ["sigmoid", "isotonic"]:
        cal = CalibratedClassifierCV(
            estimator=build_pipeline(name, params, seed),
            method=method,
            cv=TimeSeriesSplit(n_splits=4),
        )
        variants[method] = cal.fit(X_tr, y_tr)

    scores, rows = {}, []
    for k, m in variants.items():
        s = m.predict_proba(X_va)[:, 1]
        scores[k] = s
        rows.append({
            "variant": k,
            "pr_auc": average_precision_score(yv, s),
            "brier": brier_score_loss(yv, s),
            "log_loss": log_loss(yv, s),
        })
    tab = pd.DataFrame(rows)
    print("\n=== CALIBRATION (val) ===")
    print(tab.round(5).to_string(index=False))
    chosen = tab.sort_values("brier").iloc[0]["variant"]
    print("Chosen (lowest Brier):", chosen)

    plt.figure(figsize=(6, 6))
    for k, s in scores.items():
        frac, mean = calibration_curve(yv, s, n_bins=8, strategy="quantile")
        plt.plot(mean, frac, marker="o", label=k)
    plt.plot([0, 1], [0, 1], "k--", label="perfect")
    plt.xlabel("Predicted probability"); plt.ylabel("Observed fraud rate")
    plt.title("Calibration (val)"); plt.legend(); plt.grid(alpha=0.3)
    plt.savefig(FIG / "calibration_curve.png", dpi=150, bbox_inches="tight")
    plt.close()

    # 2) Cost-based threshold (val par)
    s = scores[chosen]
    ths = np.linspace(0.01, 0.99, 197)

    def total_cost(t):
        pred = s >= t
        fn_cost = amount_va[(yv == 1) & ~pred].sum()      # miss hua fraud = uska Amount
        fp_cost = ((yv == 0) & pred).sum() * C_FP         # galat alarm = review cost
        return fn_cost + fp_cost

    costs = np.array([total_cost(t) for t in ths])
    best_t = float(ths[costs.argmin()])
    no_model = float(amount_va[yv == 1].sum())
    c_default, c_best = total_cost(0.5), float(costs.min())
    near = ths[costs <= c_best * 1.10]

    pred = s >= best_t
    tp = int(((yv == 1) & pred).sum()); fp = int(((yv == 0) & pred).sum()); fn = int(((yv == 1) & ~pred).sum())
    print("\n=== THRESHOLD (val) ===")
    print(f"C_FP = {C_FP}")
    print(f"Cost without model (all fraud missed): {no_model:,.0f}")
    print(f"Cost at threshold 0.50:                {c_default:,.0f}")
    print(f"Cost at best threshold {best_t:.3f}:          {c_best:,.0f}")
    print(f"Reduction vs 0.50: {100 * (1 - c_best / c_default):.1f}% | vs no model: {100 * (1 - c_best / no_model):.1f}%")
    print(f"TP={tp} FP={fp} FN={fn}  precision={tp / max(tp + fp, 1):.3f} recall={tp / max(tp + fn, 1):.3f}")
    print(f"Thresholds within 10% of best cost: {near.min():.2f} to {near.max():.2f}")

    plt.figure(figsize=(8, 4))
    plt.plot(ths, costs)
    plt.axvline(best_t, color="r", ls="--", label=f"best = {best_t:.2f}")
    plt.axvline(0.5, color="gray", ls=":", label="default 0.5")
    plt.xlabel("Threshold"); plt.ylabel("Total cost (val)")
    plt.title("Cost vs threshold"); plt.legend(); plt.grid(alpha=0.3)
    plt.savefig(FIG / "cost_curve.png", dpi=150, bbox_inches="tight")
    plt.close()

    # 3) Explainability (xgboost ka exact TreeSHAP)
    if name == "xgboost":
        rng = np.random.default_rng(seed)
        fraud_idx = np.where(yv == 1)[0]
        legit_idx = np.where(yv == 0)[0]
        idx = np.concatenate([fraud_idx, rng.choice(legit_idx, 4000, replace=False)])
        Xs = X_va.iloc[idx]
        Xt = base[:-1].transform(Xs)
        booster = base.named_steps["model"].get_booster()
        contrib = booster.predict(xgb.DMatrix(Xt), pred_contribs=True)[:, :-1]
        names = feature_names()
        assert contrib.shape[1] == len(names)

        imp = np.abs(contrib).mean(axis=0)
        order = np.argsort(imp)[::-1][:15][::-1]
        plt.figure(figsize=(7, 6))
        plt.barh([names[i] for i in order], imp[order])
        plt.title("Mean |SHAP| (val sample: all fraud + 4000 legit)")
        plt.savefig(FIG / "shap_bar.png", dpi=150, bbox_inches="tight")
        plt.close()

        nf = len(fraud_idx)
        sb = base.predict_proba(Xs)[:, 1]
        print("\n=== TOP REASONS for 3 highest-scoring fraud cases ===")
        for j in np.argsort(sb[:nf])[::-1][:3]:
            top = np.argsort(np.abs(contrib[j]))[::-1][:3]
            reasons = ", ".join(f"{names[i]} ({contrib[j][i]:+.2f})" for i in top)
            print(f"  score={sb[j]:.3f} -> {reasons}")
        try:
            import shap
            shap.summary_plot(contrib, Xt, feature_names=names, show=False, max_display=15)
            plt.savefig(FIG / "shap_summary.png", dpi=150, bbox_inches="tight")
            plt.close()
            print("Saved shap_summary.png")
        except Exception as e:
            print("shap beeswarm skipped:", type(e).__name__)

    # 4) Save
    Path("model").mkdir(exist_ok=True)
    joblib.dump(variants[chosen], "model/final_model.joblib")
    with open("reports/phase4_threshold.json", "w") as f:
        json.dump({"model": name, "calibration": chosen, "threshold": best_t,
                   "c_fp": C_FP, "val_cost": c_best, "val_cost_at_0.5": c_default}, f, indent=2)
    print("\nSaved model/final_model.joblib, reports/phase4_threshold.json, figures")


if __name__ == "__main__":
    main()
