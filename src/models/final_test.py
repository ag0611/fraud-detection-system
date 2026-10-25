import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, precision_recall_curve

from src.data.load import load_config
from src.models.metrics import evaluate


def main():
    cfg = load_config()
    target = cfg["columns"]["target"]
    test = pd.read_csv(Path(cfg["paths"]["processed_dir"]) / "test.csv")
    X, y = test.drop(columns=[target]), test[target].to_numpy()
    amount = X["Amount"].to_numpy()

    model = joblib.load("model/final_model.joblib")
    with open("reports/phase4_threshold.json") as f:
        cfg4 = json.load(f)
    t, c_fp = cfg4["threshold"], cfg4["c_fp"]
    s = model.predict_proba(X)[:, 1]

    res = evaluate(y, s)
    print("=== FINAL TEST (frozen model, ek baar) ===")
    print(f"Test rows={len(y)}, fraud={int(y.sum())}")
    print(f"PR-AUC {res['pr_auc']:.3f}  95% CI [{res['pr_auc_lo']:.3f}, {res['pr_auc_hi']:.3f}]")
    print(f"Recall@Precision>=0.80: {res['recall@p80']:.3f} | ROC-AUC: {res['roc_auc']:.3f}")

    def stats(th):
        pred = s >= th
        tp = int(((y == 1) & pred).sum())
        fp = int(((y == 0) & pred).sum())
        fn = int(((y == 1) & ~pred).sum())
        cost = float(amount[(y == 1) & ~pred].sum() + fp * c_fp)
        return dict(threshold=th, tp=tp, fp=fp, fn=fn,
                    precision=tp / max(tp + fp, 1), recall=tp / max(tp + fn, 1), cost=cost)

    a, b = stats(0.5), stats(t)
    no_model = float(amount[y == 1].sum())
    for tag, d in [("threshold 0.50", a), (f"threshold {t:.3f} (val se chuna)", b)]:
        print(f"\n{tag}: TP={d['tp']} FP={d['fp']} FN={d['fn']} "
              f"precision={d['precision']:.3f} recall={d['recall']:.3f} cost={d['cost']:,.0f}")
    print(f"\nCost without model: {no_model:,.0f}")
    print(f"Cost reduction vs 0.50: {100 * (1 - b['cost'] / a['cost']):.1f}%")
    print(f"Cost reduction vs no model: {100 * (1 - b['cost'] / no_model):.1f}%")

    fig_dir = Path("reports/figures")
    ConfusionMatrixDisplay.from_predictions(y, s >= t, display_labels=["legit", "fraud"], cmap="Blues")
    plt.title(f"Test confusion matrix (threshold {t:.2f})")
    plt.savefig(fig_dir / "test_confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close()
    p, r, _ = precision_recall_curve(y, s)
    plt.figure(figsize=(6, 5))
    plt.plot(r, p)
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title(f"Test PR curve (AP = {res['pr_auc']:.3f})"); plt.grid(alpha=0.3)
    plt.savefig(fig_dir / "test_pr_curve.png", dpi=150, bbox_inches="tight")
    plt.close()

    with open("reports/final_test_report.json", "w") as f:
        json.dump({"metrics": {k: float(v) for k, v in res.items()},
                   "at_0.5": a, "at_best": b, "no_model_cost": no_model}, f, indent=2)
    print("\nSaved reports/final_test_report.json + test plots")


if __name__ == "__main__":
    main()
