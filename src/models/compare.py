import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve

from src.data.load import load_config
from src.features.preprocess import build_preprocessor
from src.models.metrics import evaluate

STRATEGIES = ["none", "class_weight", "smote", "undersample"]
MODELS = ["dummy", "logreg", "random_forest"]


def make_model(name: str, strategy: str, seed: int):
    cw = "balanced" if strategy == "class_weight" else None
    if name == "dummy":
        return DummyClassifier(strategy="prior")
    if name == "logreg":
        return LogisticRegression(max_iter=1000, class_weight=cw, random_state=seed)
    if name == "random_forest":
        cw = "balanced_subsample" if strategy == "class_weight" else None
        return RandomForestClassifier(
            n_estimators=100, min_samples_leaf=2, n_jobs=-1,
            class_weight=cw, random_state=seed,
        )
    raise ValueError(name)


def make_sampler(strategy: str, seed: int):
    if strategy == "smote":
        return SMOTE(sampling_strategy=0.1, random_state=seed)
    if strategy == "undersample":
        return RandomUnderSampler(sampling_strategy=0.1, random_state=seed)
    return None


def build_pipeline(model: str, strategy: str, seed: int) -> ImbPipeline:
    steps = list(build_preprocessor().steps)
    sampler = make_sampler(strategy, seed)
    if sampler is not None:
        steps.append(("sampler", sampler))
    steps.append(("model", make_model(model, strategy, seed)))
    return ImbPipeline(steps)


def main():
    cfg = load_config()
    seed, target = cfg["seed"], cfg["columns"]["target"]
    proc = Path(cfg["paths"]["processed_dir"])

    train = pd.read_csv(proc / "train.csv")
    val = pd.read_csv(proc / "val.csv")
    X_tr, y_tr = train.drop(columns=[target]), train[target]
    X_va, y_va = val.drop(columns=[target]), val[target]

    rows, curves = [], {}
    for model in MODELS:
        for strategy in (["none"] if model == "dummy" else STRATEGIES):
            t0 = time.time()
            pipe = build_pipeline(model, strategy, seed)
            pipe.fit(X_tr, y_tr)
            score = pipe.predict_proba(X_va)[:, 1]
            res = evaluate(y_va, score)
            res.update(model=model, strategy=strategy, secs=round(time.time() - t0, 1))
            rows.append(res)
            curves[f"{model}+{strategy}"] = score
            print(
                f"{model:14s} {strategy:13s} PR-AUC={res['pr_auc']:.3f} "
                f"[{res['pr_auc_lo']:.3f}-{res['pr_auc_hi']:.3f}] "
                f"R@P80={res['recall@p80']:.3f} ROC={res['roc_auc']:.3f} ({res['secs']}s)"
            )

    df = pd.DataFrame(rows).sort_values("pr_auc", ascending=False)
    cols = ["model", "strategy", "pr_auc", "pr_auc_lo", "pr_auc_hi", "recall@p80", "roc_auc", "secs"]
    out = Path("reports")
    out.mkdir(exist_ok=True)
    df[cols].round(4).to_csv(out / "phase2_results.csv", index=False)
    print("\n=== RANKED (val set) ===")
    print(df[cols].round(3).to_string(index=False))

    # PR curves: top 5 + dummy
    plt.figure(figsize=(8, 6))
    for key in list(df.apply(lambda r: f"{r.model}+{r.strategy}", axis=1))[:5]:
        p, r, _ = precision_recall_curve(y_va, curves[key])
        plt.plot(r, p, label=key)
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("PR curves (val, top 5)")
    plt.legend(); plt.grid(alpha=0.3)
    plt.savefig(out / "figures" / "phase2_pr_curves.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved reports/phase2_results.csv and figures/phase2_pr_curves.png")


if __name__ == "__main__":
    main()

