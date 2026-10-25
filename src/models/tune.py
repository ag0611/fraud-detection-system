import json
import time
from pathlib import Path

import joblib
import mlflow
import optuna
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.data.load import load_config
from src.features.preprocess import build_preprocessor
from src.models.metrics import evaluate

N_TRIALS = 20    # test run. Full run ke liye 20 kar dena
N_SPLITS = 4


def suggest_params(name, trial):
    p = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 400),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1.0, 600.0, log=True),
    }
    if name == "xgboost":
        p["min_child_weight"] = trial.suggest_int("min_child_weight", 1, 10)
    else:
        p["min_child_samples"] = trial.suggest_int("min_child_samples", 5, 50)
        p["num_leaves"] = trial.suggest_int("num_leaves", 15, 127)
        p["subsample_freq"] = 1
    return p


def make_estimator(name, params, seed):
    if name == "xgboost":
        return XGBClassifier(
            **params, tree_method="hist", eval_metric="aucpr",
            n_jobs=-1, random_state=seed,
        )
    if name == "lightgbm":
        return LGBMClassifier(**params, n_jobs=-1, random_state=seed, verbose=-1)
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=100, min_samples_leaf=2, class_weight="balanced_subsample",
            n_jobs=-1, random_state=seed,
        )
    raise ValueError(name)


def build_pipeline(name, params, seed):
    steps = list(build_preprocessor().steps)
    steps.append(("model", make_estimator(name, params, seed)))
    return Pipeline(steps)


def cv_score(name, params, X, y, seed):
    """Time-aware CV: har fold me train = past, valid = future. Mean/std PR-AUC."""
    scores = []
    for tr_idx, va_idx in TimeSeriesSplit(n_splits=N_SPLITS).split(X):
        y_va = y.iloc[va_idx]
        if y_va.sum() == 0:
            continue
        pipe = build_pipeline(name, params, seed)
        pipe.fit(X.iloc[tr_idx], y.iloc[tr_idx])
        s = pipe.predict_proba(X.iloc[va_idx])[:, 1]
        scores.append(average_precision_score(y_va, s))
    return float(pd.Series(scores).mean()), float(pd.Series(scores).std())


def main():
    cfg = load_config()
    seed, target = cfg["seed"], cfg["columns"]["target"]
    proc = Path(cfg["paths"]["processed_dir"])

    train = pd.read_csv(proc / "train.csv")   # Time-sorted hai (split.py ne sort kiya)
    val = pd.read_csv(proc / "val.csv")
    X_tr, y_tr = train.drop(columns=[target]), train[target]
    X_va, y_va = val.drop(columns=[target]), val[target]

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("fraud-detection")

    candidates = {"random_forest": {}}
    with mlflow.start_run(run_name="phase3-tuning"):
        for name in ["xgboost", "lightgbm"]:
            print(f"\n--- Tuning {name} ({N_TRIALS} trials, {N_SPLITS}-fold time CV) ---")

            def objective(trial, name=name):
                params = suggest_params(name, trial)
                trial.set_user_attr("full", params)
                t0 = time.time()
                mean, std = cv_score(name, params, X_tr, y_tr, seed)
                with mlflow.start_run(run_name=f"{name}-trial-{trial.number}", nested=True):
                    mlflow.log_params(params)
                    mlflow.log_metric("cv_pr_auc", mean)
                    mlflow.log_metric("cv_pr_auc_std", std)
                print(f"  trial {trial.number:2d}  cv PR-AUC={mean:.3f} +/- {std:.3f}  ({time.time() - t0:.0f}s)")
                return mean

            study = optuna.create_study(
                direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed)
            )
            study.optimize(objective, n_trials=N_TRIALS)
            candidates[name] = study.best_trial.user_attrs["full"]
            print(f"  best cv PR-AUC for {name}: {study.best_value:.3f}")

        # Final comparison: CV (selection ke liye) + val (sirf report)
        rows, fitted = [], {}
        for name, params in candidates.items():
            cv_mean, cv_std = cv_score(name, params, X_tr, y_tr, seed)
            pipe = build_pipeline(name, params, seed)
            pipe.fit(X_tr, y_tr)
            res = evaluate(y_va, pipe.predict_proba(X_va)[:, 1])
            res.update(model=name, cv_pr_auc=cv_mean, cv_std=cv_std)
            rows.append(res)
            fitted[name] = pipe
            with mlflow.start_run(run_name=f"final-{name}", nested=True):
                mlflow.log_params({k: str(v) for k, v in params.items()})
                mlflow.log_metrics({
                    "cv_pr_auc": cv_mean, "val_pr_auc": res["pr_auc"],
                    "val_recall_at_p80": res["recall@p80"], "val_roc_auc": res["roc_auc"],
                })

        df = pd.DataFrame(rows).sort_values("cv_pr_auc", ascending=False)
        cols = ["model", "cv_pr_auc", "cv_std", "pr_auc", "pr_auc_lo", "pr_auc_hi",
                "recall@p80", "roc_auc"]
        print("\n=== FINAL (selection = CV PR-AUC, val = report only) ===")
        print(df[cols].round(3).to_string(index=False))

        best = df.iloc[0]["model"]
        Path("model").mkdir(exist_ok=True)
        joblib.dump(fitted[best], "model/best_model.joblib")
        mlflow.log_artifact("model/best_model.joblib")
        df[cols].round(4).to_csv("reports/phase3_results.csv", index=False)
        with open("reports/phase3_best_params.json", "w") as f:
            json.dump({"selected": best, "params": candidates}, f, indent=2)
        print(f"\nSelected (best CV): {best}")
        print("Saved model/best_model.joblib, reports/phase3_results.csv, reports/phase3_best_params.json")


if __name__ == "__main__":
    main()

