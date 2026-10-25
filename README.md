# Credit Card Fraud Detection: Leakage-Aware ML Pipeline

End-to-end fraud detection on the real Kaggle ULB credit card dataset (284,807 transactions, 0.17% fraud): time-based validation, imbalance-strategy comparison, tuned gradient boosting, cost-based thresholding, SHAP explanations, a FastAPI inference service and a Streamlit dashboard.

The focus is **honest evaluation**: with only 473 fraud cases, small differences are noise, and this repo reports confidence intervals and a one-shot held-out test result rather than the best-looking validation number.

## Headline result (held-out test set, evaluated once)

| Metric | Value |
|---|---|
| PR-AUC (Average Precision) | **0.735** (95% bootstrap CI 0.612 - 0.847) |
| Recall at Precision >= 0.80 | 0.712 |
| ROC-AUC | 0.983 (misleading on imbalanced data, shown for reference) |

At threshold 0.50 on the test set (52 fraud, 42,559 transactions): 37 caught, 15 missed, 5 false alarms (precision 0.88, recall 0.71).

Validation PR-AUC was 0.861, CV PR-AUC 0.783 +/- 0.057 and test PR-AUC 0.735. The validation set (55 fraud) was optimistic; CV was the better estimate.

## What was done

1. **Data validation and dedupe:** schema/null/target checks, 1,081 duplicate rows removed (473 fraud remain).
2. **Time-based split (70/15/15):** train on the past, validate and test on the future, with automated leakage assertions and unit tests.
3. **Metrics for imbalance:** PR-AUC as the primary metric, Recall@Precision, bootstrap CIs. Accuracy is meaningless at 0.17% fraud.
4. **Imbalance strategies:** class weights vs SMOTE vs undersampling, all applied inside the training pipeline only. For Random Forest, SMOTE did not beat class weights (PR-AUC 0.871 vs 0.871) and cost about 10x more training time. Undersampling hurt Logistic Regression badly (0.54).
5. **Model selection with time-aware CV:** Optuna-tuned XGBoost (CV 0.783) and LightGBM (0.776) vs Random Forest (0.762). Differences are within one standard deviation, so no model is claimed to be clearly better. XGBoost was selected by the pre-defined rule (best CV).
6. **Calibration:** Platt and isotonic were tested; neither improved Brier score, so the raw model is used.
7. **Explainability:** exact TreeSHAP; top drivers are V14, V10, V17, V4 (anonymized PCA features, so no business meaning).
8. **Experiment tracking:** every Optuna trial logged to MLflow.
9. **Serving:** FastAPI (`/predict`, `/predict/batch`, `/health`, `/model-info`) with SHAP reasons per prediction, and a Streamlit dashboard with a threshold slider.
10. **Drift monitoring:** PSI and KS-tests per feature plus score drift (`python -m src.monitoring.drift`).

## Limitations (read these)

- **Threshold tuning did not generalize.** A cost-based threshold (0.16, assuming a review cost of 5 per false alarm) cut validation cost by 43% versus 0.50, but on the test set it cost 2,468 vs 2,438 at 0.50 (-1.2%). With about 50 fraud cases per split, threshold selection is noisy. The API default stays at the pre-registered 0.16 and is configurable.
- Cost model is an assumption (missed fraud = transaction amount, false alarm = fixed 5).
- Features are anonymized PCA components, so reason codes are not human-interpretable.
- Single dataset covering about two days; no real-world drift or adversarial adaptation tested.
- The test set has only 52 fraud cases; the confidence interval is wide.

## Run it

Download `creditcard.csv` from Kaggle (mlg-ulb/creditcardfraud) into `data/`, then:

    pip install -r requirements.txt
    python -m src.data.prepare
    python -m src.models.compare
    python -m src.models.tune
    python -m src.models.phase4
    python -m src.models.final_test
    uvicorn src.api.main:app --port 8000
    streamlit run app/dashboard.py
    pytest -q

Models are not committed; the commands above regenerate them (`model/` is git-ignored).

## Structure

    configs/        config.yaml
    src/data/       load, validate, split, prepare
    src/features/   preprocess (log Amount, hour sin/cos, RobustScaler)
    src/models/     metrics, compare, tune, phase4, final_test
    src/api/        FastAPI app and schemas
    src/monitoring/ drift (PSI, KS)
    app/            Streamlit dashboard
    tests/          unit and API tests
    reports/        results, figures

## Tech stack

Python, pandas, scikit-learn, imbalanced-learn, XGBoost, LightGBM, Optuna, MLflow, SHAP, FastAPI, Streamlit, pytest, GitHub Actions.

## Drift monitoring findings (train -> val/test)

- Fraud rate fell from 0.184% (train) to 0.129% (val) and 0.122% (test).
- Input feature drift is large: 6 features exceed PSI 0.2 on val and 7 on test (V1 1.01, V3 0.75, V28 0.53, V11 0.34, V25 0.29 on test).
- Model score drift is negligible (PSI 0.008, mean score 0.00160 to 0.00146), and test ROC-AUC (0.983) matches validation, so the retrain rule fired on features only.
- Hypothesis (not verified): since the data covers about two days, part of the feature shift is likely time-of-day mix rather than true concept drift. A production setup should treat feature drift as "investigate" and require score drift or a measured performance drop before retraining.
