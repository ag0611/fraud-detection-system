import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import xgboost as xgb

HERE = Path(__file__).resolve().parent
FIG = HERE.parent / "reports" / "figures"
REPORT = HERE.parent / "reports" / "final_test_report.json"
V_COLS = [f"V{i}" for i in range(1, 29)]
FEATURES = ["Time"] + V_COLS + ["Amount"]
FEATURE_NAMES = ["log_amount"] + V_COLS + ["hour_sin", "hour_cos"]
MAX_ROWS = 100_000

st.set_page_config(page_title="Fraud Detection", layout="wide")


@st.cache_resource
def load_artifacts():
    booster = xgb.Booster()
    booster.load_model(str(HERE / "xgb_model.json"))
    meta = json.loads((HERE / "preprocess.json").read_text())
    sample = pd.read_csv(HERE / "test_sample.csv")
    report = json.loads(REPORT.read_text()) if REPORT.exists() else None
    return booster, meta, sample, report


booster, meta, sample, report = load_artifacts()


def make_features(df):
    hour = (df["Time"] // 3600) % 24
    out = pd.DataFrame({"log_amount": (np.log1p(df["Amount"]) - meta["center"]) / meta["scale"]})
    for c in V_COLS:
        out[c] = df[c].to_numpy()
    out["hour_sin"] = np.sin(2 * np.pi * hour / 24).to_numpy()
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24).to_numpy()
    return out


def score(df, reasons=False):
    dm = xgb.DMatrix(make_features(df[FEATURES]).to_numpy(dtype=float))
    probs = booster.predict(dm)
    if not reasons:
        return probs
    contrib = booster.predict(dm, pred_contribs=True)[:, :-1]
    return probs, contrib


def confusion(y, p, th):
    pred = p >= th
    tp = int(((y == 1) & pred).sum())
    fp = int(((y == 0) & pred).sum())
    fn = int(((y == 1) & ~pred).sum())
    return tp, fp, fn, tp / max(tp + fp, 1), tp / max(tp + fn, 1)


st.title("Credit Card Fraud Detection")
st.caption("Demo on the public Kaggle ULB credit card dataset. Not for real financial decisions.")

default_t = float(meta["threshold"])
t = st.sidebar.slider("Decision threshold", 0.01, 0.99, default_t, 0.01)
st.sidebar.caption(
    f"Default {default_t:.2f} was picked on the validation set. On the held-out test set it did not "
    "beat 0.50 (see Model report). Move the slider to see the precision/recall trade-off."
)
st.sidebar.write(f"Model: **{meta['model']}** (raw probabilities)")

tab1, tab2, tab3 = st.tabs(["Single transaction", "Batch CSV", "Model report"])

with tab1:
    kind = st.radio("Sample from the held-out test set", ["Random fraud", "Random legit"], horizontal=True)
    if st.button("Load sample"):
        pool = sample[sample["Class"] == (1 if kind == "Random fraud" else 0)]
        row = pool.sample(1).iloc[0]
        st.session_state["payload"] = {c: float(row[c]) for c in FEATURES}
        st.session_state["truth"] = int(row["Class"])
    with st.expander("Or paste a transaction as JSON"):
        txt = st.text_area("JSON with Time, V1..V28, Amount", height=120)
        if st.button("Score JSON"):
            try:
                st.session_state["payload"] = {k: float(v) for k, v in json.loads(txt).items()}
                st.session_state["truth"] = None
            except Exception:
                st.error("Invalid JSON.")

    payload = st.session_state.get("payload")
    if payload:
        missing = [c for c in FEATURES if c not in payload]
        if missing:
            st.error(f"Missing fields: {missing}")
        else:
            probs, contrib = score(pd.DataFrame([payload]), reasons=True)
            p = float(probs[0])
            c1, c2, c3 = st.columns(3)
            c1.metric("Fraud probability", f"{p:.3f}")
            c2.metric("Decision", "FRAUD" if p >= t else "LEGIT")
            truth = st.session_state.get("truth")
            c3.metric("Actual label", "-" if truth is None else ("FRAUD" if truth == 1 else "LEGIT"))
            top = np.argsort(np.abs(contrib[0]))[::-1][:3]
            chart = pd.DataFrame({
                "feature": [FEATURE_NAMES[i] for i in top],
                "impact": [round(float(contrib[0][i]), 3) for i in top],
            }).set_index("feature")
            st.subheader("Top reasons (SHAP, positive = pushes towards fraud)")
            st.bar_chart(chart)
            st.caption("V1-V28 are anonymized PCA components, so the reasons have no business meaning.")

with tab2:
    up = st.file_uploader("CSV with Time, V1..V28, Amount (optional Class column)", type="csv")
    if up is not None:
        df = None
        try:
            df = pd.read_csv(up)
        except Exception:
            st.error("Could not read this CSV.")
        if df is not None:
            missing = [c for c in FEATURES if c not in df.columns]
            if missing:
                st.error(f"Missing columns: {missing}")
            elif len(df) > MAX_ROWS:
                st.error(f"Maximum {MAX_ROWS:,} rows.")
            else:
                feats = df[FEATURES].apply(pd.to_numeric, errors="coerce")
                if feats.isna().any().any():
                    st.error("Feature columns contain empty or non-numeric values.")
                else:
                    probs = score(feats)
                    out = df.copy()
                    out["fraud_probability"] = np.round(probs, 6)
                    out["decision"] = np.where(probs >= t, "FRAUD", "LEGIT")
                    st.write(f"{len(out):,} rows, **{int((probs >= t).sum())}** flagged FRAUD at threshold {t:.2f}")
                    if "Class" in out.columns:
                        y = out["Class"].to_numpy()
                        rows = []
                        for th in sorted({0.5, round(t, 2)}):
                            tp, fp, fn, pr, rc = confusion(y, probs, th)
                            rows.append({"threshold": th, "TP": tp, "FP": fp, "FN": fn,
                                         "precision": round(pr, 3), "recall": round(rc, 3)})
                        st.table(pd.DataFrame(rows))
                    st.dataframe(out.sort_values("fraud_probability", ascending=False).head(50))
                    st.download_button("Download scored CSV", out.to_csv(index=False), "scored.csv", "text/csv")

with tab3:
    if report:
        m = report["metrics"]
        st.subheader("Held-out test set (frozen model, evaluated once)")
        c1, c2, c3 = st.columns(3)
        c1.metric("PR-AUC", f"{m['pr_auc']:.3f}", f"95% CI {m['pr_auc_lo']:.2f}-{m['pr_auc_hi']:.2f}", delta_color="off")
        c2.metric("Recall at Precision >= 0.80", f"{m['recall@p80']:.3f}")
        c3.metric("ROC-AUC", f"{m['roc_auc']:.3f}")
    st.markdown(
        "- Validation PR-AUC was 0.861, CV 0.783 and test 0.735: the validation set (55 fraud) was optimistic.\n"
        "- The cost-based threshold (0.16) did not generalize: on the test set its cost was 1.2% higher than at 0.50.\n"
        "- The test set has only 52 fraud cases, so the confidence interval is wide."
    )
    cols = st.columns(2)
    for i, name in enumerate(["test_confusion_matrix", "test_pr_curve", "shap_summary", "drift_psi"]):
        p = FIG / f"{name}.png"
        if p.exists():
            cols[i % 2].image(str(p), caption=name)
