import io
import json
import os
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

API = os.getenv("API_URL", "http://127.0.0.1:8000")
FEATURES = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
TEST = Path("data/processed/test.csv")

st.set_page_config(page_title="Fraud Detection", layout="wide")
st.title("Credit Card Fraud Detection")


@st.cache_data(ttl=30)
def get_info():
    r = requests.get(f"{API}/model-info", timeout=10)
    r.raise_for_status()
    return r.json()


def confusion(y, p, th):
    pred = p >= th
    tp = int(((y == 1) & pred).sum())
    fp = int(((y == 0) & pred).sum())
    fn = int(((y == 1) & ~pred).sum())
    return tp, fp, fn, tp / max(tp + fp, 1), tp / max(tp + fn, 1)


try:
    info = get_info()
except Exception:
    st.error(f"API nahi mili ({API}). Dusre terminal me chalao: uvicorn src.api.main:app --port 8000")
    st.stop()

default_t = float(info["default_threshold"])
t = st.sidebar.slider("Decision threshold", 0.01, 0.99, default_t, 0.01)
st.sidebar.caption(
    f"Default {default_t:.2f} val set se chuna gaya tha. Held-out test par 0.50 ke barabar cost aaya "
    "(threshold tuning generalize nahi hui). Slider se precision/recall trade-off dekho."
)
st.sidebar.write(f"Model: **{info['model']}** ({info['calibration']})")

tab1, tab2, tab3 = st.tabs(["Single transaction", "Batch CSV", "Model report"])

with tab1:
    if TEST.exists():
        kind = st.radio("Test set se sample", ["Random fraud", "Random legit"], horizontal=True)
        if st.button("Load sample"):
            df_test = pd.read_csv(TEST)
            pool = df_test[df_test["Class"] == (1 if kind == "Random fraud" else 0)]
            row = pool.sample(1).iloc[0]
            st.session_state["payload"] = {c: float(row[c]) for c in FEATURES}
            st.session_state["truth"] = int(row["Class"])
    with st.expander("Ya JSON paste karo"):
        txt = st.text_area("Transaction JSON (Time, V1..V28, Amount)", height=120)
        if st.button("Score JSON"):
            try:
                st.session_state["payload"] = json.loads(txt)
                st.session_state["truth"] = None
            except Exception:
                st.error("JSON valid nahi hai")

    if "payload" in st.session_state:
        r = requests.post(
            f"{API}/predict", params={"threshold": t},
            json=st.session_state["payload"], timeout=30,
        )
        if r.ok:
            res = r.json()
            c1, c2, c3 = st.columns(3)
            c1.metric("Fraud probability", f"{res['fraud_probability']:.3f}")
            c2.metric("Decision", res["decision"])
            truth = st.session_state.get("truth")
            c3.metric("Actual label", "-" if truth is None else ("FRAUD" if truth == 1 else "LEGIT"))
            if res["top_reasons"]:
                st.subheader("Top reasons (SHAP, + = fraud ki taraf)")
                st.bar_chart(pd.DataFrame(res["top_reasons"]).set_index("feature")["impact"])
                st.caption("V1-V28 anonymized PCA features hain, isliye reasons ka business meaning nahi nikalta.")
        else:
            st.error(r.text)

with tab2:
    up = st.file_uploader("CSV (Time, V1..V28, Amount; optional Class)", type="csv")
    if up and st.button("Score batch"):
        r = requests.post(
            f"{API}/predict/batch", params={"threshold": t},
            files={"file": ("in.csv", up.getvalue(), "text/csv")}, timeout=300,
        )
        if r.ok:
            st.session_state["batch"] = pd.read_csv(io.BytesIO(r.content))
        else:
            st.error(r.text)
    if "batch" in st.session_state:
        out = st.session_state["batch"]
        flagged = int((out["fraud_probability"] >= t).sum())
        st.write(f"{len(out):,} rows, **{flagged}** flagged FRAUD at threshold {t:.2f}")
        if "Class" in out.columns:
            y, p = out["Class"].to_numpy(), out["fraud_probability"].to_numpy()
            rows = []
            for th in sorted({0.5, round(t, 2)}):
                tp, fp, fn, pr, rc = confusion(y, p, th)
                rows.append({"threshold": th, "TP": tp, "FP": fp, "FN": fn,
                             "precision": round(pr, 3), "recall": round(rc, 3)})
            st.table(pd.DataFrame(rows))
        st.dataframe(out.sort_values("fraud_probability", ascending=False).head(50))
        st.download_button("Download scored CSV", out.to_csv(index=False), "scored.csv", "text/csv")

with tab3:
    m = info.get("test_metrics")
    if m:
        st.subheader("Held-out test set (frozen model, ek baar evaluate)")
        c1, c2, c3 = st.columns(3)
        c1.metric("PR-AUC", f"{m['pr_auc']:.3f}", f"CI {m['pr_auc_lo']:.2f}-{m['pr_auc_hi']:.2f}", delta_color="off")
        c2.metric("Recall @ Precision>=0.8", f"{m['recall@p80']:.3f}")
        c3.metric("ROC-AUC", f"{m['roc_auc']:.3f}")
    cols = st.columns(2)
    for i, name in enumerate(["test_confusion_matrix", "test_pr_curve", "shap_summary", "cost_curve"]):
        p = Path(f"reports/figures/{name}.png")
        if p.exists():
            cols[i % 2].image(str(p), caption=name)
