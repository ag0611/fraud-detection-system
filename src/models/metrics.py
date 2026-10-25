import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score


def recall_at_precision(y_true, y_score, min_precision: float = 0.80) -> float:
    """Kam se kam min_precision rakhte hue max recall."""
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    mask = precision >= min_precision
    return float(recall[mask].max()) if mask.any() else 0.0


def bootstrap_ap_ci(y_true, y_score, n_boot: int = 200, seed: int = 42):
    """PR-AUC ka 95% bootstrap confidence interval (fraud kam hain, isliye noisy)."""
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    n = len(y_true)
    scores = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if y_true[idx].sum() == 0:
            continue
        scores.append(average_precision_score(y_true[idx], y_score[idx]))
    return float(np.percentile(scores, 2.5)), float(np.percentile(scores, 97.5))


def evaluate(y_true, y_score) -> dict:
    lo, hi = bootstrap_ap_ci(y_true, y_score)
    return {
        "pr_auc": average_precision_score(y_true, y_score),
        "pr_auc_lo": lo,
        "pr_auc_hi": hi,
        "recall@p80": recall_at_precision(y_true, y_score, 0.80),
        "roc_auc": roc_auc_score(y_true, y_score),
    }
