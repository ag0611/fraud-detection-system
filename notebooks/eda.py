from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from src.data.load import load_config, load_raw

cfg = load_config()
fig_dir = Path(cfg["paths"]["figures_dir"])
fig_dir.mkdir(parents=True, exist_ok=True)

df = load_raw(cfg["paths"]["raw_data"]).drop_duplicates()

# 1. Class imbalance
ax = sns.countplot(x="Class", data=df)
ax.set_yscale("log")
ax.set_title("Class distribution (log scale)")
plt.savefig(fig_dir / "class_distribution.png", dpi=150, bbox_inches="tight")
plt.close()

# 2. Amount: legit vs fraud
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
sns.histplot(np.log1p(df.loc[df.Class == 0, "Amount"]), bins=50, ax=axes[0], color="steelblue")
axes[0].set_title("log1p(Amount) - Legit")
sns.histplot(np.log1p(df.loc[df.Class == 1, "Amount"]), bins=50, ax=axes[1], color="crimson")
axes[1].set_title("log1p(Amount) - Fraud")
plt.savefig(fig_dir / "amount_dist.png", dpi=150, bbox_inches="tight")
plt.close()
print(df.groupby("Class")["Amount"].describe())

# 3. Fraud rate by hour
df["hour"] = (df["Time"] // 3600) % 24
(df.groupby("hour")["Class"].mean() * 100).plot(
    kind="bar", figsize=(10, 4), title="Fraud rate (%) by hour"
)
plt.savefig(fig_dir / "fraud_by_hour.png", dpi=150, bbox_inches="tight")
plt.close()

# 4. Correlation of V features with Class
v_cols = [c for c in df.columns if c.startswith("V")]
corr = df[v_cols + ["Class"]].corr()["Class"].drop("Class").sort_values()
corr.plot(kind="barh", figsize=(7, 8), title="Correlation with Class")
plt.savefig(fig_dir / "feature_corr.png", dpi=150, bbox_inches="tight")
plt.close()
print("Top +ve:", corr.tail(5).round(3).to_dict())
print("Top -ve:", corr.head(5).round(3).to_dict())

# 5. Top features KDE
top = corr.abs().sort_values(ascending=False).index[:4]
fig, axes = plt.subplots(2, 2, figsize=(11, 7))
for ax, col in zip(axes.ravel(), top):
    sns.kdeplot(df.loc[df.Class == 0, col], ax=ax, label="Legit", fill=True)
    sns.kdeplot(df.loc[df.Class == 1, col], ax=ax, label="Fraud", fill=True)
    ax.set_title(col)
    ax.legend()
plt.tight_layout()
plt.savefig(fig_dir / "top_features_kde.png", dpi=150, bbox_inches="tight")
plt.close()
print("Plots saved in", fig_dir)
