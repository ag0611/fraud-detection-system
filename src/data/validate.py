import pandas as pd

EXPECTED_COLUMNS = (
    ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount", "Class"]
)


def validate(df: pd.DataFrame, target: str = "Class") -> dict:
    """Basic data checks. Problem mile to error raise karta hai, warna report dict deta hai."""
    missing_cols = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns: {missing_cols}")

    null_count = int(df[EXPECTED_COLUMNS].isna().sum().sum())
    if null_count > 0:
        raise ValueError(f"Dataset me {null_count} null values hain")

    if not set(df[target].unique()).issubset({0, 1}):
        raise ValueError("Target sirf 0/1 hona chahiye")

    if (df["Amount"] < 0).any():
        raise ValueError("Negative Amount mila")

    return {
        "rows": len(df),
        "fraud_count": int(df[target].sum()),
        "fraud_rate_pct": round(float(df[target].mean() * 100), 4),
        "duplicate_rows": int(df.duplicated().sum()),
    }


def drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop_duplicates().reset_index(drop=True)
