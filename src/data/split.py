import pandas as pd


def time_split(
    df: pd.DataFrame,
    time_col: str = "Time",
    train_frac: float = 0.70,
    val_frac: float = 0.15,
):
    """Chronological split: train = past, val = beech, test = future.
    Random split nahi, taaki future ka data training me leak na ho."""
    if train_frac + val_frac >= 1:
        raise ValueError("train_frac + val_frac 1 se kam hona chahiye")

    df = df.sort_values(time_col, kind="stable").reset_index(drop=True)
    n = len(df)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))

    train = df.iloc[:train_end].copy()
    val = df.iloc[train_end:val_end].copy()
    test = df.iloc[val_end:].copy()

    assert train[time_col].max() <= val[time_col].min(), "Train/Val time leakage"
    assert val[time_col].max() <= test[time_col].min(), "Val/Test time leakage"

    return train, val, test
