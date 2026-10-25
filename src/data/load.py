from pathlib import Path

import pandas as pd
import yaml


def load_config(path: str = "configs/config.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_raw(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"{p} nahi mila. Kaggle se creditcard.csv download karke data/ me rakho."
        )
    return pd.read_csv(p)
