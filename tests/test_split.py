import pandas as pd

from src.data.split import time_split


def make_df(n=1000):
    return pd.DataFrame({"Time": range(n), "Amount": 1.0, "Class": [0] * n})


def test_no_time_leakage():
    train, val, test = time_split(make_df())
    assert train["Time"].max() <= val["Time"].min()
    assert val["Time"].max() <= test["Time"].min()


def test_sizes_add_up():
    df = make_df()
    train, val, test = time_split(df)
    assert len(train) + len(val) + len(test) == len(df)
