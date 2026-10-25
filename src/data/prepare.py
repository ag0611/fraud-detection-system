from pathlib import Path

from src.data.load import load_config, load_raw
from src.data.split import time_split
from src.data.validate import drop_duplicates, validate


def main():
    cfg = load_config()
    target = cfg["columns"]["target"]

    df = load_raw(cfg["paths"]["raw_data"])
    report = validate(df, target)
    print("RAW REPORT:", report)

    df = drop_duplicates(df)
    print(f"After dedupe: {len(df)} rows, fraud = {int(df[target].sum())}")

    train, val, test = time_split(
        df,
        time_col=cfg["columns"]["time"],
        train_frac=cfg["split"]["train_frac"],
        val_frac=cfg["split"]["val_frac"],
    )

    out = Path(cfg["paths"]["processed_dir"])
    out.mkdir(parents=True, exist_ok=True)

    for name, part in [("train", train), ("val", val), ("test", test)]:
        part.to_csv(out / f"{name}.csv", index=False)
        print(
            f"{name:5s} -> rows: {len(part):>7,} | "
            f"fraud: {int(part[target].sum()):>4} | "
            f"fraud%: {part[target].mean() * 100:.3f}"
        )


if __name__ == "__main__":
    main()
