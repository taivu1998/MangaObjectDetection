from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from .constants import DEFAULT_METADATA_ROOT


def split_dataframe(df: pd.DataFrame, train_ratio=0.8, valid_ratio=0.1, test_ratio=0.1, random_state=0):
    if abs((train_ratio + valid_ratio + test_ratio) - 1.0) > 1e-8:
        raise ValueError("train_ratio + valid_ratio + test_ratio must equal 1.0")
    if df.empty:
        raise ValueError("Cannot split an empty dataframe")

    train_valid, test = train_test_split(df, test_size=test_ratio, random_state=random_state)
    valid_size = valid_ratio / (train_ratio + valid_ratio)
    train, valid = train_test_split(train_valid, test_size=valid_size, random_state=random_state)
    return train.reset_index(drop=True), valid.reset_index(drop=True), test.reset_index(drop=True)


def write_split_family(input_path: Path, output_dir: Path, prefix: str, random_state: int = 0):
    df = pd.read_pickle(input_path)
    train_df, valid_df, test_df = split_dataframe(df, random_state=random_state)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        f"{prefix}_train.pkl": train_df,
        f"{prefix}_valid.pkl": valid_df,
        f"{prefix}_test.pkl": test_df,
    }
    for filename, split_df in outputs.items():
        split_df.to_pickle(output_dir / filename)
    return outputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Split metadata pickle files into train/valid/test.")
    parser.add_argument("--input-dir", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_METADATA_ROOT)
    parser.add_argument("--random-state", type=int, default=0)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    families = ["data_full", "data_condensed", "data_condensed_fixed"]
    wrote_any = False
    for family in families:
        input_path = args.input_dir / f"{family}.pkl"
        if not input_path.exists():
            continue
        outputs = write_split_family(input_path, args.output_dir, family, random_state=args.random_state)
        wrote_any = True
        print(f"Created splits for {family}:")
        for filename, split_df in outputs.items():
            print(f"- {filename}: {len(split_df)} rows")
    if not wrote_any:
        raise FileNotFoundError(
            f"No metadata pickle files were found in {args.input_dir}. Expected one of: {', '.join(families)}.pkl"
        )


if __name__ == "__main__":  # pragma: no cover
    main()
