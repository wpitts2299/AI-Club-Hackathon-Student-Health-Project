import argparse
from pathlib import Path
from typing import Iterable, Tuple

import pandas as pd


DEPRESSION_KEYWORDS = (
    "depress",
    "empty",
    "hopeless",
    "worthless",
    "sad",
    "tear",
    "numb",
    "lost interest",
    "no interest",
    "fatigue",
)
SUICIDAL_KEYWORDS = (
    "suicide",
    "suicidal",
    "kill myself",
    "end my life",
    "want to die",
    "wish i were dead",
    "die by",
    "i want to die",
    "i wanna die",
    "take my life",
)


def _match_keywords(text: str, keywords: Iterable[str]) -> bool:
    if not isinstance(text, str):
        return False
    lowered = text.lower()
    return any(keyword in lowered for keyword in keywords)


def _prepare_dataframe(df: pd.DataFrame, text_column: str, label_column: str) -> pd.DataFrame:
    if text_column not in df.columns:
        raise ValueError(f"Column '{text_column}' not found in dataset.")
    if label_column not in df.columns:
        raise ValueError(f"Column '{label_column}' not found in dataset.")

    df = df.dropna(subset=[text_column]).copy()
    df[text_column] = df[text_column].astype(str)
    label_numeric = pd.to_numeric(df[label_column], errors="coerce").fillna(0)
    df["stress"] = (label_numeric > 0).astype(int)
    df["depression"] = df[text_column].apply(lambda text: int(_match_keywords(text, DEPRESSION_KEYWORDS)))
    df["suicidal"] = df[text_column].apply(lambda text: int(_match_keywords(text, SUICIDAL_KEYWORDS)))
    any_positive = df[["depression", "suicidal"]].sum(axis=1) > 0
    df.loc[(df["stress"] == 1) & (~any_positive), "depression"] = 1
    return df


def _train_val_split(df: pd.DataFrame, val_fraction: float, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if not 0 < val_fraction < 1:
        raise ValueError("--val-fraction must be between 0 and 1.")
    shuffled = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    val_size = int(len(shuffled) * val_fraction)
    val_size = max(1, val_size)
    val_df = shuffled.iloc[:val_size]
    train_df = shuffled.iloc[val_size:]
    if train_df.empty:
        raise ValueError("Validation fraction too high; no samples left for training.")
    return train_df, val_df


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate heuristic multi-label columns (stress/depression/suicidal) and split train/val CSVs."
    )
    parser.add_argument(
        "--input-file",
        default="dreaddit_StressAnalysis - Sheet1.csv",
        help="Source CSV containing at least the text and label columns.",
    )
    parser.add_argument(
        "--text-column",
        default="text",
        help="Column containing user text.",
    )
    parser.add_argument(
        "--label-column",
        default="label",
        help="Column containing the existing stress label (0/1).",
    )
    parser.add_argument(
        "--train-output",
        default="data/mental_train.csv",
        help="Path to save the generated training split.",
    )
    parser.add_argument(
        "--val-output",
        default="data/mental_val.csv",
        help="Path to save the generated validation split.",
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.2,
        help="Fraction of samples to reserve for validation (between 0 and 1).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for shuffling before the split.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    input_path = Path(args.input_file)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file '{input_path}' does not exist.")
    df = pd.read_csv(input_path)
    df = _prepare_dataframe(df, args.text_column, args.label_column)
    train_df, val_df = _train_val_split(df, args.val_fraction, args.seed)

    Path(args.train_output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.val_output).parent.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(args.train_output, index=False)
    val_df.to_csv(args.val_output, index=False)

    def _label_counts(frame: pd.DataFrame):
        return {
            "stress": int(frame["stress"].sum()),
            "depression": int(frame["depression"].sum()),
            "suicidal": int(frame["suicidal"].sum()),
        }

    print(f"Wrote {len(train_df)} training samples to {args.train_output} with counts { _label_counts(train_df) }")
    print(f"Wrote {len(val_df)} validation samples to {args.val_output} with counts { _label_counts(val_df) }")


if __name__ == "__main__":
    main()
