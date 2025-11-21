import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict
from sklearn.metrics import f1_score, precision_recall_fscore_support
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)


def _load_dataframe(path: str, text_column: str, label_columns: Sequence[str]) -> pd.DataFrame:
    file_path = Path(path)
    suffix = file_path.suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(file_path)
    elif suffix in {".json", ".jsonl"}:
        df = pd.read_json(file_path, lines=suffix == ".jsonl")
    else:
        raise ValueError(f"Unsupported dataset format for '{path}'. Use CSV/JSON/JSONL.")

    required = [text_column] + list(label_columns)
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Columns {missing} not found in '{path}'.")

    df = df[required]
    df = df.dropna(subset=required)
    return df


def _to_dataset_dict(
    train_path: str,
    eval_path: str,
    text_column: str,
    label_columns: Sequence[str],
) -> DatasetDict:
    data_splits: Dict[str, Dataset] = {}
    train_df = _load_dataframe(train_path, text_column, label_columns)
    data_splits["train"] = Dataset.from_pandas(train_df, preserve_index=False)
    if eval_path:
        eval_df = _load_dataframe(eval_path, text_column, label_columns)
        data_splits["validation"] = Dataset.from_pandas(eval_df, preserve_index=False)
    return DatasetDict(data_splits)


def _build_tokenize_fn(tokenizer, text_column: str, label_columns: Sequence[str], max_length: int):
    def tokenize(batch):
        encoded = tokenizer(
            batch[text_column],
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )
        labels: List[List[float]] = []
        for row_idx in range(len(batch[text_column])):
            row_values = []
            for column in label_columns:
                value = batch[column][row_idx]
                # Accept ints, floats, bools, and "0"/"1" strings.
                if isinstance(value, str):
                    value = value.strip()
                row_values.append(float(value))
            labels.append(row_values)
        encoded["labels"] = labels
        return encoded

    return tokenize


def _build_metrics_fn(threshold: float, label_count: int):
    threshold = float(threshold)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        probs = 1.0 / (1.0 + np.exp(-logits))
        preds = (probs >= threshold).astype(int)
        labels = labels.astype(int)
        micro_f1 = f1_score(labels, preds, average="micro", zero_division=0)
        macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
        weighted_precision, weighted_recall, _, _ = precision_recall_fscore_support(
            labels, preds, average="weighted", zero_division=0
        )
        per_label_recall = precision_recall_fscore_support(
            labels, preds, average=None, zero_division=0
        )[1]
        metrics = {
            "micro_f1": micro_f1,
            "macro_f1": macro_f1,
            "weighted_precision": weighted_precision,
            "weighted_recall": weighted_recall,
        }
        for idx in range(label_count):
            metrics[f"recall_label_{idx}"] = float(per_label_recall[idx])
        return metrics

    return compute_metrics


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune a multi-label mental-health classifier (sigmoid/BCE)."
    )
    parser.add_argument(
        "--train-file",
        required=True,
        help="CSV/JSON/JSONL file containing the training split.",
    )
    parser.add_argument(
        "--eval-file",
        help="Optional CSV/JSON/JSONL file for evaluation split.",
    )
    parser.add_argument(
        "--text-column",
        default="text",
        help="Column that holds the free-form text.",
    )
    parser.add_argument(
        "--label-columns",
        required=True,
        help="Comma-separated list of binary columns (0/1) representing each mental health label.",
    )
    parser.add_argument(
        "--base-model",
        default="distilroberta-base",
        help="Backbone checkpoint to fine-tune.",
    )
    parser.add_argument(
        "--output-dir",
        default="models/mental-model",
        help="Directory to save the fine-tuned model.",
    )
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--threshold", type=float, default=0.5, help="Decision threshold used for validation metrics.")
    parser.add_argument("--logging-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    label_columns = [col.strip() for col in args.label_columns.split(",") if col.strip()]
    if not label_columns:
        raise ValueError("Provide at least one label column via --label-columns.")

    dataset = _to_dataset_dict(args.train_file, args.eval_file, args.text_column, label_columns)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    tokenize_fn = _build_tokenize_fn(tokenizer, args.text_column, label_columns, args.max_length)
    remove_columns = dataset["train"].column_names
    tokenized = dataset.map(
        tokenize_fn,
        batched=True,
        remove_columns=remove_columns,
    )

    config = AutoConfig.from_pretrained(
        args.base_model,
        num_labels=len(label_columns),
        problem_type="multi_label_classification",
        id2label={idx: label for idx, label in enumerate(label_columns)},
        label2id={label: idx for idx, label in enumerate(label_columns)},
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model,
        config=config,
    )

    eval_dataset = tokenized.get("validation")
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=max(1, args.batch_size),
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        eval_strategy="steps" if eval_dataset is not None else "no",
        save_strategy="steps" if eval_dataset is not None else "epoch",
        eval_steps=args.logging_steps * 2 if eval_dataset is not None else None,
        save_total_limit=2,
        load_best_model_at_end=eval_dataset is not None,
        metric_for_best_model="micro_f1",
        greater_is_better=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        compute_metrics=_build_metrics_fn(args.threshold, len(label_columns))
        if eval_dataset is not None
        else None,
    )

    trainer.train()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    metadata = {
        "text_column": args.text_column,
        "label_columns": label_columns,
        "threshold": args.threshold,
    }
    with (Path(args.output_dir) / "mental_label_config.json").open("w", encoding="utf-8") as fp:
        json.dump(metadata, fp, indent=2)
    print(f"Finished training. Model saved to {args.output_dir}")


if __name__ == "__main__":
    main()
