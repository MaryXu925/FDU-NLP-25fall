# -*- coding: utf-8 -*-
"""Export aggregated CSVs for ABSA experiments.

This script reads the JSON metric files produced by run_experiments.py
and writes three CSVs summarizing, for each model+dataset:

- train_acc (per epoch)
- train_loss (per epoch)
- test_acc, test_f1 (per run)

You can easily extend it to include val_acc / val_f1 if desired.
"""

import csv
import json
import os
from typing import Dict, Any, List


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
METRICS_ROOT = os.path.join(BASE_DIR, "experiment_outputs", "metrics")
OUTPUT_ROOT = os.path.join(BASE_DIR, "experiment_outputs", "csv")


MODELS = ["bert_spc", "lcf_bert"]
DATASETS = ["twitter", "restaurant", "laptop"]


def load_metrics(model: str, dataset: str) -> Dict[str, Any]:
    metrics_dir = os.path.join(METRICS_ROOT, model)
    if not os.path.isdir(metrics_dir):
        raise FileNotFoundError(f"Metrics directory not found for model {model!r}: {metrics_dir}")

    # Each model/dataset currently has exactly one JSON file whose name starts with dataset-
    candidates = [
        f for f in os.listdir(metrics_dir)
        if f.startswith(dataset + "-") and f.endswith(".json")
    ]
    if not candidates:
        raise FileNotFoundError(f"No metrics JSON for {model}/{dataset} under {metrics_dir}")
    # If multiple, pick the last one lexicographically (usually newest timestamp)
    candidates.sort()
    metrics_path = os.path.join(metrics_dir, candidates[-1])

    with open(metrics_path, "r", encoding="utf-8") as f:
        return json.load(f)


def ensure_output_dir() -> None:
    os.makedirs(OUTPUT_ROOT, exist_ok=True)


def export_train_metrics(all_records: List[Dict[str, Any]]) -> None:
    """Write per-epoch train_acc and train_loss to CSV."""
    path = os.path.join(OUTPUT_ROOT, "train_metrics.csv")
    fieldnames = ["model", "dataset", "epoch", "train_acc", "train_loss"]
    with open(path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for record in all_records:
            model = record["model"]
            dataset = record["dataset"]
            history = record.get("history", [])
            for entry in history:
                writer.writerow(
                    {
                        "model": model,
                        "dataset": dataset,
                        "epoch": entry.get("epoch"),
                        "train_acc": entry.get("train_acc"),
                        "train_loss": entry.get("train_loss"),
                    }
                )
    print(f"Wrote train metrics CSV to {path}")


def export_test_metrics(all_records: List[Dict[str, Any]]) -> None:
    """Write per-run test_acc and test_f1 to CSV."""
    path = os.path.join(OUTPUT_ROOT, "test_metrics.csv")
    fieldnames = ["model", "dataset", "test_acc", "test_f1"]
    with open(path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for record in all_records:
            writer.writerow(
                {
                    "model": record["model"],
                    "dataset": record["dataset"],
                    "test_acc": record.get("test_acc"),
                    "test_f1": record.get("test_f1"),
                }
            )
    print(f"Wrote test metrics CSV to {path}")


def main() -> None:
    ensure_output_dir()

    all_records: List[Dict[str, Any]] = []
    for model in MODELS:
        for dataset in DATASETS:
            metrics = load_metrics(model, dataset)
            all_records.append(metrics)

    export_train_metrics(all_records)
    export_test_metrics(all_records)


if __name__ == "__main__":
    main()
