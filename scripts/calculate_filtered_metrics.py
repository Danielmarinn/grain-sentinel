#!/usr/bin/env python3
import argparse
from pathlib import Path

import pandas as pd


def parse_bool_series(values: pd.Series) -> pd.Series:
    if values.dtype == bool:
        return values
    return values.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def compute_metrics(ground_truth_path: Path, final_anomalies_path: Path) -> dict[str, float | int]:
    truth_df = pd.read_csv(ground_truth_path)
    anomalies_df = pd.read_csv(final_anomalies_path)

    if "timestamp" not in truth_df.columns:
        raise ValueError("Ground truth CSV must contain a timestamp column.")
    required = {"timestamp", "kept"}
    missing = required - set(anomalies_df.columns)
    if missing:
        raise ValueError(f"Final anomalies CSV is missing columns: {sorted(missing)}")

    truth = set(pd.to_datetime(truth_df["timestamp"]))
    kept = anomalies_df.loc[parse_bool_series(anomalies_df["kept"])].copy()
    flagged = set(pd.to_datetime(kept["timestamp"]))

    true_positives = len(flagged & truth)
    false_positives = len(flagged - truth)
    false_negatives = len(truth - flagged)
    precision = true_positives / (true_positives + false_positives) if true_positives + false_positives else 0.0
    recall = true_positives / (true_positives + false_negatives) if true_positives + false_negatives else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    return {
        "ground_truth_count": len(truth),
        "candidate_count": int(len(anomalies_df)),
        "final_flagged_count": int(len(flagged)),
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def markdown_table(metrics: dict[str, float | int]) -> str:
    return "\n".join(
        [
            "| metric | value |",
            "| --- | ---: |",
            f"| ground_truth_count | {metrics['ground_truth_count']} |",
            f"| candidate_count | {metrics['candidate_count']} |",
            f"| final_flagged_count | {metrics['final_flagged_count']} |",
            f"| true_positives | {metrics['true_positives']} |",
            f"| false_positives | {metrics['false_positives']} |",
            f"| false_negatives | {metrics['false_negatives']} |",
            f"| precision | {metrics['precision']:.6f} |",
            f"| recall | {metrics['recall']:.6f} |",
            f"| f1 | {metrics['f1']:.6f} |",
        ]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Recompute filtered detector metrics from audit CSVs.")
    parser.add_argument("--ground-truth", type=Path, default=Path("data/processed/ground_truth_timestamps.csv"))
    parser.add_argument("--final-anomalies", type=Path, default=Path("data/processed/final_anomalies.csv"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/filtered_metrics.txt"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = compute_metrics(args.ground_truth, args.final_anomalies)
    table = markdown_table(metrics)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(table + "\n", encoding="utf-8")
    print(table)


if __name__ == "__main__":
    main()
