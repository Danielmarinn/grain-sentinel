#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from calculate_filtered_metrics import compute_metrics  # noqa: E402


EXPECTED = {
    "ground_truth_count": 25,
    "candidate_count": 148,
    "final_flagged_count": 42,
    "true_positives": 22,
    "false_positives": 20,
    "false_negatives": 3,
    "precision": 0.5238095238095238,
    "recall": 0.88,
    "f1": 0.6567164179104478,
}


def assert_close(name: str, actual: float | int, expected: float | int) -> None:
    if isinstance(expected, float):
        if not math.isclose(float(actual), expected, rel_tol=1e-9, abs_tol=1e-9):
            raise AssertionError(f"{name}: expected {expected}, got {actual}")
    elif actual != expected:
        raise AssertionError(f"{name}: expected {expected}, got {actual}")


def test_metrics() -> None:
    metrics = compute_metrics(
        ROOT / "data" / "processed" / "ground_truth_timestamps.csv",
        ROOT / "data" / "processed" / "final_anomalies.csv",
    )
    for key, expected in EXPECTED.items():
        assert_close(key, metrics[key], expected)


def test_detector_cli() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        output_log = Path(tmp_dir) / "alerts.jsonl"
        cmd = [
            sys.executable,
            str(ROOT / "scripts" / "detector_filtered.py"),
            str(ROOT / "data" / "processed" / "validation_with_injection.csv"),
            "--timestamp-column",
            "timestamp",
            "--sensor-column",
            "temperature",
            "--output-log",
            str(output_log),
        ]
        result = subprocess.run(cmd, cwd=ROOT, check=True, capture_output=True, text=True)
        payload = json.loads(result.stdout.strip().splitlines()[-1])
        written_payload = json.loads(output_log.read_text(encoding="utf-8").strip().splitlines()[-1])

    if payload != written_payload:
        raise AssertionError("stdout payload and JSONL output payload differ")

    assert_close("candidate_count", payload["candidate_count"], EXPECTED["candidate_count"])
    assert_close("anomaly_count", payload["anomaly_count"], EXPECTED["final_flagged_count"])
    if len(payload["anomalies"]) != EXPECTED["final_flagged_count"]:
        raise AssertionError("unexpected number of anomaly records")


def main() -> None:
    test_metrics()
    test_detector_cli()
    print("Smoke test passed.")


if __name__ == "__main__":
    main()
