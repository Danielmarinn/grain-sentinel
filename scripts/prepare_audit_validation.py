#!/usr/bin/env python3
from pathlib import Path
import sys

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from scripts.stl_anomaly_detection import (
    ANOMALY_RISE_C_PER_HOUR,
    ANOMALY_START,
    DATA_PATH,
    RESAMPLE_RULE,
    TABLE_NAME,
    TEMP_COL,
    TIMESTAMP_COL,
    inject_slow_rise,
    load_temperature_series,
)


OUTPUT_SERIES = PROJECT_DIR / "data" / "processed" / "validation_with_injection.csv"
GROUND_TRUTH = PROJECT_DIR / "data" / "processed" / "ground_truth_timestamps.csv"


def main() -> None:
    _, _, raw_temperature, _ = load_temperature_series()
    analysis_temperature, injected_mask, anomaly_end = inject_slow_rise(raw_temperature)

    OUTPUT_SERIES.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "timestamp": analysis_temperature.index,
            "temperature": analysis_temperature.values,
            "original_temperature": raw_temperature.values,
            "is_injected": injected_mask.values,
        }
    ).to_csv(OUTPUT_SERIES, index=False)

    pd.DataFrame({"timestamp": analysis_temperature.index[injected_mask]}).to_csv(GROUND_TRUTH, index=False)

    print(f"Source database: {DATA_PATH}")
    print(f"Source table: {TABLE_NAME}")
    print(f"Timestamp column: {TIMESTAMP_COL}")
    print(f"Temperature column: {TEMP_COL}")
    print(f"Resample rule: {RESAMPLE_RULE}")
    print(f"Injection start: {ANOMALY_START}")
    print(f"Injection end: {anomaly_end}")
    print(f"Injection rise: {ANOMALY_RISE_C_PER_HOUR} C/hour")
    print(f"Validation CSV: {OUTPUT_SERIES}")
    print(f"Ground truth CSV: {GROUND_TRUTH}")
    print(f"Ground truth count: {int(injected_mask.sum())}")


if __name__ == "__main__":
    main()
