#!/usr/bin/env python3
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL


DEFAULT_SENSOR_COLUMN = "temperature"
DEFAULT_RESAMPLE_RULE = "30min"
DEFAULT_PERIOD = 48
DEFAULT_ROLLING_WINDOW = 144
DEFAULT_THRESHOLD_MULTIPLIER = 1.24
DEFAULT_CONSECUTIVE_POINTS = 2
DEFAULT_DIRECTION = "positive"


def infer_timestamp_column(df: pd.DataFrame) -> str:
    candidates = ["timestamp", "datetime", "date_time", "date", "time"]
    lower_to_original = {column.lower(): column for column in df.columns}

    for candidate in candidates:
        if candidate in lower_to_original:
            return lower_to_original[candidate]

    for column in df.columns:
        parsed = pd.to_datetime(df[column], errors="coerce")
        if parsed.notna().mean() >= 0.8:
            return column

    raise ValueError("Could not infer a timestamp column. Pass --timestamp-column.")


def make_trend_window(period: int, trend_window: int | None) -> int:
    if trend_window is None:
        trend_window = period * 7 + 1
    if trend_window <= period:
        trend_window = period + 1
    if trend_window % 2 == 0:
        trend_window += 1
    return trend_window


def load_sensor_series(
    csv_path: Path,
    sensor_column: str,
    timestamp_column: str | None,
    resample_rule: str,
) -> tuple[pd.Series, str]:
    df = pd.read_csv(csv_path)
    timestamp_column = timestamp_column or infer_timestamp_column(df)

    if timestamp_column not in df.columns:
        raise ValueError(f"Timestamp column not found: {timestamp_column}")
    if sensor_column not in df.columns:
        raise ValueError(f"Sensor column not found: {sensor_column}")

    working = df[[timestamp_column, sensor_column]].copy()
    working[timestamp_column] = pd.to_datetime(working[timestamp_column], errors="coerce")
    working[sensor_column] = pd.to_numeric(working[sensor_column], errors="coerce")
    working = working.dropna(subset=[timestamp_column, sensor_column])

    if working.empty:
        raise ValueError("No valid timestamp/sensor rows found after parsing.")

    grouped = working.groupby(timestamp_column, as_index=True)[sensor_column].mean().sort_index()
    series = grouped.resample(resample_rule).mean().interpolate(method="time")
    series.name = sensor_column

    return series, timestamp_column


def rolling_mad(values: pd.Series, window: int, min_periods: int) -> pd.Series:
    return values.rolling(window=window, center=True, min_periods=min_periods).apply(
        lambda x: np.median(np.abs(x - np.median(x))),
        raw=True,
    )


def require_consecutive(mask: pd.Series, min_points: int) -> pd.Series:
    run_id = mask.ne(mask.shift(fill_value=False)).cumsum()
    run_lengths = mask.groupby(run_id).transform("sum")
    return mask & (run_lengths >= min_points)


def detect_anomalies(
    series: pd.Series,
    period: int,
    trend_window: int,
    rolling_window: int,
    threshold_multiplier: float,
    consecutive_points: int,
    direction: str,
) -> dict[str, pd.Series]:
    min_required_points = max(period * 2, rolling_window)
    if len(series) < min_required_points:
        return {
            "residual": pd.Series(dtype=float),
            "threshold": pd.Series(dtype=float),
            "anomalies": pd.Series(dtype=bool),
        }

    fit = STL(series, period=period, trend=trend_window, robust=True).fit()
    residual = pd.Series(fit.resid, index=series.index, name="residual")
    mad = rolling_mad(residual, rolling_window, min_periods=min(rolling_window, period))
    threshold = threshold_multiplier * mad * 1.4826
    threshold.name = "threshold"

    if direction == "positive":
        exceeds = residual > threshold
    elif direction == "negative":
        exceeds = residual < -threshold
    elif direction == "both":
        exceeds = residual.abs() > threshold
    else:
        raise ValueError("direction must be 'positive', 'negative', or 'both'")

    anomalies = require_consecutive(exceeds.fillna(False), consecutive_points)
    anomalies.name = "anomaly"

    return {
        "residual": residual,
        "threshold": threshold,
        "anomalies": anomalies,
    }


def build_records(series: pd.Series, detection: dict[str, pd.Series], sensor_column: str) -> list[dict[str, object]]:
    anomalies = detection["anomalies"]
    residual = detection["residual"]
    threshold = detection["threshold"]

    records = []
    for timestamp in anomalies[anomalies].index:
        records.append(
            {
                "timestamp": timestamp.isoformat(),
                "sensor": float(series.loc[timestamp]),
                "sensor_column": sensor_column,
                "residual": float(residual.loc[timestamp]),
                "threshold": float(threshold.loc[timestamp]),
            }
        )
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run tuned STL + rolling MAD anomaly detection on a CSV file.")
    parser.add_argument("input_csv", type=Path, help="Path to the input CSV file.")
    parser.add_argument("--timestamp-column", default=None, help="Timestamp column. Inferred if omitted.")
    parser.add_argument("--sensor-column", default=DEFAULT_SENSOR_COLUMN, help="Numeric temperature/sensor column.")
    parser.add_argument("--output-log", type=Path, default=None, help="Optional JSONL log path.")
    parser.add_argument("--resample-rule", default=DEFAULT_RESAMPLE_RULE, help="Pandas resample rule, e.g. 30min.")
    parser.add_argument("--period", type=int, default=DEFAULT_PERIOD, help="STL seasonal period in resampled points.")
    parser.add_argument("--trend-window", type=int, default=None, help="Optional odd STL trend window.")
    parser.add_argument("--rolling-window", type=int, default=DEFAULT_ROLLING_WINDOW, help="Rolling MAD window.")
    parser.add_argument("--threshold-multiplier", type=float, default=DEFAULT_THRESHOLD_MULTIPLIER)
    parser.add_argument("--consecutive-points", type=int, default=DEFAULT_CONSECUTIVE_POINTS)
    parser.add_argument("--direction", choices=["positive", "negative", "both"], default=DEFAULT_DIRECTION)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    trend_window = make_trend_window(args.period, args.trend_window)
    series, timestamp_column = load_sensor_series(
        args.input_csv,
        sensor_column=args.sensor_column,
        timestamp_column=args.timestamp_column,
        resample_rule=args.resample_rule,
    )
    detection = detect_anomalies(
        series,
        period=args.period,
        trend_window=trend_window,
        rolling_window=args.rolling_window,
        threshold_multiplier=args.threshold_multiplier,
        consecutive_points=args.consecutive_points,
        direction=args.direction,
    )
    records = build_records(series, detection, args.sensor_column)

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "input_csv": str(args.input_csv),
        "timestamp_column": timestamp_column,
        "sensor_column": args.sensor_column,
        "resampled_points": int(len(series)),
        "params": {
            "resample_rule": args.resample_rule,
            "period": args.period,
            "trend_window": trend_window,
            "rolling_window": args.rolling_window,
            "threshold_multiplier": args.threshold_multiplier,
            "consecutive_points": args.consecutive_points,
            "direction": args.direction,
        },
        "anomaly_count": len(records),
        "anomalies": records,
    }

    line = json.dumps(payload, separators=(",", ":"))
    print(line)

    if args.output_log is not None:
        args.output_log.parent.mkdir(parents=True, exist_ok=True)
        with args.output_log.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")


if __name__ == "__main__":
    main()
