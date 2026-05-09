#!/usr/bin/env python3
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
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
DEFAULT_SLOPE_WINDOW = 6
DEFAULT_SLOPE_THRESHOLD = -0.007
DEFAULT_RESIDUAL_MIN = 1.2


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


def detect_candidates(
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
        empty_float = pd.Series(dtype=float)
        empty_bool = pd.Series(dtype=bool)
        return {
            "trend": empty_float,
            "seasonal": empty_float,
            "residual": empty_float,
            "threshold": empty_float,
            "candidate_anomalies": empty_bool,
        }

    fit = STL(series, period=period, trend=trend_window, robust=True).fit()
    trend = pd.Series(fit.trend, index=series.index, name="trend")
    seasonal = pd.Series(fit.seasonal, index=series.index, name="seasonal")
    residual = pd.Series(fit.resid, index=series.index, name="residual")
    mad = rolling_mad(residual, rolling_window, min_periods=min(rolling_window, period))
    threshold = threshold_multiplier * mad * 1.4826
    threshold.name = "threshold_value"

    if direction == "positive":
        exceeds = residual > threshold
    elif direction == "negative":
        exceeds = residual < -threshold
    elif direction == "both":
        exceeds = residual.abs() > threshold
    else:
        raise ValueError("direction must be 'positive', 'negative', or 'both'")

    candidate_anomalies = require_consecutive(exceeds.fillna(False), consecutive_points)
    candidate_anomalies.name = "candidate_anomaly"

    return {
        "trend": trend,
        "seasonal": seasonal,
        "residual": residual,
        "threshold": threshold,
        "candidate_anomalies": candidate_anomalies,
    }


def apply_ramp_gate(
    detection: dict[str, pd.Series],
    slope_window: int,
    slope_threshold: float,
    residual_min: float,
) -> dict[str, pd.Series]:
    previous_trend = detection["trend"].shift(1)
    trend_slope = (previous_trend - previous_trend.shift(slope_window)) / slope_window
    trend_slope.name = "trend_slope"
    ramp_gate = trend_slope > slope_threshold
    residual_gate = detection["residual"] >= residual_min
    anomalies = detection["candidate_anomalies"] & ramp_gate.fillna(False) & residual_gate.fillna(False)
    anomalies.name = "anomaly"

    return {
        **detection,
        "trend_slope": trend_slope,
        "ramp_gate": ramp_gate,
        "residual_gate": residual_gate,
        "anomalies": anomalies,
    }


def load_ground_truth(path: Path | None) -> set[pd.Timestamp]:
    if path is None or not path.exists():
        return set()
    truth = pd.read_csv(path)
    if "timestamp" not in truth.columns:
        raise ValueError("Ground truth CSV must contain a timestamp column.")
    return set(pd.to_datetime(truth["timestamp"]))


def write_audit_outputs(
    audit_dir: Path,
    series: pd.Series,
    detection: dict[str, pd.Series],
    ground_truth_path: Path | None,
) -> None:
    audit_dir.mkdir(parents=True, exist_ok=True)
    truth = load_ground_truth(ground_truth_path)

    components = pd.DataFrame(
        {
            "timestamp": series.index,
            "observed": series.values,
            "trend": detection["trend"].reindex(series.index).values,
            "seasonal": detection["seasonal"].reindex(series.index).values,
            "residual": detection["residual"].reindex(series.index).values,
            "threshold_value": detection["threshold"].reindex(series.index).values,
            "trend_slope": detection["trend_slope"].reindex(series.index).values,
            "candidate": detection["candidate_anomalies"].reindex(series.index, fill_value=False).values,
            "kept": detection["anomalies"].reindex(series.index, fill_value=False).values,
        }
    )
    components.to_csv(audit_dir / "stl_components.csv", index=False)

    candidate_mask = detection["candidate_anomalies"].astype(bool)
    candidates = pd.DataFrame(
        {
            "timestamp": detection["residual"].index[candidate_mask],
            "residual": detection["residual"][candidate_mask].values,
            "threshold_value": detection["threshold"][candidate_mask].values,
        }
    )
    candidates.to_csv(audit_dir / "candidates_before_filter.csv", index=False)

    final_rows = pd.DataFrame(
        {
            "timestamp": detection["residual"].index[candidate_mask],
            "residual": detection["residual"][candidate_mask].values,
            "trend_slope": detection["trend_slope"][candidate_mask].values,
            "kept": detection["anomalies"][candidate_mask].values,
            "threshold_value": detection["threshold"][candidate_mask].values,
            "ramp_gate": detection["ramp_gate"][candidate_mask].values,
            "residual_gate": detection["residual_gate"][candidate_mask].values,
        }
    )
    if truth:
        final_rows["is_ground_truth"] = final_rows["timestamp"].isin(truth)
    final_rows.to_csv(audit_dir / "final_anomalies.csv", index=False)


def build_records(series: pd.Series, detection: dict[str, pd.Series], sensor_column: str) -> list[dict[str, object]]:
    records = []
    for timestamp in detection["anomalies"][detection["anomalies"]].index:
        records.append(
            {
                "timestamp": timestamp.isoformat(),
                "sensor": float(series.loc[timestamp]),
                "sensor_column": sensor_column,
                "residual": float(detection["residual"].loc[timestamp]),
                "threshold_value": float(detection["threshold"].loc[timestamp]),
                "trend_slope": float(detection["trend_slope"].loc[timestamp]),
                "kept": True,
            }
        )
    return records


def plot_filtered_detection(
    series: pd.Series,
    detection: dict[str, pd.Series],
    plot_path: Path,
    slope_window: int,
    slope_threshold: float,
    residual_min: float,
) -> None:
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    anomalies = detection["anomalies"].astype(bool)

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True, constrained_layout=True)

    axes[0].plot(series.index, series, color="#2f6f8f", linewidth=1.2, label="Temperature")
    axes[0].scatter(series.index[anomalies], series[anomalies], color="#c7352d", s=24, label="Final anomaly")
    axes[0].set_title("Filtered Temperature Anomalies")
    axes[0].set_ylabel("Temperature (C)")
    axes[0].legend(loc="upper right")
    axes[0].grid(alpha=0.25)

    axes[1].plot(detection["residual"].index, detection["residual"], color="#455a64", linewidth=1.0, label="STL residual")
    axes[1].plot(detection["threshold"].index, detection["threshold"], color="#8d3f2d", linewidth=1.0, label="+ threshold")
    axes[1].axhline(residual_min, color="#6a7f2e", linewidth=1.0, linestyle="--", label="residual minimum")
    axes[1].scatter(
        detection["residual"].index[anomalies],
        detection["residual"][anomalies],
        color="#c7352d",
        s=24,
        label="Final residual",
    )
    axes[1].set_title("Residual Threshold And Absolute Residual Gate")
    axes[1].set_ylabel("Residual (C)")
    axes[1].legend(loc="upper right")
    axes[1].grid(alpha=0.25)

    axes[2].plot(
        detection["trend_slope"].index,
        detection["trend_slope"],
        color="#586f7c",
        linewidth=1.0,
        label=f"Trend slope over previous {slope_window} readings",
    )
    axes[2].axhline(slope_threshold, color="#8d3f2d", linewidth=1.0, linestyle="--", label="slope gate")
    axes[2].scatter(
        detection["trend_slope"].index[anomalies],
        detection["trend_slope"][anomalies],
        color="#c7352d",
        s=24,
        label="Accepted by gate",
    )
    axes[2].set_title("STL Trend Slope Gate")
    axes[2].set_ylabel("C per reading")
    axes[2].set_xlabel("Time")
    axes[2].legend(loc="upper right")
    axes[2].grid(alpha=0.25)

    fig.savefig(plot_path, dpi=150)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run auditable filtered STL + rolling MAD anomaly detection.")
    parser.add_argument("input_csv", type=Path, help="Path to the input CSV file.")
    parser.add_argument("--timestamp-column", default=None, help="Timestamp column. Inferred if omitted.")
    parser.add_argument("--sensor-column", default=DEFAULT_SENSOR_COLUMN, help="Numeric temperature/sensor column.")
    parser.add_argument("--output-log", type=Path, default=None, help="Optional JSONL log path.")
    parser.add_argument("--audit-dir", type=Path, default=None, help="Directory for audit CSV outputs.")
    parser.add_argument("--ground-truth", type=Path, default=None, help="Optional ground truth timestamp CSV.")
    parser.add_argument("--plot-path", type=Path, default=None, help="Optional PNG plot path.")
    parser.add_argument("--resample-rule", default=DEFAULT_RESAMPLE_RULE, help="Pandas resample rule, e.g. 30min.")
    parser.add_argument("--period", type=int, default=DEFAULT_PERIOD, help="STL seasonal period in resampled points.")
    parser.add_argument("--trend-window", type=int, default=None, help="Optional odd STL trend window.")
    parser.add_argument("--rolling-window", type=int, default=DEFAULT_ROLLING_WINDOW, help="Rolling MAD window.")
    parser.add_argument("--threshold-multiplier", type=float, default=DEFAULT_THRESHOLD_MULTIPLIER)
    parser.add_argument("--consecutive-points", type=int, default=DEFAULT_CONSECUTIVE_POINTS)
    parser.add_argument("--direction", choices=["positive", "negative", "both"], default=DEFAULT_DIRECTION)
    parser.add_argument("--slope-window", type=int, default=DEFAULT_SLOPE_WINDOW)
    parser.add_argument("--slope-threshold", type=float, default=DEFAULT_SLOPE_THRESHOLD)
    parser.add_argument("--residual-min", type=float, default=DEFAULT_RESIDUAL_MIN)
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
    candidates = detect_candidates(
        series,
        period=args.period,
        trend_window=trend_window,
        rolling_window=args.rolling_window,
        threshold_multiplier=args.threshold_multiplier,
        consecutive_points=args.consecutive_points,
        direction=args.direction,
    )
    detection = apply_ramp_gate(
        candidates,
        slope_window=args.slope_window,
        slope_threshold=args.slope_threshold,
        residual_min=args.residual_min,
    )

    if args.audit_dir is not None:
        write_audit_outputs(args.audit_dir, series, detection, args.ground_truth)

    if args.plot_path is not None:
        plot_filtered_detection(
            series,
            detection,
            args.plot_path,
            slope_window=args.slope_window,
            slope_threshold=args.slope_threshold,
            residual_min=args.residual_min,
        )

    records = build_records(series, detection, args.sensor_column)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "input_csv": str(args.input_csv),
        "timestamp_column": timestamp_column,
        "sensor_column": args.sensor_column,
        "resampled_points": int(len(series)),
        "candidate_count": int(detection["candidate_anomalies"].sum()),
        "anomaly_count": len(records),
        "params": {
            "resample_rule": args.resample_rule,
            "period": args.period,
            "trend_window": trend_window,
            "rolling_window": args.rolling_window,
            "threshold_multiplier": args.threshold_multiplier,
            "consecutive_points": args.consecutive_points,
            "direction": args.direction,
            "slope_window": args.slope_window,
            "slope_threshold": args.slope_threshold,
            "residual_min": args.residual_min,
        },
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
