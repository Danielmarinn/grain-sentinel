from pathlib import Path
import os
import sqlite3

import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL


PROJECT_DIR = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_DIR / ".matplotlib"))

import matplotlib.pyplot as plt


DATA_PATH = PROJECT_DIR / "data" / "raw" / "weather-humidity-temperature" / "weather.db"
PLOT_PATH = PROJECT_DIR / "plots" / "stl_anomalies.png"
TUNED_PLOT_PATH = PROJECT_DIR / "plots" / "stl_anomalies_tuned.png"
FILTERED_PLOT_PATH = PROJECT_DIR / "plots" / "stl_anomalies_filtered.png"
TUNING_RESULTS_PATH = PROJECT_DIR / "data" / "processed" / "tuning_results.csv"
FILTERED_RESULTS_PATH = PROJECT_DIR / "data" / "processed" / "filtered_metrics.csv"

TABLE_NAME = "sensor_readings"
TIMESTAMP_COL = "timestamp"
TEMP_COL = "temperature"
RESAMPLE_RULE = "30min"
STL_PERIOD = 48
STL_TREND_WINDOW = STL_PERIOD * 7 + 1
ROLLING_WINDOW = STL_PERIOD * 3
MIN_CONSECUTIVE_POINTS = 2
BASELINE_THRESHOLD_MULTIPLIER = 3.0
REQUESTED_THRESHOLD_MULTIPLIERS = [2.5, 2.0, 1.5]
EXTENDED_THRESHOLD_MULTIPLIERS = [1.24, 1.23, 1.22, 1.21, 1.2, 1.1, 1.0, 0.9, 0.8]
TUNING_CONSECUTIVE_OPTIONS = [1, 2]
MIN_TARGET_RECALL = 0.90
DETECTION_DIRECTION = "positive"
BEST_THRESHOLD_MULTIPLIER = 1.24
BEST_CONSECUTIVE_POINTS = 2
FILTER_SLOPE_WINDOW = 6
FILTER_SLOPE_THRESHOLD = -0.007
FILTER_RESIDUAL_MIN = 1.2
ANALYSIS_START = pd.Timestamp("2025-09-11 00:00:00")
ANALYSIS_END = pd.Timestamp("2025-09-28 23:59:59")

INJECT_SYNTHETIC_ANOMALY = True
ANOMALY_START = pd.Timestamp("2025-09-20 06:00:00")
ANOMALY_HOURS = 12
ANOMALY_RISE_C_PER_HOUR = 0.5


def load_temperature_series() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Timedelta]:
    with sqlite3.connect(DATA_PATH) as con:
        df = pd.read_sql_query(f"SELECT * FROM {TABLE_NAME}", con, parse_dates=[TIMESTAMP_COL])

    clean = df.dropna(subset=[TIMESTAMP_COL, TEMP_COL]).copy()
    analysis_df = clean.loc[
        (clean[TIMESTAMP_COL] >= ANALYSIS_START) & (clean[TIMESTAMP_COL] <= ANALYSIS_END)
    ].copy()

    # Average duplicate timestamps before resampling to a regular 30-minute grid.
    grouped = analysis_df.groupby(TIMESTAMP_COL, as_index=True)[TEMP_COL].mean().sort_index()
    median_frequency = grouped.index.to_series().diff().dropna().median()
    temperature = grouped.resample(RESAMPLE_RULE).mean().interpolate(method="time")
    temperature.name = TEMP_COL

    return df, analysis_df, temperature, median_frequency


def inject_slow_rise(
    series: pd.Series,
    start: pd.Timestamp = ANOMALY_START,
    hours: int = ANOMALY_HOURS,
    c_per_hour: float = ANOMALY_RISE_C_PER_HOUR,
) -> tuple[pd.Series, pd.Series, pd.Timestamp]:
    injected = series.copy()
    end = start + pd.Timedelta(hours=hours)
    anomaly_mask = (injected.index >= start) & (injected.index <= end)

    elapsed_hours = (injected.index[anomaly_mask] - start) / pd.Timedelta(hours=1)
    injected.loc[anomaly_mask] = injected.loc[anomaly_mask] + elapsed_hours * c_per_hour

    return injected, pd.Series(anomaly_mask, index=injected.index), end


def rolling_mad(values: pd.Series, window: int, min_periods: int = STL_PERIOD) -> pd.Series:
    return values.rolling(window=window, center=True, min_periods=min_periods).apply(
        lambda x: np.median(np.abs(x - np.median(x))),
        raw=True,
    )


def require_consecutive(mask: pd.Series, min_points: int) -> pd.Series:
    run_id = mask.ne(mask.shift(fill_value=False)).cumsum()
    run_lengths = mask.groupby(run_id).transform("sum")
    return mask & (run_lengths >= min_points)


def decompose_series(
    series: pd.Series,
    period: int = STL_PERIOD,
    trend_window: int = STL_TREND_WINDOW,
) -> dict[str, pd.Series]:
    # A week-scale trend keeps shorter temperature ramps in the residual.
    fit = STL(series, period=period, trend=trend_window, robust=True).fit()
    residual = pd.Series(fit.resid, index=series.index, name="residual")

    return {
        "trend": pd.Series(fit.trend, index=series.index, name="trend"),
        "seasonal": pd.Series(fit.seasonal, index=series.index, name="seasonal"),
        "residual": residual,
    }


def detect_from_residual(
    residual: pd.Series,
    threshold_multiplier: float = BASELINE_THRESHOLD_MULTIPLIER,
    consecutive_points_required: int = MIN_CONSECUTIVE_POINTS,
    rolling_window: int = ROLLING_WINDOW,
    direction: str = DETECTION_DIRECTION,
) -> dict[str, pd.Series]:
    mad = rolling_mad(residual, rolling_window, min_periods=min(rolling_window, STL_PERIOD))
    threshold = threshold_multiplier * mad * 1.4826
    threshold.name = "threshold"

    if direction == "positive":
        exceeds_threshold = residual > threshold
    elif direction == "negative":
        exceeds_threshold = residual < -threshold
    elif direction == "both":
        exceeds_threshold = residual.abs() > threshold
    else:
        raise ValueError("direction must be 'positive', 'negative', or 'both'")

    anomalies = require_consecutive(exceeds_threshold.fillna(False), consecutive_points_required)
    anomalies.name = "anomaly"

    return {
        "residual": residual,
        "threshold": threshold,
        "exceeds_threshold": exceeds_threshold,
        "anomalies": anomalies,
    }


def detect_anomalies(
    series: pd.Series,
    threshold_multiplier: float = BASELINE_THRESHOLD_MULTIPLIER,
    consecutive_points_required: int = MIN_CONSECUTIVE_POINTS,
    rolling_window: int = ROLLING_WINDOW,
    period: int = STL_PERIOD,
    trend_window: int = STL_TREND_WINDOW,
    direction: str = DETECTION_DIRECTION,
) -> dict[str, pd.Series]:
    decomposition = decompose_series(series, period=period, trend_window=trend_window)
    detection = detect_from_residual(
        decomposition["residual"],
        threshold_multiplier=threshold_multiplier,
        consecutive_points_required=consecutive_points_required,
        rolling_window=rolling_window,
        direction=direction,
    )
    return {**decomposition, **detection}


def compute_metrics(anomalies: pd.Series, injected_mask: pd.Series) -> dict[str, float | int]:
    anomalies = anomalies.reindex(injected_mask.index, fill_value=False).astype(bool)
    injected_mask = injected_mask.astype(bool)

    true_positives = int((anomalies & injected_mask).sum())
    false_positives = int((anomalies & ~injected_mask).sum())
    false_negatives = int((~anomalies & injected_mask).sum())

    precision = true_positives / (true_positives + false_positives) if true_positives + false_positives else 0.0
    recall = true_positives / (true_positives + false_negatives) if true_positives + false_negatives else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    return {
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "total_flags": int(anomalies.sum()),
    }


def compute_trend_slope(trend: pd.Series, window: int = FILTER_SLOPE_WINDOW) -> pd.Series:
    slope = (trend - trend.shift(window)) / window
    slope.name = "trend_slope"
    return slope


def apply_ramp_gate(
    detection: dict[str, pd.Series],
    slope_window: int = FILTER_SLOPE_WINDOW,
    slope_threshold: float = FILTER_SLOPE_THRESHOLD,
    residual_min: float = FILTER_RESIDUAL_MIN,
) -> dict[str, pd.Series]:
    trend_slope = compute_trend_slope(detection["trend"], slope_window)
    ramp_gate = trend_slope > slope_threshold
    residual_gate = detection["residual"] >= residual_min
    filtered_anomalies = detection["anomalies"] & ramp_gate.fillna(False) & residual_gate.fillna(False)
    filtered_anomalies.name = "filtered_anomaly"

    return {
        **detection,
        "trend_slope": trend_slope,
        "ramp_gate": ramp_gate,
        "residual_gate": residual_gate,
        "filtered_anomalies": filtered_anomalies,
    }


def run_tuning_grid(
    series: pd.Series,
    injected_mask: pd.Series,
    threshold_multipliers: list[float],
    consecutive_options: list[int],
    rolling_window: int = ROLLING_WINDOW,
    direction: str = DETECTION_DIRECTION,
) -> pd.DataFrame:
    decomposition = decompose_series(series)
    rows = []

    for threshold_multiplier in threshold_multipliers:
        for consecutive_points_required in consecutive_options:
            detection = detect_from_residual(
                decomposition["residual"],
                threshold_multiplier=threshold_multiplier,
                consecutive_points_required=consecutive_points_required,
                rolling_window=rolling_window,
                direction=direction,
            )
            metrics = compute_metrics(detection["anomalies"], injected_mask)
            rows.append(
                {
                    "threshold_multiplier": threshold_multiplier,
                    "consecutive_points_required": consecutive_points_required,
                    "rolling_window": rolling_window,
                    "direction": direction,
                    **metrics,
                }
            )

    return pd.DataFrame(rows)


def select_best_parameters(results: pd.DataFrame, min_recall: float = MIN_TARGET_RECALL) -> pd.Series:
    eligible = results.loc[results["recall"] >= min_recall].copy()
    if eligible.empty:
        return results.sort_values(
            ["recall", "false_positives", "precision", "f1"],
            ascending=[False, True, False, False],
        ).iloc[0]

    return eligible.sort_values(
        ["false_positives", "precision", "f1"],
        ascending=[True, False, False],
    ).iloc[0]


def plot_detection(
    raw_temperature: pd.Series,
    analysis_temperature: pd.Series,
    residual: pd.Series,
    threshold: pd.Series,
    anomalies: pd.Series,
    anomaly_window: tuple[pd.Timestamp, pd.Timestamp] | None,
    output_path: Path = PLOT_PATH,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True, constrained_layout=True)

    axes[0].plot(raw_temperature.index, raw_temperature, color="#2f6f8f", linewidth=1.2, label="Temperature")
    if INJECT_SYNTHETIC_ANOMALY:
        axes[0].plot(
            analysis_temperature.index,
            analysis_temperature,
            color="#d28c25",
            linewidth=1.0,
            alpha=0.85,
            label="Temperature with synthetic rise",
        )
    axes[0].scatter(
        analysis_temperature.index[anomalies],
        analysis_temperature[anomalies],
        color="#c7352d",
        s=22,
        label="Flagged anomaly",
        zorder=3,
    )
    if anomaly_window is not None:
        axes[0].axvspan(anomaly_window[0], anomaly_window[1], color="#f2c078", alpha=0.25)
    axes[0].set_title("Temperature Series")
    axes[0].set_ylabel("Temperature (C)")
    axes[0].legend(loc="upper right")
    axes[0].grid(alpha=0.25)

    axes[1].plot(residual.index, residual, color="#455a64", linewidth=1.0, label="STL residual")
    axes[1].plot(threshold.index, threshold, color="#8d3f2d", linewidth=1.0, label="+ threshold")
    axes[1].plot(threshold.index, -threshold, color="#8d3f2d", linewidth=1.0, label="- threshold")
    axes[1].scatter(
        residual.index[anomalies],
        residual[anomalies],
        color="#c7352d",
        s=22,
        label="Flagged residual",
        zorder=3,
    )
    axes[1].set_title("Residuals After STL Decomposition")
    axes[1].set_ylabel("Residual (C)")
    axes[1].set_xlabel("Time")
    axes[1].legend(loc="upper right")
    axes[1].grid(alpha=0.25)

    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_filtered_detection(
    raw_temperature: pd.Series,
    analysis_temperature: pd.Series,
    detection: dict[str, pd.Series],
    anomaly_window: tuple[pd.Timestamp, pd.Timestamp] | None,
    output_path: Path = FILTERED_PLOT_PATH,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    anomalies = detection["filtered_anomalies"]
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True, constrained_layout=True)

    axes[0].plot(raw_temperature.index, raw_temperature, color="#2f6f8f", linewidth=1.2, label="Temperature")
    axes[0].plot(
        analysis_temperature.index,
        analysis_temperature,
        color="#d28c25",
        linewidth=1.0,
        alpha=0.85,
        label="Temperature with synthetic rise",
    )
    axes[0].scatter(
        analysis_temperature.index[anomalies],
        analysis_temperature[anomalies],
        color="#c7352d",
        s=24,
        label="Filtered anomaly",
        zorder=3,
    )
    if anomaly_window is not None:
        axes[0].axvspan(anomaly_window[0], anomaly_window[1], color="#f2c078", alpha=0.25)
    axes[0].set_title("Filtered Temperature Anomalies")
    axes[0].set_ylabel("Temperature (C)")
    axes[0].legend(loc="upper right")
    axes[0].grid(alpha=0.25)

    axes[1].plot(detection["residual"].index, detection["residual"], color="#455a64", linewidth=1.0, label="STL residual")
    axes[1].plot(detection["threshold"].index, detection["threshold"], color="#8d3f2d", linewidth=1.0, label="+ threshold")
    axes[1].scatter(
        detection["residual"].index[anomalies],
        detection["residual"][anomalies],
        color="#c7352d",
        s=24,
        label="Filtered residual",
        zorder=3,
    )
    axes[1].axhline(FILTER_RESIDUAL_MIN, color="#6a7f2e", linewidth=1.0, linestyle="--", label="residual minimum")
    axes[1].set_title("Residual Threshold And Absolute Residual Gate")
    axes[1].set_ylabel("Residual (C)")
    axes[1].legend(loc="upper right")
    axes[1].grid(alpha=0.25)

    axes[2].plot(
        detection["trend_slope"].index,
        detection["trend_slope"],
        color="#586f7c",
        linewidth=1.0,
        label=f"Trend slope over {FILTER_SLOPE_WINDOW} readings",
    )
    axes[2].axhline(FILTER_SLOPE_THRESHOLD, color="#8d3f2d", linewidth=1.0, linestyle="--", label="slope gate")
    axes[2].scatter(
        detection["trend_slope"].index[anomalies],
        detection["trend_slope"][anomalies],
        color="#c7352d",
        s=24,
        label="Accepted by gate",
        zorder=3,
    )
    axes[2].set_title("STL Trend Slope Gate")
    axes[2].set_ylabel("C per reading")
    axes[2].set_xlabel("Time")
    axes[2].legend(loc="upper right")
    axes[2].grid(alpha=0.25)

    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> None:
    df, analysis_df, raw_temperature, median_frequency = load_temperature_series()

    if INJECT_SYNTHETIC_ANOMALY:
        analysis_temperature, injected_mask, anomaly_end = inject_slow_rise(raw_temperature)
        anomaly_window = (ANOMALY_START, anomaly_end)
    else:
        analysis_temperature = raw_temperature
        injected_mask = pd.Series(False, index=raw_temperature.index)
        anomaly_window = None

    raw_detection = detect_anomalies(raw_temperature)
    detection = detect_anomalies(analysis_temperature)
    anomalies = detection["anomalies"]

    plot_detection(
        raw_temperature=raw_temperature,
        analysis_temperature=analysis_temperature,
        residual=detection["residual"],
        threshold=detection["threshold"],
        anomalies=anomalies,
        anomaly_window=anomaly_window,
    )

    requested_results = run_tuning_grid(
        analysis_temperature,
        injected_mask,
        threshold_multipliers=REQUESTED_THRESHOLD_MULTIPLIERS,
        consecutive_options=TUNING_CONSECUTIVE_OPTIONS,
    )
    extended_results = run_tuning_grid(
        analysis_temperature,
        injected_mask,
        threshold_multipliers=EXTENDED_THRESHOLD_MULTIPLIERS,
        consecutive_options=TUNING_CONSECUTIVE_OPTIONS,
    )
    tuning_results = pd.concat([requested_results, extended_results], ignore_index=True)
    TUNING_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    tuning_results.to_csv(TUNING_RESULTS_PATH, index=False)

    best = select_best_parameters(tuning_results)
    tuned_detection = detect_anomalies(
        analysis_temperature,
        threshold_multiplier=float(best["threshold_multiplier"]),
        consecutive_points_required=int(best["consecutive_points_required"]),
    )
    filtered_detection = apply_ramp_gate(tuned_detection)
    plot_detection(
        raw_temperature=raw_temperature,
        analysis_temperature=analysis_temperature,
        residual=tuned_detection["residual"],
        threshold=tuned_detection["threshold"],
        anomalies=tuned_detection["anomalies"],
        anomaly_window=anomaly_window,
        output_path=TUNED_PLOT_PATH,
    )
    plot_filtered_detection(
        raw_temperature=raw_temperature,
        analysis_temperature=analysis_temperature,
        detection=filtered_detection,
        anomaly_window=anomaly_window,
        output_path=FILTERED_PLOT_PATH,
    )
    filtered_metrics = compute_metrics(filtered_detection["filtered_anomalies"], injected_mask)
    pd.DataFrame([filtered_metrics]).to_csv(FILTERED_RESULTS_PATH, index=False)

    injected_points = int(injected_mask.sum())
    flagged_injected_points = int((anomalies & injected_mask).sum())

    print(f"Dataset: {DATA_PATH}")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print("First 5 rows:")
    print(df.head(5).to_string(index=False))
    print(f"Analysis window: {ANALYSIS_START} to {ANALYSIS_END}")
    print(f"Analysis rows before resampling: {analysis_df.shape[0]}")
    print(f"Timestamp column: {TIMESTAMP_COL}")
    print(f"Temperature column: {TEMP_COL}")
    print(f"Median original sampling interval: {median_frequency}")
    print(f"Resampled frequency: {RESAMPLE_RULE}")
    print(f"STL period: {STL_PERIOD}")
    print(f"STL trend window: {STL_TREND_WINDOW}")
    print(f"Rolling MAD window: {ROLLING_WINDOW}")
    print(f"Raw-series anomaly flags: {int(raw_detection['anomalies'].sum())}")
    print(f"Validation-series anomaly flags: {int(anomalies.sum())}")
    print(f"Synthetic anomaly points: {injected_points}")
    print(f"Flagged points inside synthetic anomaly: {flagged_injected_points}")
    print(f"Plot saved to: {PLOT_PATH}")
    print(f"Tuning results saved to: {TUNING_RESULTS_PATH}")
    print("Tuning results:")
    print(
        tuning_results[
            [
                "threshold_multiplier",
                "consecutive_points_required",
                "true_positives",
                "false_positives",
                "false_negatives",
                "precision",
                "recall",
                "f1",
                "total_flags",
            ]
        ].to_string(index=False)
    )
    print(
        "Best parameters: "
        f"threshold_multiplier={best['threshold_multiplier']}, "
        f"consecutive_points_required={int(best['consecutive_points_required'])}"
    )
    print(f"Tuned plot saved to: {TUNED_PLOT_PATH}")
    print(f"Filtered plot saved to: {FILTERED_PLOT_PATH}")
    print(f"Filtered metrics saved to: {FILTERED_RESULTS_PATH}")
    print(f"Filtered metrics: {filtered_metrics}")

    if anomalies.any():
        flagged_times = anomalies[anomalies].index
        print(f"First flagged point: {flagged_times.min()}")
        print(f"Last flagged point: {flagged_times.max()}")


if __name__ == "__main__":
    main()
