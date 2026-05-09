# Grain Sentinel — STL Anomaly Detector for Industrial Temperature Data

A lightweight, auditable anomaly detection system for time-series temperature
data. Built with STL decomposition, robust statistics (rolling MAD), and ramp-
gating to suppress false positives. Deployed on a $6/mo VPS as a cron-based
monitoring service.

## What It Does
- Ingests timestamped temperature CSVs from any sensor source.
- Decomposes the signal into trend, seasonal, and residual components (STL).
- Flags anomalies when residuals exceed a rolling MAD threshold (×1.24 × 1.4826)
  for at least two consecutive readings.
- Applies a ramp-gating filter to remove false positives from normal daily
  temperature transitions (e.g., morning ramp, evening drop).
- Outputs structured JSON alerts ready for Telegram, email, or dashboard
  integration.

## Detection Pipeline

Raw temperature → STL decomposition → Rolling MAD threshold → Ramp-gating filter → JSON alerts

## Performance
Evaluated on a weather temperature dataset with 25 injected synthetic anomalies
(slow 0.5°C/hour ramps, mimicking biological heating in stored grain).

| Stage              | Recall | Precision | False Positives |
|--------------------|--------|-----------|-----------------|
| Initial STL + MAD  | 68%    | —         | —               |
| Tuned threshold    | 92%    | 15.5%     | 125             |
| **After ramp-gate**| **88%**| **52.4%** | **20**          |

## Deployment
Runs on a $6/mo DigitalOcean droplet (Ubuntu) via cron every 10 minutes.

- **Input:** `data/input/latest.csv` — any timestamp + temperature CSV.
- **Output:** `data/output/alerts.jsonl` — one JSON line per flagged anomaly.
- **Cron:** `*/10 * * * * cd /root/grain-sentinel && ./venv/bin/python detector_filtered.py --input data/input/latest.csv --output data/output/alerts.jsonl >> logs/cron.log 2>&1`

## Repository Structure
grain-sentinel/
├── scripts/
│ ├── detector_filtered.py # Standalone detector (STL + MAD + ramp-gate)
│ ├── stl_anomaly_detection.py # Original detection script
│ ├── calculate_filtered_metrics.py # Metrics computation
│ └── prepare_audit_validation.py # Audit trail generation
├── notebooks/
│ └── stl_validation.ipynb # Development & tuning notebook
├── data/
│ └── processed/ # Audit files (candidates, final alerts, metrics)
├── plots/
│ ├── stl_anomalies.png # Initial detection plot
│ ├── stl_anomalies_tuned.png # After threshold tuning
│ └── stl_anomalies_filtered.png # Final ramp-gated results
├── DEPLOY.md # VPS deployment guide
├── results.md # Performance summary & tuning log
└── README.md


## Skills Demonstrated
- Time-series analysis (STL decomposition)
- Robust statistics (rolling MAD, hyperparameter tuning)
- Python engineering (standalone scripts, CLI arguments, auditable pipelines)
- Linux deployment (VPS, cron, virtual environments, SSH)
- MLOps fundamentals (input/output contracts, logging, metric tracking)
- Performance evaluation (precision/recall trade-off analysis)

## Why This Project Exists
Built as a learning project to develop ML engineering and cloud deployment
skills. The detection logic is domain-agnostic — it works on grain silo
thermocouples, server room ambient sensors, cold storage monitors, or any
timestamped temperature stream.

## License
MIT — use it, fork it, deploy it wherever you need lightweight anomaly detection.
