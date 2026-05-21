# Grain Sentinel Deployment Guide

This guide describes the lightweight VPS deployment used by the final ramp-gated detector. The production-facing entrypoint is `scripts/detector_filtered.py`.

## 1. Provision the VPS

Use a small Ubuntu VPS or droplet. Install Python and virtual environment support:

```bash
sudo apt update
sudo apt install -y git python3 python3-venv python3-pip
```

## 2. Clone and install

```bash
mkdir -p /root
cd /root
git clone https://github.com/Danielmarinn/grain-sentinel.git
cd /root/grain-sentinel

python3 -m venv venv
. venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## 3. Prepare runtime folders

```bash
mkdir -p data/input data/output logs
```

The final deployment expects the latest sensor export at:

```text
/root/grain-sentinel/data/input/latest.csv
```

The CSV must contain:

- a timestamp column, for example `timestamp`
- a numeric temperature or sensor column, for example `temperature`

## 4. Run manually

Run the detector once before enabling cron:

```bash
cd /root/grain-sentinel
. venv/bin/activate

python scripts/detector_filtered.py \
  --input data/input/latest.csv \
  --timestamp-column timestamp \
  --sensor-column temperature \
  --output-log data/output/alerts.jsonl
```

For a local validation smoke run, use the included validation input:

```bash
python scripts/detector_filtered.py \
  data/processed/validation_with_injection.csv \
  --timestamp-column timestamp \
  --sensor-column temperature \
  --output-log data/output/alerts.jsonl
```

The command prints one JSON payload to stdout and appends the same payload to `data/output/alerts.jsonl`.

## 5. Schedule with cron

Open the crontab:

```bash
crontab -e
```

Run every 10 minutes:

```cron
*/10 * * * * cd /root/grain-sentinel && ./venv/bin/python scripts/detector_filtered.py --input data/input/latest.csv --timestamp-column timestamp --sensor-column temperature --output-log data/output/alerts.jsonl >> logs/cron.log 2>&1
```

## 6. Check logs

```bash
tail -n 5 /root/grain-sentinel/data/output/alerts.jsonl
tail -n 50 /root/grain-sentinel/logs/cron.log
```

Operational checks to add in a production deployment:

- verify that `latest.csv` is fresh
- monitor cron failures and detector runtime
- rotate `logs/cron.log` and `data/output/alerts.jsonl`
- track row rejection counts and anomaly counts over time
- send JSONL payloads to Telegram, email, or dashboards via a downstream integration

## Tuned vs filtered detector

`scripts/detector_tuned.py` is kept as historical context for the threshold-tuned detector before ramp-gating. New deployments should use `scripts/detector_filtered.py`.
