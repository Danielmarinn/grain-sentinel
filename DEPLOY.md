# Grain Sentinel Deployment Notes

This is a lightweight cron-based deployment outline for a Linux VPS.

## 1. Install System Packages

```bash
sudo apt update
sudo apt install -y python3 python3-venv python3-pip
```

## 2. Create Project And Virtual Environment

```bash
mkdir -p ~/projects/grain-sentinel
cd ~/projects/grain-sentinel
python3 -m venv .venv
. .venv/bin/activate
pip install --upgrade pip
pip install pandas numpy statsmodels scipy
```

Copy the project scripts into `~/projects/grain-sentinel/scripts/`.

## 3. Run The Detector Manually

The tuned detector expects a CSV with a timestamp column and a numeric temperature/sensor column.

```bash
cd ~/projects/grain-sentinel
. .venv/bin/activate
python scripts/detector_tuned.py /path/to/input.csv \
  --timestamp-column timestamp \
  --sensor-column temperature \
  --output-log logs/anomalies.jsonl
```

Default tuned parameters:

- Resample rule: `30min`
- STL period: `48`
- Rolling MAD window: `144`
- Threshold multiplier: `1.24`
- Consecutive points: `2`
- Direction: `positive`

Use enough recent history in the input CSV for STL decomposition, ideally several days of data.

## 4. Schedule With Cron

Open cron:

```bash
crontab -e
```

Example job, every 10 minutes:

```cron
*/10 * * * * cd /home/ubuntu/projects/grain-sentinel && . .venv/bin/activate && python scripts/detector_tuned.py /home/ubuntu/projects/grain-sentinel/data/latest_temperature.csv --timestamp-column timestamp --sensor-column temperature --output-log /home/ubuntu/projects/grain-sentinel/logs/anomalies.jsonl >> /home/ubuntu/projects/grain-sentinel/logs/detector_stdout.log 2>> /home/ubuntu/projects/grain-sentinel/logs/detector_stderr.log
```

Telegram alerts are not implemented yet.
