# Stock Data Directory

This directory is used to store financial data downloaded from SimFin.

The data files (CSV) are automatically downloaded by the GitHub Actions workflow and are excluded from version control via `.gitignore`.

## Data Sources

- SimFin API: https://simfin.com/
- Free API key required (see SimFin website for registration)

## Automated Downloads

The workflow `.github/workflows/download-simfin-data.yml` runs daily at 2 AM UTC to download the latest data.

## Local Usage

To download data locally, run:
```bash
python scripts/download_simfin_data.py
```

Make sure to set the `SIMFIN_API_KEY` environment variable with your SimFin API key.
