# Stock Data Directory

This directory is used to store financial data downloaded from SimFin.

The data files (CSV) are automatically downloaded by the GitHub Actions workflow and committed to the `data-update` branch.

## Data Sources

- SimFin API: https://simfin.com/
- Free API key required (see SimFin website for registration)

## Automated Downloads

The workflow `.github/workflows/download-simfin-data.yml` runs daily at 2 AM UTC to download the latest data to the `data-update` branch.

## Large File Handling

Due to GitHub's 100MB file size limit, large CSV files (particularly `us-shareprices-daily.csv`) are automatically split into 50MB chunks with `.part*` extensions.

To reconstruct split files:
```bash
# Reconstruct a split file (e.g., us-shareprices-daily.csv)
cat stock_data/us-shareprices-daily.csv.part* > stock_data/us-shareprices-daily.csv
```

## Local Usage

To download data locally, run:
```bash
python scripts/download_simfin_data.py
```

Make sure to set the `SIMFIN_API_KEY` environment variable with your SimFin API key.
