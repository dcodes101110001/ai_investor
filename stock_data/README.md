# Stock Data Directory

This directory is used to store financial data downloaded from SimFin.

The data files (CSV) are automatically downloaded by the GitHub Actions workflow.

## Data Sources

- SimFin API: https://simfin.com/
- Free API key required (see SimFin website for registration)

## Automated Downloads

The workflow `.github/workflows/download-simfin-data.yml` runs daily at 2 AM UTC to download the latest data.

**Workflow Process:**
1. Downloads data to the `data-update` branch with large files split into 50MB chunks (for GitHub compatibility)
2. Pushes the split files to the `main` branch automatically

**Result:**
- `data-update` branch: Contains split files (*.part*) for large CSVs
- `main` branch: Contains the same split files (*.part*) for large CSVs

## Large File Handling

Due to GitHub's 100MB file size limit, large CSV files (particularly `us-shareprices-daily.csv`) are automatically split into 50MB chunks with `.part*` extensions.

**Note:** Split files are pushed to both `data-update` and `main` branches. To use the data, you'll need to manually reconstruct the split files.

### Manual Reconstruction

To reconstruct split files manually:
```bash
# Example: Reconstruct us-shareprices-daily.csv from split parts
# Run from repository root
cat stock_data/us-shareprices-daily.csv.part* > stock_data/us-shareprices-daily.csv

# Or from within the stock_data directory
cd stock_data
cat us-shareprices-daily.csv.part* > us-shareprices-daily.csv
```

## Local Usage

To download data locally, run:
```bash
python scripts/download_simfin_data.py
```

Make sure to set the `SIMFIN_API_KEY` environment variable with your SimFin API key.
