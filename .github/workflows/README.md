# SimFin Data Download Automation

This directory contains the automated workflow for downloading stock data from SimFin.

## Overview

The GitHub Actions workflow (`download-simfin-data.yml`) automatically downloads financial data from [SimFin](https://simfin.com/) on a daily schedule and saves it to the `stock_data` directory.

## Required Files

The workflow downloads the following CSV files from SimFin:
- `us-income-annual-full-asreported.csv` - Annual income statements
- `us-balance-annual-full-asreported.csv` - Annual balance sheets
- `us-cashflow-annual-full-asreported.csv` - Annual cash flow statements
- `us-shareprices-daily.csv` - Daily share prices

## Setup Instructions

### 1. Get a SimFin API Key

1. Sign up for a free account at [SimFin](https://simfin.com/)
2. Navigate to your account settings
3. Generate an API key (free tier is sufficient for basic data)

### 2. Add API Key to GitHub Secrets

1. Go to your repository on GitHub
2. Click on **Settings** > **Secrets and variables** > **Actions**
3. Click **New repository secret**
4. Name: `SIMFIN_API_KEY`
5. Value: Paste your SimFin API key
6. Click **Add secret**

### 3. Workflow Schedule

The workflow runs automatically:
- **Daily at 2:00 AM UTC** (via cron schedule)
- Can also be triggered manually from the Actions tab

### 4. Manual Trigger

To manually trigger the workflow:
1. Go to the **Actions** tab in your repository
2. Select **Download SimFin Data** workflow
3. Click **Run workflow**
4. Select the branch and click **Run workflow**

## How It Works

1. **Checkout**: Checks out the repository code
2. **Setup Python**: Installs Python 3.12 and dependencies
3. **Create Directory**: Ensures `stock_data` directory exists
4. **Download Data**: Runs the Python script to download data from SimFin
5. **Verify**: Checks that all expected files were downloaded
6. **Upload Artifacts**: Saves downloaded files as workflow artifacts (kept for 7 days)
7. **Logging**: Outputs success or failure messages

## Error Handling

The workflow includes comprehensive error handling:
- Connection timeouts (5 minutes per file)
- HTTP errors (401 authentication, 404 not found, etc.)
- Missing files verification
- Detailed logging for debugging

## Files

- **`download-simfin-data.yml`**: GitHub Actions workflow definition
- **`../scripts/download_simfin_data.py`**: Python script that performs the download

## Troubleshooting

### Downloads failing?

Check the following:
1. **API Key**: Ensure `SIMFIN_API_KEY` secret is set correctly
2. **SimFin Status**: Check if SimFin service is operational
3. **Account Limits**: Verify your account hasn't exceeded download limits
4. **Workflow Logs**: Review the Actions tab for detailed error messages

### Files not appearing?

The `stock_data` directory and its contents are in `.gitignore` to prevent committing large CSV files. Files are:
- Created during workflow execution
- Available as workflow artifacts for 7 days
- Should be regenerated daily by the workflow

## Local Testing

To test the download script locally:

```bash
# Set your API key (optional for free data)
export SIMFIN_API_KEY="your-api-key-here"

# Run the script
python scripts/download_simfin_data.py
```

## Notes

- Downloaded files are **not committed** to the repository (excluded via `.gitignore`)
- Files are available as GitHub Actions artifacts
- The workflow will fail if any file download fails
- Large files may take several minutes to download
