# SimFin Data Download Automation

This workflow automates the daily download of SimFin financial data required for the AI Investor project.

## Overview

The workflow runs daily at 2 AM UTC and downloads the following data from SimFin:
- US Income Statements (Annual)
- US Balance Sheets (Annual)
- US Cash Flow Statements (Annual)
- US Share Prices (Daily)

All data is saved to the `stock_data/` directory in the repository.

## Setup

### 1. Configure SimFin API Key as GitHub Secret

To use this workflow, you need to configure your SimFin API key as a **GitHub Secret**:

1. Get your API key from [SimFin](https://www.simfin.com/) (free tier is sufficient)
2. Go to your repository settings on GitHub
3. Navigate to `Settings` > `Secrets and variables` > `Actions`
4. Click `New repository secret`
5. Name: `SIMFIN_API_KEY`
6. Value: Your SimFin API key
7. Click `Add secret`

**Why GitHub Secrets?** GitHub Secrets securely store sensitive information like API keys. The workflow accesses the secret using `${{ secrets.SIMFIN_API_KEY }}`, which prevents the key from being exposed in logs or code.

### 2. Enable GitHub Actions

Ensure GitHub Actions is enabled for your repository:
1. Go to `Settings` > `Actions` > `General`
2. Under "Actions permissions", select "Allow all actions and reusable workflows"

## Manual Trigger

You can manually trigger the workflow from the GitHub Actions tab:
1. Go to the `Actions` tab in your repository
2. Select `Daily SimFin Data Download` workflow
3. Click `Run workflow`
4. Select the branch and click `Run workflow`

## Script Details

### Download Script: `scripts/download_simfin_data.py`

The Python script handles:
- Creating the `stock_data/` directory if it doesn't exist
- Authenticating with the SimFin API using the provided API key
- Downloading all required datasets
- Error handling and logging for download failures
- Verification of successful downloads

### Workflow: `.github/workflows/download-simfin-data.yml`

The GitHub Actions workflow:
- Runs daily via cron schedule: `0 2 * * *` (2 AM UTC)
- Can be triggered manually for testing
- Sets up Python environment
- Installs required dependencies (simfin, pandas, numpy)
- Runs the download script with API key from secrets
- Verifies downloaded files
- Uploads logs as artifacts for troubleshooting
- Reports success or failure status

## Error Handling

The workflow includes comprehensive error handling:

1. **API Key Missing**: Logs error if `SIMFIN_API_KEY` is not configured
2. **Download Failures**: Individual dataset download failures are logged
3. **Verification**: Checks that files were actually downloaded
4. **Logs**: All logs are uploaded as artifacts for debugging

## Troubleshooting

### Check Workflow Logs

1. Go to the `Actions` tab
2. Click on the latest workflow run
3. Expand the failed step to view detailed logs

### Download Logs

Workflow logs are available as artifacts:
1. Go to the workflow run page
2. Scroll to the bottom to find `Artifacts`
3. Download `download-logs-*` for detailed information

### Common Issues

- **API Key Invalid**: Verify your SimFin API key is correct in GitHub secrets
- **Network Errors**: SimFin API might be temporarily unavailable, workflow will retry next day
- **Storage Limits**: If data files are too large, consider storing them elsewhere or downloading only needed datasets

## Local Testing

To test the script locally:

```bash
# Set your API key
export SIMFIN_API_KEY="your-api-key-here"

# Run the script
python scripts/download_simfin_data.py
```

## Data Usage

The downloaded data is used by the AI Investor analysis scripts in:
- `Chapter_5_to_7_AI_and_Backtesting/1_Get_X_Y_Learning_Data_Raw.py`
- Other analysis and backtesting scripts

The data files are excluded from git commits via `.gitignore` to prevent repository bloat.
