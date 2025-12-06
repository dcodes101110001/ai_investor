# SimFin Data Download Setup Guide

This guide explains how to set up the automated daily SimFin data download workflow for the AI Investor project.

## Overview

The repository includes a GitHub Actions workflow that automatically downloads financial data from SimFin every day at 2:00 AM UTC. The data is saved to the `stock_data/` directory and can be used by the Python scripts in the educational materials.

## Prerequisites

1. A GitHub account with access to this repository
2. A SimFin API key (free or paid)

## Getting a SimFin API Key

1. Go to [SimFin.com](https://simfin.com/)
2. Create a free account
3. Navigate to your account settings
4. Copy your API key

## Setting up the GitHub Secret

To enable the automated workflow, you must configure the `SIMFIN_API_KEY` as a GitHub repository secret:

### Steps:

1. Navigate to your GitHub repository
2. Click on **Settings** (repository settings, not account settings)
3. In the left sidebar, click on **Secrets and variables** → **Actions**
4. Click the **New repository secret** button
5. Enter the following details:
   - **Name:** `SIMFIN_API_KEY`
   - **Secret:** Paste your SimFin API key
6. Click **Add secret**

## Verifying the Setup

### Manual Test Run

You can manually trigger the workflow to verify it works:

1. Go to the **Actions** tab in your repository
2. Select the **Download SimFin Data** workflow from the left sidebar
3. Click the **Run workflow** button
4. Select the branch you want to update (e.g., `main`)
5. Click **Run workflow**
6. Wait for the workflow to complete (typically 2-5 minutes)
7. Check the logs to ensure data was downloaded successfully
8. Verify that new data files appear in the `stock_data/` directory with a new commit

### Expected Behavior

When the workflow runs successfully:
- ✅ The workflow will download 4 datasets (Income Statements, Balance Sheets, Cash Flow Statements, Share Prices)
- ✅ CSV files will be saved to the `stock_data/` directory
- ✅ Changes will be automatically committed and pushed to the repository
- ✅ Logs will show the number of rows downloaded for each dataset
- ✅ A workflow artifact containing the data will be created
- ✅ The repository will contain the latest financial data
- ✅ A new commit by `github-actions[bot]` will appear in the commit history with message like "chore: update SimFin data - YYYY-MM-DD HH:MM:SS UTC"

### Verifying the Commit

After a successful workflow run:

1. Go to the repository's main page
2. Check the latest commit - it should be from `github-actions[bot]`
3. Click on the commit to see the changes
4. Verify that files in `stock_data/` were added or modified
5. Navigate to the `stock_data/` directory to see the CSV files

### Troubleshooting

If the workflow fails:

1. **Check the API key:**
   - Ensure `SIMFIN_API_KEY` is set correctly in GitHub Secrets
   - Verify your API key is valid on SimFin.com
   
2. **Check the logs:**
   - Go to the Actions tab
   - Click on the failed workflow run
   - Review the logs for specific error messages
   
3. **Common issues:**
   - Invalid API key: The secret may be incorrectly copied or expired
   - Network issues: Temporary connectivity problems (workflow will retry next day)
   - API rate limits: Free accounts have download limits
   - SimFin service down: Check SimFin's status page
   - Push failures: May occur if the branch was updated during workflow execution
   
4. **Auto-commit specific issues:**
   - **Permission denied on push:** Ensure the workflow has `contents: write` permission
   - **No changes committed:** This is normal if the data hasn't changed since the last download
   - **Merge conflicts:** Should not occur as the workflow only modifies files in `stock_data/`
   - **Bot commits not appearing:** Check that the workflow completed successfully and didn't fail at the push step

## Accessing Downloaded Data

The downloaded data files are:
- **Automatically committed to the repository** after each successful download
- Updated daily at 2:00 AM UTC
- Available directly in the `stock_data/` directory on the main branch
- Also available as workflow artifacts for 30 days

To access the data:
1. **From the repository:** The latest data is always available in the `stock_data/` directory
2. **From workflow artifacts:** 
   - Go to the **Actions** tab
   - Click on a successful workflow run
   - Scroll to the **Artifacts** section at the bottom
   - Download the `simfin-data-*` artifact

## Local Usage

To download data locally for development:

1. Install the required package:
   ```bash
   pip install simfin
   ```

2. Set your API key as an environment variable:
   ```bash
   export SIMFIN_API_KEY='your-api-key-here'
   ```

3. Run the download script:
   ```bash
   python scripts/download_simfin_data.py
   ```

The data will be saved to the `stock_data/` directory.

## Data Files

The workflow downloads the following datasets:
- `us-income-annual.csv` - Annual income statements
- `us-balance-annual.csv` - Annual balance sheets
- `us-cashflow-annual.csv` - Annual cash flow statements
- `us-shareprices-daily.csv` - Daily share prices

## Schedule

The workflow runs automatically:
- **Time:** 2:00 AM UTC daily
- **Timezone:** UTC (Coordinated Universal Time)
- **Can be manually triggered:** Yes, via the Actions tab

## Support

For issues related to:
- **SimFin API:** Visit [SimFin Support](https://simfin.com/support)
- **This workflow:** Open an issue in this repository
- **Python package:** Check [SimFin Python API docs](https://simfin.readthedocs.io/)

## Security Notes

- ✅ API key is stored securely in GitHub Secrets
- ✅ API key is never logged or exposed in workflow runs
- ✅ Workflow has write permissions to commit downloaded data
- ✅ Commits are made by `github-actions[bot]` user
- ✅ Downloaded data files are committed to version control for easy access
