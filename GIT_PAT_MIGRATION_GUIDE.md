# GIT_PAT Migration Guide

## Overview

This document explains the migration from `GITHUB_PAT` to `GIT_PAT` for GitHub Actions workflow authentication.

## What Changed?

All references to `GITHUB_PAT` have been replaced with `GIT_PAT` throughout the repository. This includes:
- GitHub Actions workflow file
- All setup and configuration documentation
- Error messages and troubleshooting guides

## Why This Change?

This change provides better distinction between:
- **GITHUB_TOKEN**: The default token automatically provided by GitHub Actions (limited permissions on forks)
- **GIT_PAT**: Your custom Personal Access Token with elevated permissions

## Action Required

If you previously set up `GITHUB_PAT`, you need to:

### Option 1: Rename Your Existing Secret (Recommended)

1. Go to: **Repository Settings → Secrets and variables → Actions**
2. Delete the old `GITHUB_PAT` secret (or keep it if used elsewhere)
3. Create a new secret:
   - **Name**: `GIT_PAT`
   - **Value**: (use the same token value)

### Option 2: Create a New Token

If your old token expired or you want a fresh start:

1. Create a new Personal Access Token:
   - Go to: https://github.com/settings/tokens
   - Click: "Generate new token (classic)"
   - Scopes: ✅ `repo`, ✅ `workflow`
   - Set expiration: 90 days (recommended)
   - Copy the token

2. Add as `GIT_PAT` secret:
   - Go to: **Repository Settings → Secrets and variables → Actions**
   - Click: "New repository secret"
   - Name: `GIT_PAT`
   - Value: (paste your token)
   - Click: "Add secret"

## Testing the Migration

### Step 1: Verify Secret Exists

1. Go to: **Repository Settings → Secrets and variables → Actions**
2. Confirm `GIT_PAT` appears in the secrets list

### Step 2: Run the Workflow

1. Go to: **Actions** tab
2. Select: "Download SimFin Data" workflow
3. Click: "Run workflow"
4. Select branch: The current branch
5. Click: "Run workflow" button

### Step 3: Check for Success

Monitor the workflow logs for these success messages:

```
✓ GIT_PAT secret found
✓ Personal Access Token configured successfully
✓ Authentication ready for Git LFS push operations
✓ Changes pushed successfully to branch: data-updates
```

## Troubleshooting

### Issue: Workflow fails with "GIT_PAT secret not found"

**Solution**: You haven't created the `GIT_PAT` secret yet. Follow "Action Required" steps above.

### Issue: Workflow fails with authentication error

**Possible causes**:
- Token lacks required scopes (needs `repo` + `workflow`)
- Token has expired
- Token was copied incorrectly (check for extra spaces)

**Solution**: Create a new token with correct scopes and update the secret.

### Issue: Still seeing references to GITHUB_PAT

**Solution**: Make sure you're looking at the latest version of the files. Pull the latest changes:
```bash
git pull origin main
```

## Files Modified

The following files were updated in this migration:

1. `.github/workflows/download-simfin-data.yml` - Main workflow file
2. `QUICK_SETUP_PAT.md` - Quick setup guide
3. `ACTION_REQUIRED.md` - Action required documentation
4. `WORKFLOW_AUTHENTICATION_SETUP.md` - Detailed authentication guide
5. `PAT_FIX_SUMMARY.md` - Implementation summary
6. `IMPLEMENTATION_SUMMARY.md` - Technical details
7. `WORKFLOW_FIX_DOCUMENTATION.md` - Troubleshooting guide

## Verification Checklist

Use this checklist to ensure the migration is complete:

- [ ] Created/renamed secret to `GIT_PAT` in repository settings
- [ ] Token has `repo` and `workflow` scopes
- [ ] Ran workflow manually to test
- [ ] Workflow completed successfully
- [ ] Verified data-updates branch was created/updated
- [ ] Checked workflow logs for success messages

## Need Help?

If you encounter issues:

1. Check the workflow logs for detailed error messages
2. Review [QUICK_SETUP_PAT.md](QUICK_SETUP_PAT.md) for setup instructions
3. See [WORKFLOW_AUTHENTICATION_SETUP.md](WORKFLOW_AUTHENTICATION_SETUP.md) for comprehensive guide
4. Verify your token hasn't expired
5. Ensure the token has correct scopes

## Alternative: SSH Authentication

If you prefer SSH authentication over PAT:

1. See [WORKFLOW_AUTHENTICATION_SETUP.md](WORKFLOW_AUTHENTICATION_SETUP.md) for SSH setup
2. This migration doesn't affect SSH authentication
3. SSH authentication continues to work as before

---

**Migration Date**: December 2024
**Status**: Complete ✅
