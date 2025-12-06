# Git LFS Integration Documentation

## Overview

This document describes the Git Large File Storage (LFS) integration implemented in the AI Investor repository to efficiently handle large SimFin data files.

## What is Git LFS?

Git Large File Storage (LFS) is an open-source Git extension that replaces large files (such as CSV datasets, audio samples, videos, graphics, etc.) with text pointers inside Git, while storing the actual file contents on a remote server.

### Benefits of Git LFS

1. **Efficient Storage**: Large files are stored separately from the Git repository, reducing clone and fetch times
2. **Better Performance**: Only file pointers are tracked in Git history, making operations faster
3. **Bandwidth Savings**: LFS only downloads large files when they are checked out, not during clone/fetch
4. **Size Limit Handling**: Works around GitHub's file size limitations (100MB for regular files)

## Implementation Details

### Files Modified

1. **`.gitattributes`** (New)
   - Configures which files should be tracked by Git LFS
   - Tracks all CSV files in the `stock_data/` directory
   - Also configured to track other large data formats (parquet, hdf5, h5)

2. **`.github/workflows/download-simfin-data.yml`** (Updated)
   - Added Git LFS support throughout the workflow
   - Enhanced checkout, configuration, and push steps

### Configuration Changes

#### 1. Git LFS File Tracking (`.gitattributes`)

```
# Track all CSV files in the stock_data directory with Git LFS
stock_data/*.csv filter=lfs diff=lfs merge=lfs -text
stock_data/**/*.csv filter=lfs diff=lfs merge=lfs -text

# Track other potentially large data files
*.parquet filter=lfs diff=lfs merge=lfs -text
*.hdf5 filter=lfs diff=lfs merge=lfs -text
*.h5 filter=lfs diff=lfs merge=lfs -text
```

#### 2. Workflow Updates

**Step 1: Enhanced Checkout**
```yaml
- name: Checkout repository
  uses: actions/checkout@v4
  with:
    fetch-depth: 1
    lfs: true  # Enable Git LFS during checkout
```

**Step 2: Git LFS Configuration**
```bash
git lfs install
git lfs version
```

**Step 3: LFS Performance Settings**
```bash
git config --global lfs.concurrenttransfers 10
git config --global lfs.transfer.maxretries 3
```

**Step 4: LFS Verification**
- Verifies that files match LFS tracking patterns
- Shows which files will be tracked by LFS

**Step 5: Enhanced Push with LFS**
- Automatically handles LFS files during push
- Shows LFS transfer summary after successful push
- Includes LFS quota exceeded error in failure messages

**Step 6: LFS Status in Summary**
- Reports LFS-tracked files in the workflow summary
- Provides visibility into what files are using LFS

## How Git LFS Works in This Workflow

### During Checkout
1. Git LFS is enabled via `lfs: true` in the checkout action
2. LFS pointers are checked out instead of actual large files
3. Large files are downloaded from LFS storage when needed

### During Data Download
1. SimFin data is downloaded to `stock_data/` directory
2. CSV files are created normally by the Python script
3. Files are ready for Git operations

### During Commit and Push
1. `git add stock_data/` stages the files
2. Git LFS detects files matching patterns in `.gitattributes`
3. Large files are converted to LFS pointers
4. Actual file contents are uploaded to LFS storage
5. Only pointers are committed to the Git repository
6. `git push` uploads both commits and LFS objects

### File Size Limits

- **Standard Git**: 100MB file size limit per file on GitHub
- **Git LFS**: 2GB file size limit per file
- **LFS Storage Quota**: 1GB free per repository, 1GB bandwidth per month
- **Paid Plans**: Additional storage and bandwidth available

## Testing and Verification

### Manual Testing

To manually test the Git LFS integration:

1. **Trigger the workflow manually**:
   - Go to Actions tab in GitHub
   - Select "Download SimFin Data" workflow
   - Click "Run workflow"

2. **Check the workflow logs** for:
   - ✓ Git LFS installed and configured
   - ✓ Git LFS version verified
   - ✓ Git LFS tracking configuration verified
   - LFS transfer summary in push step
   - LFS status in summary

3. **Verify LFS tracking locally**:
   ```bash
   git lfs ls-files
   git lfs ls-files -s  # Show file sizes
   ```

### Automated Checks

The workflow includes automated verification:
- Git LFS version check
- LFS tracking pattern verification
- LFS status reporting in summary

## Migration from Regular Git

### For Existing Large Files

If large files were previously committed to Git (not LFS), they need to be migrated:

```bash
# Install git-lfs
git lfs install

# Migrate existing files to LFS
git lfs migrate import --include="stock_data/*.csv" --everything

# Push the migrated history
git push --force
```

**Note**: Force push rewrites history and should be coordinated with team members.

### For New Repositories

New clones will automatically use LFS for tracked files:

```bash
git clone https://github.com/dcodes101110001/ai_investor.git
cd ai_investor
# LFS files are automatically downloaded during checkout
```

## Troubleshooting

### Common Issues

1. **"This exceeds Git LFS's file size limit of 2GB"**
   - Solution: Split large files or use alternative storage (GitHub Releases, cloud storage)

2. **"LFS quota exceeded"**
   - Check quota: Go to repository Settings → Billing → Git LFS Data
   - Solution: Purchase additional storage or clean up old LFS objects

3. **"Failed to push LFS objects"**
   - Check network connectivity
   - Verify LFS storage quota
   - Review retry attempts in logs

4. **Files not tracked by LFS**
   - Verify `.gitattributes` patterns match file paths
   - Run `git lfs track` to see tracking patterns
   - Use `git lfs ls-files` to see tracked files

### Workflow Failures

If the workflow fails after LFS integration:

1. **Check LFS status step**: Verify tracking patterns are correct
2. **Check push step**: Look for LFS-specific error messages
3. **Check quota**: Ensure LFS storage quota hasn't been exceeded
4. **Review logs**: Check for timeout or network issues

## Performance Improvements

With Git LFS integration:

1. **Faster Clones**: Only pointers are cloned initially
2. **Reduced Repository Size**: Large files stored separately
3. **Efficient Updates**: Only changed LFS files are downloaded
4. **Better Push Reliability**: LFS handles large files more efficiently than standard Git

## Monitoring and Maintenance

### Regular Checks

1. **Monitor LFS storage usage**:
   - Repository Settings → Billing → Git LFS Data
   - Track monthly bandwidth usage

2. **Review workflow logs**:
   - Check LFS transfer sizes
   - Monitor push success rates
   - Watch for quota warnings

3. **Clean up old data**:
   - Remove old LFS objects if needed
   - Prune unreferenced LFS files

### Best Practices

1. **Keep `.gitattributes` up to date**: Add new large file patterns as needed
2. **Document LFS usage**: Inform team members about LFS-tracked files
3. **Monitor quotas**: Set up alerts for LFS storage/bandwidth limits
4. **Test regularly**: Manually trigger workflow to verify LFS functionality

## Additional Resources

- [Git LFS Official Documentation](https://git-lfs.github.com/)
- [GitHub LFS Documentation](https://docs.github.com/en/repositories/working-with-files/managing-large-files)
- [Git LFS Tutorial](https://github.com/git-lfs/git-lfs/wiki/Tutorial)
- [LFS Pricing](https://docs.github.com/en/billing/managing-billing-for-git-large-file-storage/about-billing-for-git-large-file-storage)

## Summary

The Git LFS integration provides:
- ✅ Efficient handling of large CSV data files (up to 2GB per file)
- ✅ Reduced repository clone/fetch times
- ✅ Better push reliability for large datasets
- ✅ Automatic tracking of CSV files in `stock_data/`
- ✅ Comprehensive workflow logging and verification
- ✅ Retry logic with exponential backoff
- ✅ Clear error messages and troubleshooting guidance

The workflow is now configured to handle SimFin data efficiently, even when individual files exceed 400MB, without encountering the previous HTTP 408 timeout errors.
