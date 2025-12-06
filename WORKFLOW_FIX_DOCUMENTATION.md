# GitHub Actions Workflow Fix - HTTP 408 Timeout Error

> **✅ LATEST UPDATE (December 2024):**
> The workflow now includes comprehensive inline documentation for authentication setup.
> Check the workflow file (`.github/workflows/download-simfin-data.yml`) for detailed 
> step-by-step instructions on configuring GITHUB_PAT or SSH authentication.

> **⚠️ IMPORTANT UPDATE (December 2024):**
> If you're running this workflow on a **public fork**, you may encounter the error:
> `@github-actions[bot] can not upload new objects to public fork`
> 
> This is a GitHub limitation on fork permissions. Please see the new
> **[WORKFLOW_AUTHENTICATION_SETUP.md](WORKFLOW_AUTHENTICATION_SETUP.md)** guide for solutions using:
> - SSH Deploy Keys (recommended), or
> - Personal Access Tokens
> 
> The workflow now supports pushing to a dedicated branch (`data-updates`) with proper authentication.

---

## Problem Description

The GitHub Actions workflow for downloading SimFin data was experiencing HTTP 408 timeout errors during the `git push` operation. This occurred when attempting to push approximately 423MB of financial data (including a 414MB daily share prices CSV file) to the repository.

### Error Details
```
error: RPC failed; HTTP 408 curl 22 The requested URL returned error: 408
send-pack: unexpected disconnect while reading sideband packet
fatal: the remote end hung up unexpectedly
```

## Root Cause

The error was caused by:
1. **Large payload size**: ~423MB of data being pushed in a single operation
2. **Network timeout**: Default Git HTTP timeout settings insufficient for large transfers
3. **No retry mechanism**: Single push attempt with no fallback
4. **Full repository clone**: Unnecessary history being fetched during checkout

## Implemented Solutions

### 1. Shallow Clone Optimization

**Change:**
```yaml
- name: Checkout repository
  uses: actions/checkout@v4
  with:
    fetch-depth: 1  # Shallow clone to reduce checkout time and data transfer
```

**Benefit:**
- Reduces initial clone size by fetching only the latest commit
- Faster checkout time
- Reduces network bandwidth usage

### 2. Increased Git Buffer Size

**Change:**
```bash
git config --global http.postBuffer 157286400
```

**Benefit:**
- Sets POST buffer to 150MB (157286400 bytes)
- Allows Git to handle larger data transfers without fragmenting
- Prevents premature timeout on large payloads

### 3. HTTP Timeout Configuration

**Changes:**
```bash
git config --global http.lowSpeedLimit 1000
git config --global http.lowSpeedTime 60
```

**Benefit:**
- `http.lowSpeedLimit`: Minimum transfer rate of 1000 bytes/second
- `http.lowSpeedTime`: Allows 60 seconds at low speed before timeout
- More lenient timeout settings for large file transfers
- Handles temporary network slowdowns gracefully

### 4. Retry Logic with Exponential Backoff

**Change:**
```bash
max_attempts=3
attempt=1

while [ $attempt -le $max_attempts ]; do
  echo "Push attempt $attempt of $max_attempts..."
  
  if git push; then
    # Success handling
    exit 0
  else
    # Retry with increasing wait time
    wait_time=$((attempt * 10))
    sleep $wait_time
    attempt=$((attempt + 1))
  fi
done
```

**Benefit:**
- Up to 3 push attempts
- Exponential backoff: waits 10s, then 20s between retries
- Handles transient network issues
- Improved success rate for large transfers

## Testing Recommendations

1. **Manual Workflow Trigger**: Test the updated workflow using the "workflow_dispatch" trigger
2. **Monitor Logs**: Check workflow logs for:
   - Git configuration output
   - Push attempt counts
   - Retry wait times
   - Final success/failure status
3. **Verify Data Integrity**: Ensure pushed data matches downloaded data
4. **Performance Metrics**: Track push duration and success rate over multiple runs

## Additional Considerations

### Future Optimizations

If HTTP 408 errors persist, consider:

1. **Git LFS (Large File Storage)**
   - Store large CSV files in Git LFS
   - Reduces repository size
   - Better handling of large binary files

2. **Data Compression**
   - Compress CSV files before committing
   - Reduces transfer size
   - Requires decompression step for usage

3. **Incremental Updates**
   - Only commit changed portions of data
   - Implement delta/diff-based updates
   - Significantly reduces push size

4. **Alternative Storage**
   - Store data in GitHub Releases
   - Use external storage (S3, Azure Blob Storage)
   - Keep repository lean

### Network Stability

The workflow now handles:
- ✅ Transient network failures (via retry logic)
- ✅ Slow network connections (via timeout configuration)
- ✅ Large payload transfers (via increased buffer size)
- ✅ Repository size optimization (via shallow clone)

### Monitoring

Monitor workflow runs for:
- Success rate of push operations
- Number of retries needed
- Transfer times
- Any recurring failure patterns

## Files Modified

- `.github/workflows/download-simfin-data.yml`: Updated workflow configuration

## Summary

These changes address the HTTP 408 timeout error through a multi-layered approach:
1. Optimize data transfer with shallow clones
2. Configure Git for large file handling
3. Implement robust retry mechanism
4. Improve timeout tolerance

The workflow should now successfully handle the ~423MB data push operations with significantly improved reliability.
