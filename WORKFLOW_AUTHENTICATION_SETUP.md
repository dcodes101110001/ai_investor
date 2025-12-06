# GitHub Actions Workflow Authentication Setup

## 🚀 Quick Start

**For most users**: The workflow file (`.github/workflows/download-simfin-data.yml`) now contains comprehensive inline documentation with step-by-step setup instructions. Simply open the workflow file and follow the comments in the "Setup Personal Access Token" step.

This guide provides additional context and troubleshooting information.

---

## Overview

This document provides detailed instructions for setting up authentication for the SimFin data download workflow to resolve the `@github-actions[bot] can not upload new objects to public fork` error.

## Problem Background

When running GitHub Actions workflows on **public forks**, the default `GITHUB_TOKEN` has restricted permissions and cannot push Git LFS (Large File Storage) objects to the repository. This limitation is a security feature to prevent unauthorized code changes through forked repositories.

### Error Message
```
batch response: @github-actions[bot] can not upload new objects to public fork
```

## Solutions

This workflow supports **two authentication methods**. Choose the one that best fits your security requirements:

### 🔐 Method 1: SSH Deploy Key (Recommended)

**Advantages:**
- More secure - key is repository-specific
- Doesn't grant access to other repositories
- Can be easily revoked without affecting other services
- Best practice for production workflows

**Setup Steps:**

1. **Generate an SSH key pair** (on your local machine):
   ```bash
   ssh-keygen -t ed25519 -C "github-actions-deploy" -f ~/.ssh/github_actions_deploy
   ```
   - When prompted, **do not set a passphrase** (press Enter)
   - This creates two files:
     - `~/.ssh/github_actions_deploy` (private key)
     - `~/.ssh/github_actions_deploy.pub` (public key)

2. **Add the public key as a Deploy Key**:
   - Go to your repository on GitHub
   - Navigate to: **Settings** → **Deploy keys** → **Add deploy key**
   - Title: `GitHub Actions Data Upload`
   - Key: Paste the contents of `~/.ssh/github_actions_deploy.pub`
   - ✅ Check **"Allow write access"** (required for pushing)
   - Click **Add key**

3. **Add the private key as a Secret**:
   - Go to: **Settings** → **Secrets and variables** → **Actions**
   - Click **New repository secret**
   - Name: `SSH_DEPLOY_KEY`
   - Value: Paste the **entire contents** of `~/.ssh/github_actions_deploy` (the private key file)
   - Click **Add secret**

4. **Enable SSH authentication in the workflow**:
   - Go to: **Settings** → **Secrets and variables** → **Actions** → **Variables** tab
   - Click **New repository variable**
   - Name: `USE_SSH_AUTH`
   - Value: `true`
   - Click **Add variable**

5. **Security: Delete the local private key**:
   ```bash
   rm ~/.ssh/github_actions_deploy ~/.ssh/github_actions_deploy.pub
   ```

### 🔑 Method 2: Personal Access Token (PAT)

**Advantages:**
- Simpler to set up
- Works across multiple repositories
- Can be managed from your GitHub account settings

**Disadvantages:**
- Broader permissions (access to all your repositories)
- If compromised, affects all repositories the token has access to

**Setup Steps:**

1. **Create a Personal Access Token**:
   - Go to: https://github.com/settings/tokens
   - Click **Generate new token** → **Generate new token (classic)**
   - Token name: `GitHub Actions Workflow Data Upload`
   - Select scopes:
     - ✅ `repo` (Full control of private repositories)
     - ✅ `workflow` (Update GitHub Action workflows)
   - Set expiration (recommended: 90 days or custom)
   - Click **Generate token**
   - **Important:** Copy the token now - you won't be able to see it again!

2. **Add the token as a Secret**:
   - Go to your repository: **Settings** → **Secrets and variables** → **Actions**
   - Click **New repository secret**
   - Name: `GITHUB_PAT`
   - Value: Paste your Personal Access Token
   - Click **Add secret**

3. **Verify the Setup**:
   - The workflow will now automatically use GITHUB_PAT when pushing changes
   - Run the workflow manually to test: **Actions** → **Download SimFin Data** → **Run workflow**
   - Check the logs for: "✓ Personal Access Token configured successfully"

**Note:** The workflow file contains detailed inline instructions. See the "Setup Personal Access Token" step in `.github/workflows/download-simfin-data.yml` for complete documentation.

## Configuration Options

### Data Branch Configuration

By default, the workflow pushes data updates to a branch called `data-updates`. You can customize this:

1. Go to: **Settings** → **Secrets and variables** → **Actions** → **Variables** tab
2. Click **New repository variable**
3. Name: `DATA_BRANCH`
4. Value: Your preferred branch name (e.g., `simfin-data`, `automated-updates`)
5. Click **Add variable**

### Why Use a Separate Branch?

- **Safety**: Keeps automated data updates separate from your main codebase
- **Review**: You can review changes before merging to main
- **History**: Clearer separation between code and data changes
- **Rollback**: Easier to revert data updates if needed

## Workflow Behavior

### Authentication Priority

The workflow checks for authentication in this order:

1. If `USE_SSH_AUTH=true` and `SSH_DEPLOY_KEY` exists → Use SSH
2. If `GITHUB_PAT` exists → Use Personal Access Token
3. Otherwise → Fall back to default `GITHUB_TOKEN` (limited permissions on forks)

### Branch Strategy

1. Workflow runs on the scheduled time or manual trigger
2. Data is downloaded to the `stock_data/` directory
3. Changes are committed to the `data-updates` branch (or custom branch)
4. Branch is pushed to the remote repository
5. You can review and merge to `main` when ready

## Testing Your Setup

### Manual Workflow Run

1. Go to: **Actions** tab in your repository
2. Select **"Download SimFin Data"** workflow
3. Click **"Run workflow"** dropdown
4. Select the branch (usually `main`)
5. Click **"Run workflow"** button

### Verify Success

Check the workflow logs for:
- ✅ Git configured successfully
- ✅ SSH authentication configured (if using SSH)
- ✅ Personal Access Token configured (if using PAT)
- ✅ Changes committed successfully
- ✅ Changes pushed successfully to branch: data-updates

### Common Issues

#### Issue: "Host key verification failed"
**Solution:** The SSH key wasn't properly added to the repository. Double-check Deploy Keys settings.

#### Issue: "Permission denied (publickey)"
**Solution:** 
- Verify the private key was copied correctly (including header/footer)
- Ensure "Allow write access" is checked for the Deploy Key

#### Issue: "Authentication failed"
**Solution (PAT):**
- Verify the token has `repo` and `workflow` scopes
- Check if the token has expired
- Ensure the token was copied correctly without extra spaces

#### Issue: Still getting "can not upload new objects to public fork"
**Solution:**
- Verify that either `SSH_DEPLOY_KEY` or `GITHUB_PAT` secret exists
- Check workflow logs to confirm authentication method was used
- Ensure the secret values are correct

## Security Best Practices

### For SSH Deploy Keys

✅ **DO:**
- Use Ed25519 keys (more secure than RSA)
- Generate keys without passphrases for automation
- Delete local copies after adding to GitHub
- Rotate keys periodically (every 6-12 months)
- Use repository-specific keys (one per repo)

❌ **DON'T:**
- Share private keys via email or messaging
- Reuse the same key across multiple repositories
- Commit keys to version control
- Use keys with passphrases in automation

### For Personal Access Tokens

✅ **DO:**
- Set expiration dates (90 days recommended)
- Use minimal required scopes
- Rotate tokens before expiration
- Revoke tokens immediately if compromised
- Use different tokens for different purposes

❌ **DON'T:**
- Create tokens with no expiration
- Grant unnecessary scopes (like `admin:org`)
- Share tokens with others
- Hardcode tokens in scripts or workflows
- Use the same token across multiple services

### General Security

1. **Secrets Management:**
   - Never log secret values in workflows
   - Don't echo secrets to console or files
   - Regularly audit repository secrets

2. **Access Control:**
   - Limit who can modify workflow files
   - Review workflow changes in pull requests
   - Enable branch protection for main branch

3. **Monitoring:**
   - Review workflow run logs regularly
   - Check for unexpected push events
   - Monitor repository access patterns

## Troubleshooting

### Debug Mode

Enable detailed logging by adding this to your workflow run:

1. Go to workflow run page
2. Click **"Re-run jobs"** → **"Enable debug logging"**
3. Review detailed authentication steps

### Check Current Configuration

Run these commands locally to verify your setup:

```bash
# Check if repository is a fork
git remote -v

# Check deploy keys (requires GitHub CLI)
gh repo deploy-key list

# Check secrets (names only, not values)
gh secret list
```

### Manual Push Test

To test authentication manually:

```bash
# For SSH
git remote set-url origin git@github.com:YOUR_USERNAME/ai_investor.git
git push origin data-updates

# For PAT
git remote set-url origin https://x-access-token:YOUR_TOKEN@github.com/YOUR_USERNAME/ai_investor.git
git push origin data-updates
```

## Migration from Previous Setup

If you're upgrading from the previous workflow version:

1. **Current state:** Workflow fails with bot permission error
2. **Choose authentication method:** SSH (recommended) or PAT
3. **Follow setup steps** for your chosen method
4. **Test the workflow** with a manual run
5. **Verify data branch** is created and populated
6. **Merge data to main** when ready (optional)

## Workflow File Reference

The workflow has been updated with these new steps:

1. **Setup SSH Deploy Key** - Configures SSH authentication if enabled
2. **Commit changes** - Creates/switches to data branch
3. **Setup Personal Access Token** - Configures PAT if SSH not used
4. **Push changes** - Pushes to the data branch with retry logic

## Additional Resources

- [GitHub Deploy Keys Documentation](https://docs.github.com/en/developers/overview/managing-deploy-keys)
- [GitHub Personal Access Tokens](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/creating-a-personal-access-token)
- [GitHub Actions Secrets](https://docs.github.com/en/actions/security-guides/encrypted-secrets)
- [Git LFS Documentation](https://git-lfs.github.com/)
- [SSH Key Generation Guide](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent)

## Support

If you continue to experience issues:

1. Check the workflow run logs for detailed error messages
2. Verify all secrets and variables are set correctly
3. Review this documentation for missed steps
4. Check GitHub Status page for service issues: https://www.githubstatus.com/

## Summary

This setup resolves the fork permission issue by:
- ✅ Using dedicated authentication (SSH or PAT) instead of bot token
- ✅ Pushing to a separate data branch for safety
- ✅ Providing clear error messages and troubleshooting steps
- ✅ Following security best practices for credential management
- ✅ Supporting flexible configuration through repository variables

Choose your preferred authentication method and follow the setup steps to enable automated SimFin data updates in your fork.
