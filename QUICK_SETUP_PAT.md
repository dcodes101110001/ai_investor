# Quick Setup Guide: GitHub Personal Access Token (PAT)

## 🎯 Purpose
This guide helps you set up authentication for the SimFin data download workflow to work on public forks.

## ⚠️ When Do You Need This?
If you see this error in your workflow runs:
```
@github-actions[bot] can not upload new objects to public fork
```

You need to configure a GitHub Personal Access Token (PAT).

## 🚀 Setup Steps (5 minutes)

### Step 1: Create Personal Access Token
1. Go to: **https://github.com/settings/tokens**
2. Click: **"Generate new token"** → **"Generate new token (classic)"**
   
   > 💡 **Note**: You can also use fine-grained personal access tokens (beta) for enhanced security. 
   > They provide more granular permissions and can be scoped to specific repositories. 
   > However, classic tokens are simpler to set up and work well for most use cases.
   
3. Fill in:
   - **Note**: `AI Investor Workflow - Data Upload`
   - **Expiration**: 90 days (or custom)
   - **Scopes**: 
     - ✅ `repo` (Full control of private repositories)
     - ✅ `workflow` (Update GitHub Action workflows)
4. Click: **"Generate token"**
5. **COPY THE TOKEN** (you won't see it again!)

### Step 2: Add Token to Repository
1. Go to your repository on GitHub
2. Click: **Settings** → **Secrets and variables** → **Actions**
3. Click: **"New repository secret"**
4. Enter:
   - **Name**: `GITHUB_PAT`
   - **Secret**: (paste the token you copied)
5. Click: **"Add secret"**

### Step 3: Test the Workflow
1. Go to: **Actions** tab
2. Click: **"Download SimFin Data"** workflow
3. Click: **"Run workflow"** → **"Run workflow"** (green button)
4. Wait for completion
5. Check logs for: `✓ Personal Access Token configured successfully`

## ✅ Success Indicators

You'll know it's working when you see:
```
✓ GITHUB_PAT secret found
✓ Personal Access Token configured successfully
✓ Authentication ready for Git LFS push operations
✓ Changes pushed successfully to branch: data-updates
```

## ❌ Common Issues

### Issue: Token expired
**Solution**: Create a new token following Step 1 above, then update the `GITHUB_PAT` secret

### Issue: Still getting permission error
**Checklist**:
- [ ] Secret is named exactly `GITHUB_PAT` (case-sensitive)
- [ ] Token has both `repo` AND `workflow` scopes
- [ ] No extra spaces when copying/pasting token
- [ ] Token hasn't expired

### Issue: Need to update existing token
1. Go to repository **Settings** → **Secrets and variables** → **Actions**
2. Click the pencil icon next to `GITHUB_PAT`
3. Paste new token
4. Click **"Update secret"**

## 🔒 Security Notes

- ✅ GitHub automatically hides token values in logs
- ✅ Tokens are encrypted at rest
- ✅ Set expiration dates on tokens (90 days recommended)
- ✅ Rotate tokens regularly
- ❌ Never commit tokens to git or share them publicly

## 📚 Alternative: SSH Deploy Key

For enhanced security, consider using SSH Deploy Keys instead:
- More secure (repository-specific)
- No expiration
- See: [WORKFLOW_AUTHENTICATION_SETUP.md](WORKFLOW_AUTHENTICATION_SETUP.md)

## 🆘 Still Need Help?

1. **Detailed Guide**: [WORKFLOW_AUTHENTICATION_SETUP.md](WORKFLOW_AUTHENTICATION_SETUP.md)
2. **Inline Documentation**: Check `.github/workflows/download-simfin-data.yml`
3. **Troubleshooting**: [WORKFLOW_FIX_DOCUMENTATION.md](WORKFLOW_FIX_DOCUMENTATION.md)

---

**Last Updated**: December 2024
