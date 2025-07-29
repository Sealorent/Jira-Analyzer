# 🚀 Streamlit Cloud Deployment Guide

## Prerequisites

1. **GitHub Repository**: Your code should be in a GitHub repository
2. **Streamlit Cloud Account**: Sign up at [share.streamlit.io](https://share.streamlit.io)
3. **JIRA & Gemini API Keys**: Required for full functionality

## Step-by-Step Deployment

### 1. Prepare Your Repository

Ensure your repository contains:
- ✅ `app.py` (main application file)
- ✅ `requirements.txt` (dependencies)
- ✅ `.streamlit/config.toml` (Streamlit configuration)
- ✅ `.streamlit/secrets.toml` (secrets template - DO NOT commit real secrets)
- ✅ `README.md` (documentation)
- ✅ `.gitignore` (excludes sensitive files)

### 2. Deploy to Streamlit Cloud

1. **Connect to GitHub**:
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Sign in with your GitHub account
   - Click "New app"

2. **Configure App Settings**:
   - **Repository**: Select your repository
   - **Branch**: `main` (or your primary branch)
   - **Main file path**: `app.py`
   - **App URL**: Choose a unique name (e.g., `jira-analyzer-yourname`)

3. **Set Up Secrets**:
   - Click "Advanced settings"
   - In the "Secrets" section, paste your configuration:

```toml
# JIRA Configuration
JIRA_BASE_URL = "https://one-employee.atlassian.net/rest/api/3/"
JIRA_AUTH_TOKEN = "your-actual-jira-auth-token-here"

# Gemini AI Configuration
GEMINI_API_KEY = "your-actual-gemini-api-key-here"

# Application Configuration
APP_VERSION = "1.0.1"
APP_NAME = "JIRA Data Analyzer"

# Feature Flags
ENABLE_DATE_RANGE_FILTERING = "true"
ENABLE_ADVANCED_ANALYTICS = "true"

# Application Defaults
DEFAULT_JQL_QUERY = "labels IN (July2025)"
MAX_RESULTS_DEFAULT = "10000"
DEFAULT_FROM_MONTH = "July2025"
DEFAULT_TO_MONTH = "July2025"
```

4. **Deploy**:
   - Click "Deploy!"
   - Wait for the deployment to complete (usually 2-5 minutes)

### 3. Configure Secrets (Alternative Method)

If you prefer to configure secrets after deployment:

1. Go to your app's dashboard
2. Click "Settings" → "Secrets"
3. Add each secret individually:
   - `JIRA_AUTH_TOKEN` = `your-token`
   - `GEMINI_API_KEY` = `your-key`
   - etc.

### 4. Access Your Deployed App

Once deployed, your app will be available at:
```
https://your-app-name.streamlit.app
```

## Security Best Practices

### ✅ DO:
- Use Streamlit Cloud's secrets management
- Keep API keys in secrets, not in code
- Use `.gitignore` to exclude sensitive files
- Regularly rotate API keys

### ❌ DON'T:
- Commit real API keys to GitHub
- Share secrets in plain text
- Use production keys for testing
- Hardcode sensitive information

## Updating Your App

To update your deployed app:

1. **Push changes to GitHub**:
   ```bash
   git add .
   git commit -m "Update app"
   git push origin main
   ```

2. **Automatic redeployment**:
   - Streamlit Cloud automatically redeploys when you push to the connected branch
   - Check the deployment status in your Streamlit Cloud dashboard

## Troubleshooting

### Common Issues:

1. **Dependencies not installing**:
   - Check `requirements.txt` for correct package names and versions
   - Ensure all imports in `app.py` match the requirements

2. **Secrets not found**:
   - Verify secrets are properly configured in Streamlit Cloud
   - Check that secret names match exactly (case-sensitive)

3. **JIRA connection fails**:
   - Verify JIRA_AUTH_TOKEN is correct and not expired
   - Check JIRA_BASE_URL is accessible

4. **Gemini AI not working**:
   - Verify GEMINI_API_KEY is valid
   - Check API quota and billing status

### Debug Steps:

1. **Check app logs**:
   - Go to your app dashboard
   - Click "Manage app" → "Logs"
   - Look for error messages

2. **Test locally**:
   ```bash
   streamlit run app.py
   ```

3. **Verify secrets**:
   - Test with a minimal configuration first
   - Add secrets one by one to isolate issues

## Monitoring & Maintenance

### Performance:
- Monitor app usage in Streamlit Cloud dashboard
- Check response times and resource usage
- Consider caching strategies for better performance

### Updates:
- Keep dependencies updated
- Monitor security advisories
- Test changes in staging before production

### Backup:
- Keep backup of your configuration
- Document any custom settings
- Maintain version history in Git

## Advanced Configuration

### Custom Domain (Paid Plans):
- Configure custom domain in Streamlit Cloud settings
- Set up CNAME record with your DNS provider

### Analytics:
- Enable Google Analytics in `config.toml`
- Monitor user engagement and feature usage

### Performance Optimization:
- Use `@st.cache_data` for expensive operations
- Implement session state for user preferences
- Optimize JIRA queries for better response times

## Support

For deployment issues:
- [Streamlit Cloud Documentation](https://docs.streamlit.io/streamlit-cloud)
- [Community Forum](https://discuss.streamlit.io/)
- [GitHub Issues](https://github.com/streamlit/streamlit/issues)

---

**Happy Deploying! 🚀**