# 🚀 Deployment Guide

## Option 1: Deploy to Streamlit Cloud (Recommended)

### Step 1: Prepare GitHub Repository

1. Create a new repository on GitHub
2. Upload all files from this project:
   - app.py
   - requirements.txt
   - .streamlit/config.toml
   - README.md
   - .gitignore

### Step 2: Deploy on Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Click "New app"
3. Connect your GitHub account
4. Select your repository
5. Set main file path: `app.py`
6. Click "Deploy"

### Step 3: Add API Keys

1. Go to App Settings (⚙️ icon)
2. Click "Secrets"
3. Add your secrets in TOML format:
```toml
# No secrets needed - users enter API keys in the app!
```

**Note**: This app uses a user-input approach for API keys, so no secrets configuration needed!

### Step 4: Share Your App

Your app will be available at: `https://<your-app-name>.streamlit.app`

---

## Option 2: Run Locally

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation Steps

1. Clone/Download the repository
```bash
cd interview_prep_app
```

2. Create virtual environment (optional but recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

4. Run the app
```bash
streamlit run app.py
```

5. Open browser to `http://localhost:8501`

---

## Option 3: Deploy to Other Platforms

### Heroku

1. Create `Procfile`:
```
web: streamlit run app.py --server.port=$PORT
```

2. Create `setup.sh`:
```bash
mkdir -p ~/.streamlit/
echo "[server]
headless = true
port = $PORT
enableCORS = false
" > ~/.streamlit/config.toml
```

3. Deploy:
```bash
heroku create your-app-name
git push heroku main
```

### Google Cloud Run

1. Create `Dockerfile`:
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501"]
```

2. Deploy:
```bash
gcloud run deploy --source .
```

---

## 🔑 Getting API Keys

### Groq API Key
1. Visit [console.groq.com](https://console.groq.com/keys)
2. Sign up with Google account
3. Click "Create API Key"
4. Copy the key (starts with `gsk_...`)
5. **Free tier**: 14,400 requests/day

### Gemini API Key
1. Visit [aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey)
2. Sign in with Google account
3. Click "Create API Key"
4. Copy the key
5. **Free tier**: Very generous limits

---

## 🐛 Troubleshooting

### Issue: App won't start
- Check Python version: `python --version` (need 3.8+)
- Reinstall dependencies: `pip install -r requirements.txt --force-reinstall`

### Issue: API errors
- Verify API keys are correct
- Check API rate limits
- Ensure internet connection

### Issue: PDF parsing fails
- Ensure PDF is not password protected
- Try re-saving PDF from Word/Google Docs
- Check file size (< 10MB recommended)

### Issue: Slow response
- This is normal for first request (model loading)
- Groq is very fast for subsequent requests
- Check your internet speed

---

## 📊 Performance Tips

1. **Optimize PDF Size**: Keep resumes under 5MB
2. **Use Groq for Speed**: Technical questions use Groq for fast responses
3. **Cache Results**: Session state keeps data during navigation
4. **Batch Processing**: Generate all interview questions at once

---

## 🔒 Security Best Practices

1. **Never commit API keys** to Git
2. **Use environment variables** in production
3. **Enable rate limiting** if deploying publicly
4. **Validate user inputs** before processing
5. **Use HTTPS** in production

---

## 📈 Monitoring

### Streamlit Cloud
- Built-in analytics in dashboard
- View logs in real-time
- Monitor resource usage

### Local Development
- Check terminal for logs
- Use `st.write()` for debugging
- Enable Streamlit debug mode: `streamlit run app.py --server.runOnSave true`

---

## 🎓 For Your Project Report

### Deployment Section

Include these points in your report:

1. **Platform Used**: Streamlit Cloud / Local
2. **Hosting**: Free tier on Streamlit Cloud
3. **Technologies**: Python, Streamlit, Groq AI, Gemini AI
4. **Architecture**: Client-server model with API integration
5. **Scalability**: Handles multiple concurrent users
6. **Cost**: Free (using free API tiers)

### Screenshots to Include

1. Landing page with API configuration
2. Resume analysis results
3. Job matching scores
4. Mock interview interface
5. Performance dashboard
6. Deployment on Streamlit Cloud

---

## ✅ Pre-Deployment Checklist

- [ ] All files present (app.py, requirements.txt, etc.)
- [ ] Tested locally with `streamlit run app.py`
- [ ] API keys work correctly
- [ ] README.md updated with your info
- [ ] .gitignore includes sensitive files
- [ ] Sample data for testing
- [ ] Screenshots for documentation

---

**Need help?** Check Streamlit docs at [docs.streamlit.io](https://docs.streamlit.io)
