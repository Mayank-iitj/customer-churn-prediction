# Streamlit App Configuration for Cloud Deployment

This guide helps you deploy the Customer Churn Prediction app to Streamlit Cloud.

## Quick Deploy to Streamlit Cloud

1. **Push to GitHub** (if not already done):
   ```bash
   git init
   git add .
   git commit -m "Initial commit - Streamlit ready"
   git remote add origin https://github.com/yourusername/customer-churn-prediction.git
   git push -u origin main
   ```

2. **Visit Streamlit Cloud**:
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Sign in with GitHub
   - Click "New app"

3. **Configure Deployment**:
   - Repository: `yourusername/customer-churn-prediction`
   - Branch: `main`
   - Main file path: `app.py`
   - Click "Deploy!"

## Pre-Deployment Checklist

### Required Files (Already Included)
- ✅ `app.py` - Main Streamlit application
- ✅ `requirements.txt` - Python dependencies
- ✅ `packages.txt` - System dependencies
- ✅ `.streamlit/config.toml` - Streamlit configuration

### Environment Setup

1. **Train Model First** (Important!):
   ```bash
   python main.py
   ```
   This creates the trained model files needed by the app.

2. **Copy Model Files**:
   - Streamlit Cloud needs the trained model
   - Option 1: Commit model files (if small < 100MB)
   - Option 2: Use Streamlit secrets to download from cloud storage
   - Option 3: Train model on first run (see below)

### Data Handling

**Option A: Include Sample Data** (Recommended for demo)
```bash
# Add small sample dataset
cp data/sample_customer_churn.csv data/customer_churn.csv
git add data/customer_churn.csv
git commit -m "Add sample data"
```

**Option B: User Upload Only**
- Remove data file requirement
- Users upload their own CSV files
- Modify `load_model_and_preprocessor()` to handle missing models

**Option C: Cloud Storage**
- Store data/models in S3, Azure Blob, or Google Cloud Storage
- Use Streamlit secrets for credentials
- Download on app startup

## Streamlit Secrets Configuration

For production deployments with sensitive data:

1. **In Streamlit Cloud Dashboard**:
   - Go to app settings
   - Click "Secrets"
   - Add your secrets in TOML format

2. **Example Secrets** (`.streamlit/secrets.toml` locally, never commit!):
   ```toml
   # AWS S3 for model storage
   [aws]
   aws_access_key_id = "YOUR_ACCESS_KEY"
   aws_secret_access_key = "YOUR_SECRET_KEY"
   region = "us-east-1"
   bucket = "your-bucket-name"
   
   # Database credentials
   [database]
   host = "your-db-host"
   port = 5432
   database = "churn_db"
   username = "your_username"
   password = "your_password"
   ```

3. **Access Secrets in Code**:
   ```python
   import streamlit as st
   
   # Access secrets
   aws_key = st.secrets["aws"]["aws_access_key_id"]
   db_host = st.secrets["database"]["host"]
   ```

## Model Storage Solutions

### Option 1: GitHub (Simple, for small models)
```python
# Models stored in repository
# Works if model files < 100MB
```

### Option 2: GitHub LFS (Large File Storage)
```bash
# Install Git LFS
git lfs install

# Track model files
git lfs track "*.pkl"
git add .gitattributes
git add models/*.pkl
git commit -m "Add models with LFS"
```

### Option 3: Cloud Storage (Recommended for production)

**Using AWS S3**:
```python
import streamlit as st
import boto3
import joblib
import tempfile

@st.cache_resource
def load_model_from_s3():
    s3 = boto3.client(
        's3',
        aws_access_key_id=st.secrets["aws"]["aws_access_key_id"],
        aws_secret_access_key=st.secrets["aws"]["aws_secret_access_key"],
        region_name=st.secrets["aws"]["region"]
    )
    
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        s3.download_file(
            st.secrets["aws"]["bucket"],
            'models/best_model.pkl',
            tmp_file.name
        )
        model = joblib.load(tmp_file.name)
    
    return model
```

**Using Google Cloud Storage**:
```python
from google.cloud import storage
import streamlit as st
import joblib

@st.cache_resource
def load_model_from_gcs():
    client = storage.Client()
    bucket = client.bucket(st.secrets["gcp"]["bucket"])
    blob = bucket.blob('models/best_model.pkl')
    
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        blob.download_to_filename(tmp_file.name)
        model = joblib.load(tmp_file.name)
    
    return model
```

## Memory Management

Streamlit Cloud has memory limits. Optimize your app:

1. **Use Caching**:
   ```python
   @st.cache_resource  # For models (singleton)
   @st.cache_data      # For data processing
   ```

2. **Limit Data Size**:
   ```python
   # Set max file size for uploads
   st.file_uploader("Upload CSV", type=['csv'], 
                    help="Max file size: 200MB")
   ```

3. **Clear Matplotlib Figures**:
   ```python
   fig, ax = plt.subplots()
   # ... plotting code ...
   st.pyplot(fig)
   plt.close(fig)  # Important!
   ```

## Environment Variables

Set these in Streamlit Cloud settings or `.streamlit/config.toml`:

```toml
[server]
port = 8501
enableCORS = false
enableXsrfProtection = true

[browser]
gatherUsageStats = false

[theme]
primaryColor = "#1f77b4"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#f0f2f6"
textColor = "#262730"
```

## Custom Domain

1. **In Streamlit Cloud**:
   - Go to app settings
   - Click "Custom domain"
   - Follow instructions to add CNAME record

2. **DNS Configuration**:
   ```
   Type: CNAME
   Name: app (or your subdomain)
   Value: [your-app].streamlit.app
   ```

## Monitoring & Analytics

### Built-in Analytics
- Streamlit Cloud provides basic analytics
- View in the app dashboard

### Custom Analytics
```python
import streamlit as st
from datetime import datetime

# Log usage
if 'usage_log' not in st.session_state:
    st.session_state.usage_log = []

st.session_state.usage_log.append({
    'timestamp': datetime.now(),
    'action': 'prediction_made',
    'user': st.experimental_user.email  # If authentication enabled
})
```

## Troubleshooting

### Common Issues

**App won't start**:
- Check requirements.txt for all dependencies
- Verify app.py has no syntax errors
- Check Streamlit Cloud logs

**Model not found**:
- Ensure model files are in repository or accessible
- Check file paths are correct
- Verify model was trained and saved

**Out of memory**:
- Reduce model size or use simpler models
- Implement data sampling for large datasets
- Use external storage for large files

**Slow performance**:
- Use `@st.cache_resource` and `@st.cache_data`
- Optimize data processing
- Consider upgrading Streamlit Cloud tier

### Debug Mode

```python
import streamlit as st

# Enable debug info
if st.checkbox("Show Debug Info"):
    st.write("Session State:", st.session_state)
    st.write("Secrets Available:", list(st.secrets.keys()))
    st.write("Current Directory:", os.getcwd())
    st.write("Files:", os.listdir())
```

## Best Practices

1. **Always cache models and data**:
   ```python
   @st.cache_resource
   def load_model():
       return joblib.load('model.pkl')
   ```

2. **Handle errors gracefully**:
   ```python
   try:
       result = model.predict(data)
   except Exception as e:
       st.error(f"Prediction failed: {e}")
       st.stop()
   ```

3. **Provide user feedback**:
   ```python
   with st.spinner("Processing..."):
       # Long-running operation
       result = process_data()
   st.success("Done!")
   ```

4. **Validate user input**:
   ```python
   if uploaded_file is not None:
       if uploaded_file.size > 200_000_000:  # 200MB
           st.error("File too large!")
           st.stop()
   ```

5. **Close matplotlib figures**:
   ```python
   fig, ax = plt.subplots()
   # plotting...
   st.pyplot(fig)
   plt.close(fig)  # Always close!
   ```

## Resources

- [Streamlit Documentation](https://docs.streamlit.io)
- [Streamlit Cloud](https://streamlit.io/cloud)
- [Community Forum](https://discuss.streamlit.io)
- [GitHub Examples](https://github.com/streamlit/streamlit-example)

## Support

For issues specific to this app:
1. Check logs in Streamlit Cloud dashboard
2. Review this documentation
3. Open an issue on GitHub

---

**Your app is now Streamlit Cloud ready!** 🚀
