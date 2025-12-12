# Deployment Instructions for Streamlit Cloud / Remote Servers

## Issue: Model Compatibility Error

If you encounter this error:
```
Error loading model: cannot import name '_is_pandas_df' from 'sklearn.utils.validation'
```

This happens when models trained with one scikit-learn version are loaded in an environment with a different version.

## Solution

### Option 1: Automatic Fix (Recommended)
The app now includes automatic model retraining when compatibility issues are detected. Simply:

1. Deploy the app normally
2. The app will detect the compatibility issue and retrain automatically on first load
3. Wait for the "Model retrained successfully!" message

### Option 2: Manual Retraining on Deployment

If you're deploying to Streamlit Cloud or a remote server:

1. **Add startup command** in your deployment configuration:
   ```bash
   python init_deployment.py && streamlit run app.py
   ```

2. **Or run manually after deployment:**
   ```bash
   python train_simple.py
   ```

### Option 3: Pin Python and Package Versions

To avoid compatibility issues, ensure your deployment environment matches your development environment:

1. **Python version:** 3.11 (or match your local version)
2. **scikit-learn:** 1.3.2 (pinned in requirements.txt)

## Streamlit Cloud Deployment

### Setup Steps:

1. **Fork/Clone repository**
2. **Connect to Streamlit Cloud**
3. **Set Python version** (if available): 3.11
4. **Deploy**

### If models don't load:

1. Go to your app's settings in Streamlit Cloud
2. Add a **reboot** to clear cache
3. Or add this to `.streamlit/config.toml`:
   ```toml
   [server]
   runOnSave = true
   ```

## Docker Deployment

Add this to your Dockerfile BEFORE the CMD:
```dockerfile
RUN python init_deployment.py
```

## Environment Variables (Optional)

You can set these environment variables for better control:

- `RETRAIN_ON_LOAD=true` - Enable automatic retraining
- `MODEL_PATH=models/` - Custom model directory

## Verification

After deployment, verify the models are loaded:

```bash
python -c "import sys; sys.path.append('src'); import joblib; joblib.load('models/best_model.pkl'); print('✓ Models OK')"
```

## Support

If issues persist:
1. Check Python version: `python --version`
2. Check scikit-learn version: `pip show scikit-learn`
3. Ensure versions match requirements.txt
4. Run `python init_deployment.py` manually
