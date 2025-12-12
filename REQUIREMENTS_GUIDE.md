# Requirements Files Guide

This project includes multiple requirements files for different deployment scenarios:

## 📋 Requirements Files

### `requirements.txt` (Default)
**Use for:** Standard deployment, Docker, most cloud platforms
- Core dependencies only
- Compatible with Python 3.11-3.13
- Excludes Jupyter and development tools
- **Install:** `pip install -r requirements.txt`

### `requirements-streamlit-cloud.txt`
**Use for:** Streamlit Cloud deployment
- Optimized for Streamlit Cloud environment
- Uses latest stable versions
- Minimal dependencies for faster deployment
- **Install:** `pip install -r requirements-streamlit-cloud.txt`

### `requirements-prod.txt`
**Use for:** Production deployments with strict version control
- Pinned version ranges for stability
- Excludes optional dependencies
- Smaller deployment footprint
- **Install:** `pip install -r requirements-prod.txt`

### `requirements-dev.txt`
**Use for:** Local development
- Includes all development tools (Jupyter, testing, etc.)
- Additional utilities for model training
- Code quality tools
- **Install:** `pip install -r requirements-dev.txt`

### `requirements-streamlit.txt`
**Use for:** Alternative Streamlit deployment
- Legacy Streamlit-specific dependencies
- **Install:** `pip install -r requirements-streamlit.txt`

## 🚀 Quick Start

### For Streamlit Cloud:
1. Use `requirements-streamlit-cloud.txt`
2. Set Python version to 3.11 (recommended)
3. The app will auto-retrain models if compatibility issues occur

### For Local Development:
```bash
pip install -r requirements-dev.txt
python train_quick.py
streamlit run app.py
```

### For Docker/Production:
```bash
pip install -r requirements.txt
python init_deployment.py
streamlit run app.py
```

## ⚠️ Common Issues

### Error: "cannot import name '_is_pandas_df'"
**Solution:** Models need retraining for your environment
```bash
python train_simple.py
# or
python init_deployment.py
```

### Error: "package not found" or version conflicts
**Solution:** Use the appropriate requirements file:
- Streamlit Cloud → `requirements-streamlit-cloud.txt`
- Docker → `requirements.txt`
- Local dev → `requirements-dev.txt`

### Error: Python version incompatible
**Recommended:** Python 3.11
**Supported:** Python 3.11-3.13
- Some packages may not support Python 3.13 yet
- Use Python 3.11 for best compatibility

## 🔄 Updating Dependencies

To update all dependencies to latest compatible versions:
```bash
pip install --upgrade -r requirements.txt
```

To check for outdated packages:
```bash
pip list --outdated
```
