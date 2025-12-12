# Streamlit Cloud Configuration

## Automatic Deployment

For Streamlit Cloud deployment, the app will automatically:
1. Detect model compatibility issues
2. Retrain the model if needed (using train_quick.py)
3. Handle both old and new dataset formats

## Required Files

Make sure these files are in your repository:
- `app.py` - Main Streamlit application
- `train_quick.py` - Fast training script for large datasets
- `train_simple.py` - Fallback training script
- `data/customer_churn_large.csv` - Training dataset (or any CSV with customer data)
- `requirements-streamlit-cloud.txt` - Dependencies

## Deployment Settings

**In Streamlit Cloud:**
1. Set main file: `app.py`
2. Python version: 3.11 (recommended) or 3.12
3. Advanced settings:
   - Use `requirements-streamlit-cloud.txt` if available
   - Or use `requirements.txt` (default)

## Custom Startup (Optional)

If you need to run commands before app starts:
1. Create `.streamlit/config.toml`:
```toml
[server]
runOnSave = false

[runner]
magicEnabled = true
```

2. The app will auto-detect and fix compatibility issues

## Troubleshooting

### Error: "monotonic_cst" attribute missing
**Solution:** The app will automatically retrain. Wait for the message "Model retrained successfully" and refresh.

### Error: Requirements installation fails
**Solution:** 
- Use `requirements-streamlit-cloud.txt` 
- Or pin scikit-learn: `scikit-learn==1.5.2`

### Error: Out of memory during training
**Solution:** Use a smaller dataset or pre-trained models

## Manual Model Upload

If auto-retraining doesn't work:
1. Train locally: `python train_quick.py`
2. Commit `models/best_model.pkl` and `models/preprocessor.pkl`
3. Push to GitHub
4. Redeploy on Streamlit Cloud

## Data Requirements

The training dataset should have these columns:
- Age, Gender, Tenure
- Usage Frequency, Support Calls
- Payment Delay, Subscription Type
- Contract Length, Total Spend
- Last Interaction
- Churn (target variable)

Or use the old format with standard telecom columns.
