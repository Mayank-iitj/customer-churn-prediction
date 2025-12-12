#!/bin/bash
# Streamlit Cloud Startup Script
# This script runs before the Streamlit app starts to ensure model compatibility

echo "========================================"
echo "Streamlit Cloud Initialization"
echo "========================================"

# Check if data exists
if [ ! -f "data/customer_churn_large.csv" ]; then
    echo "⚠️  Warning: Training data not found"
    echo "   Using sample data for model training"
    # You may need to download or use smaller dataset
fi

# Check if models exist and are compatible
echo "Checking model compatibility..."
python3 -c "
import sys
import joblib
import os

try:
    if os.path.exists('models/best_model.pkl'):
        model = joblib.load('models/best_model.pkl')
        print('✓ Model loaded successfully')
        sys.exit(0)
    else:
        print('✗ Model not found')
        sys.exit(1)
except Exception as e:
    print(f'✗ Model compatibility issue: {e}')
    sys.exit(1)
"

if [ $? -ne 0 ]; then
    echo "Retraining model for current environment..."
    if [ -f "train_quick.py" ]; then
        python3 train_quick.py
        if [ $? -eq 0 ]; then
            echo "✓ Model retrained successfully"
        else
            echo "✗ Model retraining failed"
        fi
    elif [ -f "train_simple.py" ]; then
        python3 train_simple.py
        if [ $? -eq 0 ]; then
            echo "✓ Model retrained successfully"
        else
            echo "✗ Model retraining failed"
        fi
    fi
fi

echo "========================================"
echo "Initialization complete"
echo "========================================"
