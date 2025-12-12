#!/bin/bash
# Quick fix script for deployment environments
# Run this in your deployment terminal to fix the model compatibility issue

echo "=================================="
echo "Model Compatibility Fix Script"
echo "=================================="

# Install/upgrade to correct scikit-learn version
echo "Installing scikit-learn 1.3.2..."
pip install scikit-learn==1.3.2

# Retrain models
echo "Retraining models..."
python train_simple.py

echo "=================================="
echo "✓ Fix complete!"
echo "=================================="
