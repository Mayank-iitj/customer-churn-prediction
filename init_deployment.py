"""
Deployment Initialization Script
Ensures models are compatible with the deployment environment by retraining if needed.
Run this script during deployment setup.
"""

import os
import sys
import subprocess
import joblib

def check_model_compatibility():
    """Check if existing models can be loaded."""
    try:
        sys.path.append('src')
        model = joblib.load('models/best_model.pkl')
        preprocessor = joblib.load('models/preprocessor.pkl')
        print("✓ Existing models are compatible with current environment")
        return True
    except Exception as e:
        print(f"✗ Model compatibility issue detected: {e}")
        return False

def retrain_models():
    """Retrain models for current environment."""
    print("\nRetraining models for current environment...")
    result = subprocess.run([sys.executable, 'train_simple.py'], 
                          capture_output=True, text=True)
    if result.returncode == 0:
        print("✓ Models retrained successfully")
        return True
    else:
        print(f"✗ Retraining failed: {result.stderr}")
        return False

def main():
    print("="*80)
    print("Deployment Initialization")
    print("="*80)
    
    # Check if data exists
    if not os.path.exists('data/customer_churn.csv'):
        print("\n⚠️  Warning: data/customer_churn.csv not found")
        print("   Make sure to upload your training data before running this script")
        return 1
    
    # Check model compatibility
    print("\nChecking model compatibility...")
    if not check_model_compatibility():
        # Models need retraining
        if not retrain_models():
            print("\n✗ Deployment initialization failed")
            return 1
    
    print("\n" + "="*80)
    print("✓ Deployment initialization complete!")
    print("="*80)
    return 0

if __name__ == "__main__":
    sys.exit(main())
