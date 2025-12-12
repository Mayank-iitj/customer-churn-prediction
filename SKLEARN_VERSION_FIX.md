# Permanent Fix for scikit-learn `monotonic_cst` AttributeError

## Explanation

The `'DecisionTreeClassifier' object has no attribute 'monotonic_cst'` error occurs because of a **version mismatch** between the scikit-learn used for training and the one used for inference. Specifically, scikit-learn 1.4+ introduced new tree attributes like `monotonic_cst` and enhanced missing-value support. When you pickle a model trained with sklearn 1.3.x and unpickle it in sklearn 1.4+, the internal `DecisionTreeClassifier` estimators don't have these new attributes, causing AttributeError when the new sklearn code tries to access them. The solution is to use the **same pinned sklearn version** for both training and deployment.

---

## Permanent Fix Plan

### Step 1: Pin scikit-learn Version in All Requirements Files
Use **scikit-learn==1.5.2** (latest stable that's widely compatible) or **sklearn==1.3.2** (if you need older Python compatibility). We'll use **1.5.2** for this fix as it's more modern and stable.

### Step 2: Retrain and Resave Model
Retrain your model with the pinned sklearn version to ensure all internal tree attributes match what sklearn expects.

### Step 3: Update All Requirements Files
Ensure `requirements.txt`, `requirements-streamlit.txt`, and any other dependency files use the exact same sklearn version.

### Step 4: Deploy with Consistency
After updating requirements, Streamlit Cloud will automatically rebuild the environment. Verify the sklearn version matches in both environments.

### Step 5: Document Upgrade Process
When upgrading sklearn in the future, **always retrain and resave models** with the new version before deploying.

---

## Code & Config

### 1. requirements.txt (for Streamlit Cloud deployment)

```txt
# Core ML Libraries - MUST MATCH TRAINING ENVIRONMENT
scikit-learn==1.5.2
pandas>=1.3.0,<3.0.0
numpy>=1.21.0,<2.0.0

# Optional ML Libraries
xgboost>=1.5.0,<3.0.0
lightgbm>=3.3.0,<5.0.0

# Streamlit
streamlit>=1.10.0

# Visualization
matplotlib>=3.5.0
seaborn>=0.11.0

# Utilities
joblib>=1.1.0
```

### 2. requirements-dev.txt (for local development - optional)

```txt
# Include all production requirements
-r requirements.txt

# Development tools
jupyter>=1.0.0
notebook>=6.4.0
ipykernel>=6.0.0
pytest>=7.0.0
black>=22.0.0
flake8>=4.0.0
```

### 3. Training Script with Version Check

**File: `train_model_fixed.py`**

```python
"""
Training script with sklearn version enforcement.
This ensures the model is trained with the correct sklearn version.
"""

import sys
import sklearn
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import joblib
import os
from datetime import datetime

# ENFORCE SKLEARN VERSION
REQUIRED_SKLEARN_VERSION = "1.5.2"

def check_sklearn_version():
    """Verify sklearn version matches requirements."""
    current_version = sklearn.__version__
    if current_version != REQUIRED_SKLEARN_VERSION:
        print(f"⚠️  WARNING: sklearn version mismatch!")
        print(f"   Current: {current_version}")
        print(f"   Required: {REQUIRED_SKLEARN_VERSION}")
        print(f"\n   Install correct version:")
        print(f"   pip install scikit-learn=={REQUIRED_SKLEARN_VERSION}")
        response = input("\nContinue anyway? (y/n): ")
        if response.lower() != 'y':
            sys.exit(1)
    else:
        print(f"✅ sklearn version {current_version} matches requirements")

def train_model():
    """Train model with version-controlled sklearn."""
    
    # Check version first
    check_sklearn_version()
    
    print("\n" + "="*60)
    print("TRAINING CUSTOMER CHURN PREDICTION MODEL")
    print("="*60)
    
    # Load data
    print("\n1. Loading data...")
    data_path = 'data/customer_churn_large.csv'
    
    if not os.path.exists(data_path):
        print(f"❌ Error: {data_path} not found")
        print("   Please ensure the dataset exists")
        return
    
    df = pd.read_csv(data_path)
    print(f"   Loaded {len(df):,} samples with {df.shape[1]} columns")
    
    # Prepare features and target
    print("\n2. Preparing features...")
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    
    # Identify categorical and numerical columns
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    numerical_cols = X.select_dtypes(include=['number']).columns.tolist()
    
    print(f"   Categorical features: {len(categorical_cols)}")
    print(f"   Numerical features: {len(numerical_cols)}")
    
    # Encode categorical variables
    print("\n3. Encoding categorical variables...")
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        label_encoders[col] = le
    
    # Scale numerical features
    print("\n4. Scaling numerical features...")
    scaler = StandardScaler()
    if numerical_cols:
        X[numerical_cols] = scaler.fit_transform(X[numerical_cols])
    
    # Split data
    print("\n5. Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"   Training samples: {len(X_train):,}")
    print(f"   Test samples: {len(X_test):,}")
    
    # Train model
    print("\n6. Training Random Forest model...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1,
        verbose=0
    )
    
    model.fit(X_train, y_train)
    print("   ✅ Training complete!")
    
    # Evaluate
    print("\n7. Evaluating model...")
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)
    
    print(f"\n   Accuracy: {accuracy:.4f}")
    print(f"   ROC AUC: {roc_auc:.4f}")
    
    print("\n   Classification Report:")
    print(classification_report(y_test, y_pred))
    
    # Save model and preprocessor
    print("\n8. Saving model and preprocessor...")
    
    # Create models directory
    os.makedirs('models', exist_ok=True)
    
    # Save model
    model_path = 'models/best_model.pkl'
    joblib.dump(model, model_path)
    print(f"   ✅ Model saved to {model_path}")
    
    # Save preprocessor as dict (version-safe format)
    preprocessor_info = {
        'label_encoders': label_encoders,
        'scaler': scaler,
        'categorical_cols': categorical_cols,
        'numerical_cols': numerical_cols,
        'feature_names': X.columns.tolist(),
        'sklearn_version': sklearn.__version__,  # Track version
        'trained_date': datetime.now().isoformat()
    }
    
    preprocessor_path = 'models/preprocessor.pkl'
    joblib.dump(preprocessor_info, preprocessor_path)
    print(f"   ✅ Preprocessor saved to {preprocessor_path}")
    
    # Save version info
    version_info = {
        'sklearn_version': sklearn.__version__,
        'pandas_version': pd.__version__,
        'numpy_version': np.__version__,
        'trained_date': datetime.now().isoformat(),
        'model_type': type(model).__name__,
        'accuracy': accuracy,
        'roc_auc': roc_auc
    }
    
    version_path = 'models/version_info.txt'
    with open(version_path, 'w') as f:
        for key, value in version_info.items():
            f.write(f"{key}: {value}\n")
    print(f"   ✅ Version info saved to {version_path}")
    
    print("\n" + "="*60)
    print("✅ MODEL TRAINING COMPLETE!")
    print("="*60)
    print(f"\nModel Performance:")
    print(f"  • Accuracy: {accuracy:.2%}")
    print(f"  • ROC AUC: {roc_auc:.4f}")
    print(f"\nModel files saved:")
    print(f"  • {model_path}")
    print(f"  • {preprocessor_path}")
    print(f"  • {version_path}")
    print(f"\nscikit-learn version: {sklearn.__version__}")
    print("="*60)

if __name__ == "__main__":
    train_model()
```

### 4. Updated app.py Loading Code

**Add version check to `app.py`:**

```python
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import sklearn
import os

# Version check
EXPECTED_SKLEARN_VERSION = "1.5.2"

def verify_sklearn_version():
    """Check if sklearn version matches expected version."""
    current_version = sklearn.__version__
    if current_version != EXPECTED_SKLEARN_VERSION:
        st.warning(f"⚠️ sklearn version mismatch: Current={current_version}, Expected={EXPECTED_SKLEARN_VERSION}")
        return False
    return True

@st.cache_resource
def load_model_and_preprocessor():
    """Load trained model and preprocessor with version verification."""
    try:
        # Verify sklearn version
        if not verify_sklearn_version():
            st.error(f"sklearn version mismatch. Please use version {EXPECTED_SKLEARN_VERSION}")
            st.stop()
        
        # Load model
        model_path = 'models/best_model.pkl'
        preprocessor_path = 'models/preprocessor.pkl'
        
        if not os.path.exists(model_path):
            st.error(f"Model file not found: {model_path}")
            return None, None
        
        model = joblib.load(model_path)
        preprocessor = joblib.load(preprocessor_path) if os.path.exists(preprocessor_path) else None
        
        # Check preprocessor sklearn version if available
        if preprocessor and isinstance(preprocessor, dict):
            saved_version = preprocessor.get('sklearn_version', 'unknown')
            if saved_version != sklearn.__version__:
                st.warning(f"Model trained with sklearn {saved_version}, running with {sklearn.__version__}")
        
        st.success(f"✅ Model loaded successfully (sklearn {sklearn.__version__})")
        return model, preprocessor
        
    except Exception as e:
        st.error(f"Error loading model: {e}")
        import traceback
        st.code(traceback.format_exc())
        return None, None

def predict_churn(model, preprocessor, data):
    """Make prediction with version-safe preprocessing."""
    try:
        data_processed = data.copy()
        
        # Apply preprocessing if available
        if preprocessor and isinstance(preprocessor, dict):
            # Encode categorical variables
            if 'label_encoders' in preprocessor:
                for col, encoder in preprocessor['label_encoders'].items():
                    if col in data_processed.columns:
                        data_processed[col] = encoder.transform(data_processed[col].astype(str))
            
            # Scale numerical columns
            if 'scaler' in preprocessor and 'numerical_cols' in preprocessor:
                num_cols = preprocessor['numerical_cols']
                if num_cols and all(col in data_processed.columns for col in num_cols):
                    data_processed[num_cols] = preprocessor['scaler'].transform(data_processed[num_cols])
        
        # Make prediction
        prediction = model.predict(data_processed)
        probability = model.predict_proba(data_processed)
        
        return prediction[0], probability[0]
        
    except Exception as e:
        st.error(f"Error making prediction: {e}")
        import traceback
        st.code(traceback.format_exc())
        return None, None
```

### 5. Deployment Checklist for Streamlit Cloud

**When deploying to Streamlit Cloud:**

1. ✅ **Update `requirements.txt`** with `scikit-learn==1.5.2`
2. ✅ **Retrain model locally** using `train_model_fixed.py`
3. ✅ **Commit and push** both the updated requirements AND the new model files
4. ✅ **Trigger reboot** on Streamlit Cloud (or wait for auto-rebuild)
5. ✅ **Verify logs** show correct sklearn version: `scikit-learn==1.5.2`
6. ✅ **Test predictions** to ensure no AttributeError

**Important Notes:**
- Streamlit Cloud caches dependencies. If you change `requirements.txt`, you may need to **reboot the app** from the dashboard to force a rebuild.
- Always commit both `requirements.txt` and retrained model files together.
- The `models/*.pkl` files should be in your git repository (ensure they're not in `.gitignore`).

---

## Future-Proofing: Upgrading scikit-learn

When you want to upgrade scikit-learn in the future:

```bash
# Step 1: Update requirements.txt
echo "scikit-learn==1.6.0" > requirements.txt  # New version

# Step 2: Install new version locally
pip install scikit-learn==1.6.0

# Step 3: Retrain and resave model
python train_model_fixed.py

# Step 4: Commit everything
git add requirements.txt models/*.pkl
git commit -m "Upgrade sklearn to 1.6.0 and retrain model"
git push

# Step 5: Reboot Streamlit Cloud app
```

**Golden Rule:** 🔑 **Always retrain when upgrading sklearn** to avoid internal attribute mismatches.

---

## Quick Fix Commands

```bash
# Install correct sklearn version
pip install scikit-learn==1.5.2

# Retrain model
python train_model_fixed.py

# Verify version in Python
python -c "import sklearn; print(sklearn.__version__)"

# Commit and deploy
git add requirements.txt models/*.pkl
git commit -m "Fix sklearn version to 1.5.2 and retrain model"
git push
```

---

## Summary

✅ **Root Cause:** Model trained with sklearn 1.3.x, loaded with sklearn 1.6+, causing `monotonic_cst` AttributeError  
✅ **Permanent Fix:** Pin sklearn to 1.5.2 in all requirements files and retrain model  
✅ **Future Upgrades:** Always retrain model when upgrading sklearn  
✅ **Deployment:** Commit requirements.txt + model files together, reboot Streamlit Cloud  

This approach ensures **version consistency** across training and inference, eliminating pickle compatibility issues.
