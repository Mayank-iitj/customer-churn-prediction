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
    
    # Drop rows with missing target values
    df = df.dropna(subset=['Churn'])
    print(f"   After removing NaN in target: {len(df):,} samples")
    
    # Prepare features and target
    print("\n2. Preparing features...")
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    
    # Drop ID columns that shouldn't be used for training
    id_cols = ['CustomerID', 'customer_id', 'id', 'ID']
    cols_to_drop = [col for col in id_cols if col in X.columns]
    if cols_to_drop:
        X = X.drop(columns=cols_to_drop)
        print(f"   Dropped ID columns: {cols_to_drop}")
    
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
        'accuracy': float(accuracy),
        'roc_auc': float(roc_auc)
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
