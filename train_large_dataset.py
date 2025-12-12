"""
Train Customer Churn Model with New Dataset
Handles the different column structure from archive dataset
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import xgboost as xgb
import os
from datetime import datetime

def load_and_preprocess_data(filepath):
    """Load and preprocess the new dataset format."""
    print(f"\nLoading data from {filepath}...")
    df = pd.read_csv(filepath)
    
    print(f"Original shape: {df.shape}")
    
    # Drop CustomerID as it's not useful for prediction
    if 'CustomerID' in df.columns:
        df = df.drop('CustomerID', axis=1)
    
    # Handle missing values
    print(f"Missing values before: {df.isnull().sum().sum()}")
    df = df.dropna()
    print(f"Shape after dropping NaN: {df.shape}")
    
    # Separate features and target
    X = df.drop('Churn', axis=1)
    y = df['Churn'].astype(int)
    
    print(f"\nChurn distribution:")
    print(y.value_counts())
    print(f"Churn rate: {y.mean()*100:.2f}%")
    
    return X, y

def encode_features(X_train, X_test):
    """Encode categorical features."""
    print("\nEncoding features...")
    
    # Identify categorical columns
    categorical_cols = X_train.select_dtypes(include=['object']).columns.tolist()
    numerical_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    
    print(f"Categorical columns: {categorical_cols}")
    print(f"Numerical columns: {numerical_cols}")
    
    # Encode categorical variables
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        X_train[col] = le.fit_transform(X_train[col].astype(str))
        X_test[col] = le.transform(X_test[col].astype(str))
        label_encoders[col] = le
    
    # Scale numerical features
    scaler = StandardScaler()
    X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
    X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])
    
    return X_train, X_test, label_encoders, scaler

def train_models(X_train, X_test, y_train, y_test):
    """Train multiple models and select the best."""
    print("\n" + "="*80)
    print("Training Models")
    print("="*80)
    
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=20, random_state=42, n_jobs=-1),
        'XGBoost': xgb.XGBClassifier(n_estimators=100, max_depth=10, learning_rate=0.1, random_state=42, n_jobs=-1, verbosity=0)
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else y_pred
        
        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        results[name] = {
            'model': model,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc_auc': roc_auc
        }
        
        print(f"  Accuracy:  {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1 Score:  {f1:.4f}")
        print(f"  ROC AUC:   {roc_auc:.4f}")
    
    # Select best model based on ROC AUC
    best_model_name = max(results, key=lambda x: results[x]['roc_auc'])
    best_model = results[best_model_name]['model']
    
    print("\n" + "="*80)
    print(f"Best Model: {best_model_name}")
    print(f"ROC AUC: {results[best_model_name]['roc_auc']:.4f}")
    print("="*80)
    
    return best_model, best_model_name, results

def save_model(model, label_encoders, scaler, model_name, results):
    """Save model and preprocessing objects."""
    print("\nSaving model and preprocessors...")
    
    # Create directories
    os.makedirs('models', exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f'results/{timestamp}'
    os.makedirs(results_dir, exist_ok=True)
    
    # Save to models directory
    joblib.dump(model, 'models/best_model.pkl')
    joblib.dump({'label_encoders': label_encoders, 'scaler': scaler}, 'models/preprocessor.pkl')
    
    # Save to results directory
    joblib.dump(model, f'{results_dir}/best_model.pkl')
    joblib.dump({'label_encoders': label_encoders, 'scaler': scaler}, f'{results_dir}/preprocessor.pkl')
    
    # Save results
    results_df = pd.DataFrame(results).T
    results_df.drop('model', axis=1).to_csv(f'{results_dir}/model_comparison.csv')
    
    print(f"✓ Model saved to: models/best_model.pkl")
    print(f"✓ Preprocessor saved to: models/preprocessor.pkl")
    print(f"✓ Results saved to: {results_dir}/")

def main():
    print("="*80)
    print("Customer Churn Prediction - Large Dataset Training")
    print("="*80)
    
    # Load data
    X, y = load_and_preprocess_data('data/customer_churn_large.csv')
    
    # Split data
    print("\nSplitting data (80% train, 20% test)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    
    # Encode features
    X_train, X_test, label_encoders, scaler = encode_features(X_train.copy(), X_test.copy())
    
    # Train models
    best_model, model_name, results = train_models(X_train, X_test, y_train, y_test)
    
    # Save model
    save_model(best_model, label_encoders, scaler, model_name, results)
    
    print("\n" + "="*80)
    print("✓ Training Complete!")
    print("="*80)
    print("\nYou can now use the model:")
    print("  - Run: streamlit run app.py")
    print("  - Or use: python predict.py")
    print("="*80)

if __name__ == "__main__":
    main()
