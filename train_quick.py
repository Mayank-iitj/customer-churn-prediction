"""
Quick Training Script for Large Dataset
Uses a single efficient model
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
import os
from datetime import datetime

print("="*80)
print("Customer Churn Prediction - Quick Training")
print("="*80)

# Load data
print("\nLoading data...")
df = pd.read_csv('data/customer_churn_large.csv')
print(f"Original shape: {df.shape}")

# Drop CustomerID and handle missing values
if 'CustomerID' in df.columns:
    df = df.drop('CustomerID', axis=1)
df = df.dropna()
print(f"Clean shape: {df.shape}")

# Separate features and target
X = df.drop('Churn', axis=1)
y = df['Churn'].astype(int)

print(f"\nChurn distribution: {y.value_counts().to_dict()}")
print(f"Churn rate: {y.mean()*100:.2f}%")

# Split data
print("\nSplitting data...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"Train: {X_train.shape}, Test: {X_test.shape}")

# Encode features
print("\nEncoding features...")
categorical_cols = X_train.select_dtypes(include=['object']).columns.tolist()
numerical_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()

label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    X_train[col] = le.fit_transform(X_train[col].astype(str))
    X_test[col] = le.transform(X_test[col].astype(str))
    label_encoders[col] = le

scaler = StandardScaler()
X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])

# Train model
print("\nTraining Random Forest (this may take a few minutes)...")
model = RandomForestClassifier(
    n_estimators=50,  # Reduced for speed
    max_depth=15,
    min_samples_split=10,
    min_samples_leaf=4,
    random_state=42,
    n_jobs=-1,
    verbose=1
)
model.fit(X_train, y_train)

# Evaluate
print("\nEvaluating...")
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)

print("\n" + "="*80)
print("Model Performance")
print("="*80)
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")
print(f"ROC AUC:   {roc_auc:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save model
print("\nSaving model...")
os.makedirs('models', exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
results_dir = f'results/{timestamp}'
os.makedirs(results_dir, exist_ok=True)

joblib.dump(model, 'models/best_model.pkl')
joblib.dump({'label_encoders': label_encoders, 'scaler': scaler}, 'models/preprocessor.pkl')
joblib.dump(model, f'{results_dir}/best_model.pkl')
joblib.dump({'label_encoders': label_encoders, 'scaler': scaler}, f'{results_dir}/preprocessor.pkl')

# Save metrics
metrics_df = pd.DataFrame({
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC'],
    'Value': [accuracy, precision, recall, f1, roc_auc]
})
metrics_df.to_csv(f'{results_dir}/metrics.csv', index=False)

print(f"\n✓ Model saved to: models/best_model.pkl")
print(f"✓ Results saved to: {results_dir}/")
print("\n" + "="*80)
print("✓ Training Complete!")
print("="*80)
