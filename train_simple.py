"""
Simple Model Training Script
Trains a basic churn prediction model without cross-validation for small datasets
"""

import sys
import os
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from preprocessing import DataPreprocessor

def main():
    print("="*80)
    print("Simple Customer Churn Model Training")
    print("="*80)
    
    # Load data
    print("\n1. Loading data...")
    df = pd.read_csv('data/customer_churn.csv')
    print(f"Dataset shape: {df.shape}")
    
    # Preprocess
    print("\n2. Preprocessing...")
    preprocessor = DataPreprocessor()
    X, y = preprocessor.preprocess_pipeline(df, target_column='Churn', fit=True)
    print(f"Features shape: {X.shape}")
    
    # Train a simple Random Forest model
    print("\n3. Training Random Forest model...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42
    )
    model.fit(X, y)
    
    # Save model and preprocessor
    print("\n4. Saving model and preprocessor...")
    os.makedirs('models', exist_ok=True)
    
    # Save with proper protocol to avoid module issues
    joblib.dump(model, 'models/best_model.pkl', protocol=4)
    joblib.dump(preprocessor, 'models/preprocessor.pkl', protocol=4)
    
    # Also save to results directory for consistency
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f'results/{timestamp}'
    os.makedirs(results_dir, exist_ok=True)
    joblib.dump(model, f'{results_dir}/best_model.pkl', protocol=4)
    joblib.dump(preprocessor, f'{results_dir}/preprocessor.pkl', protocol=4)
    
    print("\n✓ Model training complete!")
    print(f"✓ Model saved to: models/best_model.pkl")
    print(f"✓ Preprocessor saved to: models/preprocessor.pkl")
    print("\n" + "="*80)
    print("You can now run the Streamlit app: streamlit run app.py")
    print("="*80)

if __name__ == "__main__":
    main()
