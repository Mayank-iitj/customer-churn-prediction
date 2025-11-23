# Quick Start Guide - Customer Churn Prediction

## 🚀 Getting Started in 5 Minutes

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Get Sample Data (Optional)
If you don't have your own dataset, generate a sample one:

```python
python -c "from src.utils import create_sample_dataset; create_sample_dataset()"
```

Or download the Telco Customer Churn dataset from Kaggle:
https://www.kaggle.com/datasets/blastchar/telco-customer-churn

Place it in the `data/` folder as `customer_churn.csv`.

### Step 3: Update Configuration
Edit `src/config.py` to match your dataset:
- Update `DATA_PATH` if your file has a different name
- Adjust `TARGET_COLUMN` if your target variable has a different name

### Step 4: Run the Pipeline
```bash
python main.py
```

This will:
- ✅ Load and preprocess your data
- ✅ Train multiple ML models
- ✅ Evaluate and compare models
- ✅ Save the best model
- ✅ Generate visualizations and reports

### Step 5: Launch the Web App
```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501` to:
- Make predictions for individual customers
- Upload CSV files for batch predictions
- View model information and feature importance

## 📊 Sample Workflow

### Exploratory Data Analysis
```bash
jupyter notebook notebooks/exploratory_analysis.ipynb
```

### Train Custom Model
```python
from src.preprocessing import DataPreprocessor, load_data
from src.model_training import ChurnModelTrainer, split_data
from src.evaluation import ModelEvaluator

# Load data
df = load_data('data/customer_churn.csv')

# Preprocess
preprocessor = DataPreprocessor()
X, y = preprocessor.preprocess_pipeline(df, target_column='Churn')

# Split data
X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

# Train
trainer = ChurnModelTrainer()
best_name, best_model = trainer.train_with_pipeline(X_train, y_train, X_val, y_val)

# Evaluate
evaluator = ModelEvaluator()
metrics = evaluator.evaluate_model(best_model, X_test, y_test)

print(f"Best Model: {best_name}")
print(f"Metrics: {metrics}")
```

### Make Predictions
```python
import joblib
import pandas as pd

# Load model
model = joblib.load('results/[timestamp]/best_model.pkl')
preprocessor = joblib.load('results/[timestamp]/preprocessor.pkl')

# Prepare new data
new_customer = pd.DataFrame([{
    'tenure': 12,
    'MonthlyCharges': 75.0,
    'Contract': 'Month-to-month',
    # ... other features
}])

# Predict
prediction = model.predict(new_customer)
probability = model.predict_proba(new_customer)

print(f"Churn Prediction: {prediction[0]}")
print(f"Churn Probability: {probability[0][1]:.2%}")
```

## 🛠 Common Tasks

### Change Model Parameters
Edit `src/config.py`:
```python
TUNE_HYPERPARAMETERS = True  # Enable/disable tuning
SEARCH_METHOD = 'random'     # 'grid' or 'random'
CV_FOLDS = 10                # Increase for better validation
```

### Handle Class Imbalance
In `src/config.py`:
```python
HANDLE_IMBALANCE = True  # Enable SMOTE
```

### Use Different Encoding
In `src/config.py`:
```python
ENCODING_METHOD = 'label'  # Use label encoding instead of one-hot
```

### Add Custom Features
Edit the `engineer_features` method in `src/preprocessing.py`.

## 📝 Dataset Requirements

Your CSV should include:
- **Target column**: Binary (0/1) or categorical ('Yes'/'No') churn indicator
- **Numerical features**: Tenure, charges, usage metrics, etc.
- **Categorical features**: Contract type, services, demographics, etc.

Example structure:
```
CustomerID, gender, SeniorCitizen, tenure, Contract, MonthlyCharges, Churn
1, Male, 0, 12, Month-to-month, 50.0, Yes
2, Female, 1, 24, Two year, 75.5, No
...
```

## ⚠️ Troubleshooting

### "No module named 'src'"
Make sure you're running commands from the project root directory.

### "Model file not found"
Run `python main.py` first to train and save a model.

### "Data file not found"
Ensure your data file is in the `data/` folder and the path in `config.py` is correct.

### ImportError for SHAP
SHAP is optional. Set `CALCULATE_SHAP = False` in `config.py` if you don't want to install it.

## 🎯 Next Steps

1. **Explore the EDA notebook** to understand your data
2. **Run the full pipeline** to train models
3. **Deploy the Streamlit app** for interactive predictions
4. **Customize** the code for your specific needs

## 📚 Additional Resources

- [scikit-learn Documentation](https://scikit-learn.org/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)

---

Need help? Check the full README.md for detailed documentation!
