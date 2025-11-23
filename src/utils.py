"""
Utility functions for the Customer Churn Prediction project.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
import logging

logger = logging.getLogger(__name__)


def create_sample_dataset(n_samples=1000, output_path='data/sample_customer_churn.csv'):
    """
    Create a sample customer churn dataset for testing.
    
    Parameters:
    -----------
    n_samples : int
        Number of samples to generate
    output_path : str
        Path to save the dataset
        
    Returns:
    --------
    pandas.DataFrame : Generated dataset
    """
    np.random.seed(42)
    
    # Demographics
    gender = np.random.choice(['Male', 'Female'], n_samples)
    senior_citizen = np.random.choice([0, 1], n_samples, p=[0.85, 0.15])
    partner = np.random.choice(['Yes', 'No'], n_samples, p=[0.5, 0.5])
    dependents = np.random.choice(['Yes', 'No'], n_samples, p=[0.3, 0.7])
    
    # Account information
    tenure = np.random.randint(1, 73, n_samples)
    contract = np.random.choice(['Month-to-month', 'One year', 'Two year'], 
                               n_samples, p=[0.55, 0.25, 0.20])
    payment_method = np.random.choice([
        'Electronic check', 'Mailed check', 
        'Bank transfer (automatic)', 'Credit card (automatic)'
    ], n_samples, p=[0.35, 0.15, 0.25, 0.25])
    paperless_billing = np.random.choice(['Yes', 'No'], n_samples, p=[0.6, 0.4])
    
    # Services
    phone_service = np.random.choice(['Yes', 'No'], n_samples, p=[0.9, 0.1])
    multiple_lines = np.random.choice(['No', 'Yes', 'No phone service'], 
                                     n_samples, p=[0.5, 0.4, 0.1])
    internet_service = np.random.choice(['DSL', 'Fiber optic', 'No'], 
                                       n_samples, p=[0.3, 0.5, 0.2])
    online_security = np.random.choice(['No', 'Yes', 'No internet service'], 
                                      n_samples, p=[0.5, 0.3, 0.2])
    online_backup = np.random.choice(['No', 'Yes', 'No internet service'], 
                                    n_samples, p=[0.5, 0.3, 0.2])
    device_protection = np.random.choice(['No', 'Yes', 'No internet service'], 
                                        n_samples, p=[0.5, 0.3, 0.2])
    tech_support = np.random.choice(['No', 'Yes', 'No internet service'], 
                                   n_samples, p=[0.5, 0.3, 0.2])
    streaming_tv = np.random.choice(['No', 'Yes', 'No internet service'], 
                                   n_samples, p=[0.4, 0.4, 0.2])
    streaming_movies = np.random.choice(['No', 'Yes', 'No internet service'], 
                                       n_samples, p=[0.4, 0.4, 0.2])
    
    # Charges
    monthly_charges = np.random.uniform(18.0, 120.0, n_samples)
    total_charges = monthly_charges * tenure + np.random.normal(0, 100, n_samples)
    total_charges = np.maximum(total_charges, 0)  # Ensure non-negative
    
    # Churn (with some logic)
    churn_prob = 0.1  # Base probability
    
    # Increase churn probability based on factors
    churn_prob_array = np.full(n_samples, churn_prob)
    churn_prob_array[contract == 'Month-to-month'] += 0.2
    churn_prob_array[tenure < 12] += 0.15
    churn_prob_array[payment_method == 'Electronic check'] += 0.1
    churn_prob_array[monthly_charges > 80] += 0.1
    churn_prob_array[internet_service == 'Fiber optic'] += 0.05
    
    # Decrease churn probability
    churn_prob_array[contract == 'Two year'] -= 0.15
    churn_prob_array[tenure > 36] -= 0.1
    
    # Clip probabilities
    churn_prob_array = np.clip(churn_prob_array, 0, 1)
    
    # Generate churn labels
    churn = (np.random.random(n_samples) < churn_prob_array).astype(int)
    
    # Create DataFrame
    df = pd.DataFrame({
        'gender': gender,
        'SeniorCitizen': senior_citizen,
        'Partner': partner,
        'Dependents': dependents,
        'tenure': tenure,
        'PhoneService': phone_service,
        'MultipleLines': multiple_lines,
        'InternetService': internet_service,
        'OnlineSecurity': online_security,
        'OnlineBackup': online_backup,
        'DeviceProtection': device_protection,
        'TechSupport': tech_support,
        'StreamingTV': streaming_tv,
        'StreamingMovies': streaming_movies,
        'Contract': contract,
        'PaperlessBilling': paperless_billing,
        'PaymentMethod': payment_method,
        'MonthlyCharges': np.round(monthly_charges, 2),
        'TotalCharges': np.round(total_charges, 2),
        'Churn': churn
    })
    
    # Save to file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    
    logger.info(f"Sample dataset created with {n_samples} samples")
    logger.info(f"Churn rate: {churn.mean()*100:.2f}%")
    logger.info(f"Saved to: {output_path}")
    
    return df


def plot_model_comparison(results_dict, save_path=None):
    """
    Create a comprehensive model comparison plot.
    
    Parameters:
    -----------
    results_dict : dict
        Dictionary with model names as keys and metrics as values
    save_path : str, optional
        Path to save the plot
    """
    df = pd.DataFrame(results_dict).T
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    titles = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    
    for idx, (metric, title) in enumerate(zip(metrics, titles)):
        row = idx // 2
        col = idx % 2
        
        if metric in df.columns:
            ax = axes[row, col]
            df[metric].sort_values(ascending=True).plot(kind='barh', ax=ax, color='skyblue')
            ax.set_title(f'{title} Comparison', fontsize=14, fontweight='bold')
            ax.set_xlabel('Score', fontsize=12)
            ax.set_xlim([0, 1])
            ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Comparison plot saved to {save_path}")
    
    plt.show()


def generate_report(model_name, metrics, output_path='report.txt'):
    """
    Generate a text report of model performance.
    
    Parameters:
    -----------
    model_name : str
        Name of the model
    metrics : dict
        Dictionary of metrics
    output_path : str
        Path to save the report
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    report = f"""
{'='*80}
CUSTOMER CHURN PREDICTION - MODEL PERFORMANCE REPORT
{'='*80}

Report Generated: {timestamp}
Model: {model_name}

{'='*80}
PERFORMANCE METRICS
{'='*80}

Accuracy:  {metrics.get('accuracy', 'N/A'):.4f}
Precision: {metrics.get('precision', 'N/A'):.4f}
Recall:    {metrics.get('recall', 'N/A'):.4f}
F1-Score:  {metrics.get('f1_score', 'N/A'):.4f}
ROC-AUC:   {metrics.get('roc_auc', 'N/A'):.4f}

{'='*80}
INTERPRETATION
{'='*80}

- Accuracy: Overall correctness of predictions
- Precision: Of all predicted churners, what % actually churned
- Recall: Of all actual churners, what % did we identify
- F1-Score: Harmonic mean of precision and recall
- ROC-AUC: Area under the ROC curve (discrimination ability)

{'='*80}
"""
    
    with open(output_path, 'w') as f:
        f.write(report)
    
    logger.info(f"Report saved to {output_path}")
    
    return report


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Generate sample dataset
    df = create_sample_dataset(n_samples=1000)
    print("\nSample dataset preview:")
    print(df.head())
    print(f"\nDataset shape: {df.shape}")
    print(f"Churn distribution:\n{df['Churn'].value_counts()}")
