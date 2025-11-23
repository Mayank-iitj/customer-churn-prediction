# Data Directory

Place your customer churn dataset in this directory.

## Expected Format

The dataset should be a CSV file named `customer_churn.csv` (or update the path in `src/config.py`).

## Sample Datasets

You can use:

1. **Telco Customer Churn Dataset** (Recommended)
   - Source: https://www.kaggle.com/datasets/blastchar/telco-customer-churn
   - Download and place here as `customer_churn.csv`

2. **Generate Sample Data**
   - Run: `python -c "from src.utils import create_sample_dataset; create_sample_dataset()"`
   - This will create `sample_customer_churn.csv` in this directory

## Dataset Requirements

Your CSV should include:
- A target column indicating churn (binary: 0/1 or categorical: Yes/No)
- Customer features such as:
  - Demographics (age, gender, etc.)
  - Account information (tenure, contract type, payment method)
  - Service usage (internet, phone, streaming services)
  - Financial data (monthly charges, total charges)

## Example Structure

```csv
CustomerID,gender,SeniorCitizen,Partner,Dependents,tenure,PhoneService,Contract,MonthlyCharges,TotalCharges,Churn
7590-VHVEG,Female,0,Yes,No,1,No,Month-to-month,29.85,29.85,No
5575-GNVDE,Male,0,No,No,34,Yes,One year,56.95,1889.5,No
...
```

## Note

- Add your data files to `.gitignore` to avoid committing sensitive data
- Ensure proper data privacy and compliance with relevant regulations
