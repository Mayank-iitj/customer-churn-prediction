# Customer Churn Prediction System

A comprehensive machine learning solution for predicting customer churn, enabling businesses to implement proactive retention strategies.

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0%2B-orange)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.0%2B-red)](https://streamlit.io/)

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Model Performance](#model-performance)
- [Technologies](#technologies)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

This project provides a complete end-to-end solution for customer churn prediction, from data exploration to model deployment. It analyzes customer behavior data to accurately predict which customers are likely to stop using a company's service, enabling data-driven retention strategies.

## ✨ Features

### Data Processing & Analysis
- **Comprehensive EDA**: Interactive Jupyter notebook for data exploration
- **Advanced Preprocessing**: Automated handling of missing values, feature encoding, and scaling
- **Feature Engineering**: Intelligent creation of derived features
- **Class Imbalance Handling**: SMOTE implementation for balanced training

### Model Training & Evaluation
- **Multiple Algorithms**: Logistic Regression, Random Forest, Gradient Boosting, XGBoost, SVM, and more
- **Hyperparameter Tuning**: Grid Search and Randomized Search with cross-validation
- **Comprehensive Metrics**: Accuracy, Precision, Recall, F1-Score, ROC-AUC
- **Visualization**: ROC curves, confusion matrices, feature importance plots

### Interpretability
- **Feature Importance**: Tree-based and permutation importance
- **SHAP Values**: Advanced model interpretability (optional)
- **Actionable Insights**: Clear recommendations based on predictions

### Deployment
- **Streamlit Web App**: Interactive interface for predictions
- **Batch Processing**: Upload CSV files for bulk predictions
- **Model Persistence**: Save and load trained models
- **Real-time Predictions**: Instant churn risk assessment

## 📁 Project Structure

```
customer-churn-prediction/
│
├── data/                          # Data directory
│   └── customer_churn.csv        # Your dataset (add your own)
│
├── notebooks/                     # Jupyter notebooks
│   └── exploratory_analysis.ipynb # EDA notebook
│
├── src/                          # Source code modules
│   ├── __init__.py
│   ├── config.py                 # Configuration settings
│   ├── preprocessing.py          # Data preprocessing
│   ├── model_training.py         # Model training pipeline
│   └── evaluation.py             # Model evaluation utilities
│
├── models/                       # Saved models
│   └── (generated after training)
│
├── logs/                         # Log files
│   └── (generated during execution)
│
├── results/                      # Training results
│   └── (generated after training)
│
├── main.py                       # Main pipeline script
├── app.py                        # Streamlit web application
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. **Clone the repository** (or download the project):
```bash
cd customer-churn-prediction
```

2. **Create a virtual environment** (recommended):
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

## 📊 Usage

### 1. Prepare Your Data

Place your customer churn dataset in the `data/` directory as `customer_churn.csv`. The dataset should include:
- Customer demographic information
- Account details (tenure, contract type, etc.)
- Service usage data
- Churn label (target variable)

**Example Dataset**: The [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) dataset from Kaggle works perfectly with this project.

### 2. Exploratory Data Analysis

Open the Jupyter notebook for comprehensive data exploration:

```bash
jupyter notebook notebooks/exploratory_analysis.ipynb
```

This notebook provides:
- Data quality assessment
- Distribution analysis
- Correlation studies
- Feature relationships with churn
- Key insights and recommendations

### 3. Train Models

Run the complete training pipeline:

```bash
python main.py
```

This will:
- Load and preprocess the data
- Train multiple ML models
- Perform hyperparameter tuning
- Evaluate model performance
- Save the best model and results

**Configuration**: Edit `src/config.py` to customize:
- Data paths
- Preprocessing methods
- Model parameters
- Evaluation settings

### 4. Launch Web Application

Start the interactive Streamlit app:

```bash
streamlit run app.py
```

The app provides:
- **Single Prediction**: Enter customer details for instant churn prediction
- **Batch Prediction**: Upload CSV files for bulk predictions
- **Model Info**: View model details and feature importance

Access the app at: `http://localhost:8501`

### 5. Deploy to Streamlit Cloud

For easy cloud deployment:

```bash
# Test Streamlit compatibility first
python test_streamlit.py

# Deploy to Streamlit Cloud (see STREAMLIT_DEPLOYMENT.md)
```

See **[STREAMLIT_DEPLOYMENT.md](STREAMLIT_DEPLOYMENT.md)** for detailed cloud deployment instructions.

## 📈 Model Performance

The system trains and compares multiple models:

| Model | Typical ROC-AUC | Typical Accuracy |
|-------|----------------|------------------|
| Logistic Regression | 0.75-0.80 | 75-80% |
| Random Forest | 0.82-0.88 | 80-85% |
| Gradient Boosting | 0.83-0.89 | 81-86% |
| XGBoost | 0.84-0.90 | 82-87% |

*Note: Actual performance depends on your dataset*

### Key Metrics Tracked
- **ROC-AUC**: Area under the ROC curve
- **Precision**: Accuracy of positive predictions
- **Recall**: Coverage of actual churners
- **F1-Score**: Harmonic mean of precision and recall

## 🛠 Technologies

### Core ML & Data Science
- **pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing
- **scikit-learn**: Machine learning algorithms and utilities
- **XGBoost**: Gradient boosting framework
- **imbalanced-learn**: SMOTE for class imbalance

### Visualization
- **Matplotlib**: Static visualizations
- **Seaborn**: Statistical data visualization

### Model Interpretability
- **SHAP**: SHapley Additive exPlanations (optional)

### Deployment
- **Streamlit**: Interactive web application
- **joblib**: Model serialization

## 📝 Configuration

Key configuration parameters in `src/config.py`:

```python
# Data paths
DATA_PATH = 'data/customer_churn.csv'

# Preprocessing
SCALING_METHOD = 'standard'  # or 'minmax'
ENCODING_METHOD = 'onehot'   # or 'label'
HANDLE_IMBALANCE = False     # Enable SMOTE

# Training
TUNE_HYPERPARAMETERS = True  # Enable tuning
SEARCH_METHOD = 'grid'       # or 'random'
CV_FOLDS = 5                 # Cross-validation folds

# Evaluation
CALCULATE_SHAP = True        # SHAP values (requires shap library)
```

## 🔍 Sample Workflow

1. **Data Exploration**:
   ```bash
   jupyter notebook notebooks/exploratory_analysis.ipynb
   ```

2. **Model Training**:
   ```bash
   python main.py
   ```

3. **Deploy Application**:
   ```bash
   streamlit run app.py
   ```

4. **Make Predictions**:
   - Open browser at `http://localhost:8501`
   - Enter customer details or upload CSV
   - Get instant churn predictions and recommendations

## 📊 Example Output

After training, you'll find in `results/[timestamp]/`:
- `best_model.pkl` - Trained model
- `preprocessor.pkl` - Fitted preprocessor
- `confusion_matrix.png` - Confusion matrix visualization
- `roc_curve.png` - ROC curve plot
- `feature_importance.png` - Feature importance chart
- `model_comparison.csv` - Performance comparison

## 🤝 Contributing

Contributions are welcome! Areas for enhancement:
- Additional ML algorithms
- Advanced feature engineering
- Real-time data pipeline integration
- A/B testing framework
- API deployment (Flask/FastAPI)
- Docker containerization

## 🚀 Deployment

This project is **deployment-ready** with full containerization support and comprehensive deployment documentation.

### Quick Deploy Options

#### Option 1: Docker (Recommended)
```bash
# Build and run with Docker Compose
docker-compose up -d

# Access at http://localhost:8501
```

#### Option 2: Local Python
```bash
# Use the provided script
.\run.ps1  # Windows
./run.sh   # Linux/Mac
```

#### Option 3: Cloud Platforms
- **AWS**: Deploy to ECS, Fargate, or EC2
- **Google Cloud**: Deploy to Cloud Run or GKE
- **Azure**: Deploy to Container Instances or AKS
- **Heroku**: One-click deployment
- **Kubernetes**: Use provided manifests

### Deployment Documentation

- **[DEPLOYMENT.md](DEPLOYMENT.md)**: Comprehensive deployment guide for all platforms
- **[DEPLOYMENT_CHECKLIST.md](DEPLOYMENT_CHECKLIST.md)**: Pre-deployment checklist
- **[CONTRIBUTING.md](CONTRIBUTING.md)**: Contribution guidelines

### Key Features for Production

✅ Docker containerization with multi-stage builds  
✅ Environment-based configuration  
✅ Health check endpoints  
✅ CI/CD with GitHub Actions  
✅ Kubernetes manifests with auto-scaling  
✅ Production-optimized dependencies  
✅ Comprehensive logging and monitoring  
✅ Security best practices  

### Environment Configuration

1. Copy the example environment file:
   ```bash
   cp .env.example .env
   ```

2. Edit `.env` with your configuration

3. For production, use `.env.production` as a template

See [DEPLOYMENT.md](DEPLOYMENT.md) for detailed configuration options.

## 📄 License

This project is open-source and available under the MIT License.

## 🙏 Acknowledgments

- Telco Customer Churn dataset (Kaggle)
- scikit-learn documentation
- Streamlit community

## 📧 Contact

For questions, issues, or suggestions, please open an issue in the repository.

## 📚 Additional Resources

- [Changelog](CHANGELOG.md) - Version history and changes
- [Contributing Guide](CONTRIBUTING.md) - How to contribute
- [Deployment Guide](DEPLOYMENT.md) - Detailed deployment instructions
- [Deployment Checklist](DEPLOYMENT_CHECKLIST.md) - Pre-deployment verification

---

**Version**: 1.0.0 - Deployment Ready  
**Status**: Production Ready ✅  
**Last Updated**: November 2025

**Note**: This project is production-ready and can be deployed to any major cloud platform. See the deployment documentation for platform-specific instructions.

Happy Predicting! 🚀
