"""
Configuration Module for Customer Churn Prediction

This module contains all configuration parameters for the project.
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Config:
    """Configuration class for customer churn prediction pipeline."""
    
    # Data paths
    DATA_PATH = os.getenv('DATA_PATH', os.path.join('data', 'customer_churn.csv'))
    
    # Random state for reproducibility
    RANDOM_STATE = int(os.getenv('RANDOM_STATE', '42'))
    
    # Data preprocessing
    TARGET_COLUMN = os.getenv('TARGET_COLUMN', 'Churn')
    SCALING_METHOD = os.getenv('SCALING_METHOD', 'standard')  # 'standard' or 'minmax'
    ENCODING_METHOD = os.getenv('ENCODING_METHOD', 'onehot')   # 'onehot' or 'label'
    HANDLE_IMBALANCE = os.getenv('HANDLE_IMBALANCE', 'false').lower() == 'true'  # Whether to use SMOTE for class imbalance
    
    # Data splitting
    TEST_SIZE = float(os.getenv('TEST_SIZE', '0.2'))     # Proportion of data for test set
    VAL_SIZE = float(os.getenv('VAL_SIZE', '0.1'))      # Proportion of training data for validation
    
    # Model training
    TUNE_HYPERPARAMETERS = os.getenv('TUNE_HYPERPARAMETERS', 'true').lower() == 'true'  # Whether to perform hyperparameter tuning
    SEARCH_METHOD = os.getenv('SEARCH_METHOD', 'grid')       # 'grid' or 'random'
    CV_FOLDS = int(os.getenv('CV_FOLDS', '5'))                 # Number of cross-validation folds
    
    # Model evaluation
    CALCULATE_PERMUTATION_IMPORTANCE = os.getenv('CALCULATE_PERMUTATION_IMPORTANCE', 'true').lower() == 'true'
    CALCULATE_SHAP = os.getenv('CALCULATE_SHAP', 'true').lower() == 'true'
    SHAP_SAMPLE_SIZE = int(os.getenv('SHAP_SAMPLE_SIZE', '100'))
    
    # Deployment
    MODEL_SAVE_PATH = os.getenv('MODEL_SAVE_PATH', os.path.join('models', 'best_model.pkl'))
    PREPROCESSOR_SAVE_PATH = os.getenv('PREPROCESSOR_SAVE_PATH', os.path.join('models', 'preprocessor.pkl'))
    
    def __init__(self):
        """Initialize configuration and create necessary directories."""
        self.create_directories()
    
    def create_directories(self):
        """Create necessary directories if they don't exist."""
        directories = ['data', 'models', 'logs', 'results', 'notebooks']
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    def update(self, **kwargs):
        """
        Update configuration parameters.
        
        Parameters:
        -----------
        **kwargs : dict
            Configuration parameters to update
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Invalid configuration parameter: {key}")
    
    def __repr__(self):
        """String representation of configuration."""
        config_dict = {
            key: value for key, value in self.__dict__.items()
            if not key.startswith('_')
        }
        return f"Config({config_dict})"


# Create a default configuration instance
default_config = Config()


if __name__ == "__main__":
    config = Config()
    print("Configuration:")
    print(config)
