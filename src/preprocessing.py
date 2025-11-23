"""
Data Preprocessing Module for Customer Churn Prediction

This module handles all data preprocessing tasks including:
- Missing value handling
- Feature encoding
- Feature scaling
- Feature engineering
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
import logging

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """
    Comprehensive data preprocessing class for customer churn prediction.
    """
    
    def __init__(self, scaling_method='standard'):
        """
        Initialize the preprocessor.
        
        Parameters:
        -----------
        scaling_method : str, default='standard'
            Method for scaling numerical features ('standard' or 'minmax')
        """
        self.scaling_method = scaling_method
        self.scaler = StandardScaler() if scaling_method == 'standard' else MinMaxScaler()
        self.label_encoders = {}
        self.numerical_imputer = SimpleImputer(strategy='median')
        self.categorical_imputer = SimpleImputer(strategy='most_frequent')
        self.numerical_features = []
        self.categorical_features = []
        
    def identify_feature_types(self, df, target_column='Churn'):
        """
        Identify numerical and categorical features in the dataset.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Input dataframe
        target_column : str
            Name of the target column
            
        Returns:
        --------
        tuple : (numerical_features, categorical_features)
        """
        # Exclude target column
        features = [col for col in df.columns if col != target_column]
        
        self.numerical_features = df[features].select_dtypes(
            include=['int64', 'float64']
        ).columns.tolist()
        
        self.categorical_features = df[features].select_dtypes(
            include=['object', 'category']
        ).columns.tolist()
        
        logger.info(f"Identified {len(self.numerical_features)} numerical features")
        logger.info(f"Identified {len(self.categorical_features)} categorical features")
        
        return self.numerical_features, self.categorical_features
    
    def handle_missing_values(self, df, strategy='impute'):
        """
        Handle missing values in the dataset.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Input dataframe
        strategy : str, default='impute'
            Strategy for handling missing values ('impute' or 'drop')
            
        Returns:
        --------
        pandas.DataFrame : DataFrame with missing values handled
        """
        logger.info(f"Missing values before handling:\n{df.isnull().sum()}")
        
        if strategy == 'drop':
            df_clean = df.dropna()
            logger.info(f"Dropped {len(df) - len(df_clean)} rows with missing values")
        else:
            df_clean = df.copy()
            
            # Impute numerical features
            if self.numerical_features:
                df_clean[self.numerical_features] = self.numerical_imputer.fit_transform(
                    df_clean[self.numerical_features]
                )
            
            # Impute categorical features
            if self.categorical_features:
                df_clean[self.categorical_features] = self.categorical_imputer.fit_transform(
                    df_clean[self.categorical_features]
                )
            
            logger.info("Missing values imputed successfully")
        
        return df_clean
    
    def encode_categorical_features(self, df, encoding_method='onehot'):
        """
        Encode categorical features.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Input dataframe
        encoding_method : str, default='onehot'
            Encoding method ('onehot' or 'label')
            
        Returns:
        --------
        pandas.DataFrame : DataFrame with encoded categorical features
        """
        df_encoded = df.copy()
        
        if encoding_method == 'label':
            # Label encoding for categorical features
            for col in self.categorical_features:
                le = LabelEncoder()
                df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
                self.label_encoders[col] = le
            logger.info(f"Label encoded {len(self.categorical_features)} features")
            
        elif encoding_method == 'onehot':
            # One-hot encoding
            df_encoded = pd.get_dummies(
                df_encoded, 
                columns=self.categorical_features,
                drop_first=True,
                prefix=self.categorical_features
            )
            logger.info(f"One-hot encoded {len(self.categorical_features)} features")
        
        return df_encoded
    
    def scale_numerical_features(self, df, fit=True):
        """
        Scale numerical features.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Input dataframe
        fit : bool, default=True
            Whether to fit the scaler (True for training, False for test data)
            
        Returns:
        --------
        pandas.DataFrame : DataFrame with scaled numerical features
        """
        df_scaled = df.copy()
        
        if self.numerical_features:
            if fit:
                df_scaled[self.numerical_features] = self.scaler.fit_transform(
                    df_scaled[self.numerical_features]
                )
            else:
                df_scaled[self.numerical_features] = self.scaler.transform(
                    df_scaled[self.numerical_features]
                )
            
            logger.info(f"Scaled {len(self.numerical_features)} numerical features using {self.scaling_method}")
        
        return df_scaled
    
    def engineer_features(self, df):
        """
        Create new features through feature engineering.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Input dataframe
            
        Returns:
        --------
        pandas.DataFrame : DataFrame with engineered features
        """
        df_engineered = df.copy()
        
        # Example feature engineering (adjust based on your actual dataset)
        try:
            # Calculate customer lifetime value if relevant columns exist
            if 'MonthlyCharges' in df.columns and 'tenure' in df.columns:
                df_engineered['TotalCharges_Calculated'] = (
                    df_engineered['MonthlyCharges'] * df_engineered['tenure']
                )
                logger.info("Created feature: TotalCharges_Calculated")
            
            # Create tenure groups
            if 'tenure' in df.columns:
                df_engineered['TenureGroup'] = pd.cut(
                    df_engineered['tenure'],
                    bins=[0, 12, 24, 36, 48, 60, np.inf],
                    labels=['0-12', '12-24', '24-36', '36-48', '48-60', '60+']
                )
                logger.info("Created feature: TenureGroup")
            
            # Average monthly spend ratio
            if 'TotalCharges' in df.columns and 'MonthlyCharges' in df.columns:
                df_engineered['AvgMonthlySpendRatio'] = (
                    df_engineered['TotalCharges'] / 
                    (df_engineered['MonthlyCharges'] + 1)  # Avoid division by zero
                )
                logger.info("Created feature: AvgMonthlySpendRatio")
            
            # Contract commitment indicator
            if 'Contract' in df.columns:
                df_engineered['HasLongTermContract'] = (
                    df_engineered['Contract'].isin(['One year', 'Two year'])
                ).astype(int)
                logger.info("Created feature: HasLongTermContract")
            
        except Exception as e:
            logger.warning(f"Feature engineering encountered an issue: {e}")
        
        return df_engineered
    
    def handle_class_imbalance(self, X, y, method='smote', random_state=42):
        """
        Handle class imbalance in the dataset.
        
        Parameters:
        -----------
        X : pandas.DataFrame or numpy.ndarray
            Features
        y : pandas.Series or numpy.ndarray
            Target variable
        method : str, default='smote'
            Method for handling imbalance ('smote' or 'class_weight')
        random_state : int
            Random state for reproducibility
            
        Returns:
        --------
        tuple : (X_resampled, y_resampled)
        """
        if method == 'smote':
            smote = SMOTE(random_state=random_state)
            X_resampled, y_resampled = smote.fit_resample(X, y)
            logger.info(f"Applied SMOTE: Original shape {X.shape}, Resampled shape {X_resampled.shape}")
            return X_resampled, y_resampled
        else:
            logger.info("Class imbalance will be handled using class_weight in model training")
            return X, y
    
    def preprocess_pipeline(self, df, target_column='Churn', 
                           encoding_method='onehot', 
                           handle_imbalance=False,
                           fit=True):
        """
        Complete preprocessing pipeline.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Input dataframe
        target_column : str
            Name of the target column
        encoding_method : str
            Encoding method for categorical features
        handle_imbalance : bool
            Whether to handle class imbalance
        fit : bool
            Whether to fit transformers
            
        Returns:
        --------
        tuple : (X, y) - Preprocessed features and target
        """
        logger.info("Starting preprocessing pipeline...")
        
        # Identify feature types
        if fit:
            self.identify_feature_types(df, target_column)
        
        # Engineer features first
        df = self.engineer_features(df)
        
        # Update feature types after engineering
        if fit:
            self.identify_feature_types(df, target_column)
        
        # Handle missing values
        df = self.handle_missing_values(df)
        
        # Encode categorical features
        df = self.encode_categorical_features(df, encoding_method)
        
        # Separate features and target
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        # Encode target if it's categorical
        if y.dtype == 'object':
            le_target = LabelEncoder()
            y = le_target.fit_transform(y)
            self.label_encoders['target'] = le_target
            logger.info("Encoded target variable")
        
        # Update numerical features after encoding
        self.numerical_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        
        # Scale numerical features
        X = self.scale_numerical_features(X, fit=fit)
        
        # Handle class imbalance if requested
        if handle_imbalance and fit:
            X, y = self.handle_class_imbalance(X, y)
        
        logger.info(f"Preprocessing complete. Final shape: X={X.shape}, y={len(y)}")
        
        return X, y


def load_data(filepath, **kwargs):
    """
    Load data from various file formats.
    
    Parameters:
    -----------
    filepath : str
        Path to the data file
    **kwargs : dict
        Additional arguments for pandas read functions
        
    Returns:
    --------
    pandas.DataFrame : Loaded data
    """
    try:
        if filepath.endswith('.csv'):
            df = pd.read_csv(filepath, **kwargs)
        elif filepath.endswith('.xlsx') or filepath.endswith('.xls'):
            df = pd.read_excel(filepath, **kwargs)
        elif filepath.endswith('.json'):
            df = pd.read_json(filepath, **kwargs)
        else:
            raise ValueError(f"Unsupported file format: {filepath}")
        
        logger.info(f"Successfully loaded data from {filepath}")
        logger.info(f"Dataset shape: {df.shape}")
        
        return df
    
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    print("Data Preprocessing Module - Ready for use")
