"""
Model Training Module for Customer Churn Prediction

This module handles:
- Multiple classification model training
- Hyperparameter tuning
- Cross-validation
- Model persistence
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import (
    train_test_split, 
    cross_val_score, 
    GridSearchCV, 
    RandomizedSearchCV,
    StratifiedKFold
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
import joblib
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class ChurnModelTrainer:
    """
    Comprehensive model training class for customer churn prediction.
    """
    
    def __init__(self, random_state=42):
        """
        Initialize the model trainer.
        
        Parameters:
        -----------
        random_state : int
            Random state for reproducibility
        """
        self.random_state = random_state
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.cv_results = {}
        
    def initialize_models(self):
        """
        Initialize multiple classification models.
        
        Returns:
        --------
        dict : Dictionary of initialized models
        """
        self.models = {
            'Logistic Regression': LogisticRegression(
                random_state=self.random_state,
                max_iter=1000
            ),
            'Random Forest': RandomForestClassifier(
                random_state=self.random_state,
                n_jobs=-1
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                random_state=self.random_state
            ),
            'XGBoost': XGBClassifier(
                random_state=self.random_state,
                eval_metric='logloss',
                use_label_encoder=False
            ),
            'Decision Tree': DecisionTreeClassifier(
                random_state=self.random_state
            ),
            'SVM': SVC(
                random_state=self.random_state,
                probability=True
            ),
            'Naive Bayes': GaussianNB()
        }
        
        logger.info(f"Initialized {len(self.models)} models")
        return self.models
    
    def get_hyperparameter_grids(self):
        """
        Get hyperparameter grids for each model.
        
        Returns:
        --------
        dict : Dictionary of hyperparameter grids
        """
        param_grids = {
            'Logistic Regression': {
                'C': [0.001, 0.01, 0.1, 1, 10, 100],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga']
            },
            'Random Forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'class_weight': ['balanced', None]
            },
            'Gradient Boosting': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'subsample': [0.8, 1.0]
            },
            'XGBoost': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'subsample': [0.8, 1.0],
                'colsample_bytree': [0.8, 1.0]
            },
            'Decision Tree': {
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'class_weight': ['balanced', None]
            },
            'SVM': {
                'C': [0.1, 1, 10],
                'kernel': ['rbf', 'linear'],
                'gamma': ['scale', 'auto']
            }
        }
        
        return param_grids
    
    def train_baseline_models(self, X_train, y_train, cv_folds=5):
        """
        Train baseline models without hyperparameter tuning.
        
        Parameters:
        -----------
        X_train : array-like
            Training features
        y_train : array-like
            Training target
        cv_folds : int
            Number of cross-validation folds
            
        Returns:
        --------
        dict : Cross-validation scores for each model
        """
        logger.info("Training baseline models...")
        
        if not self.models:
            self.initialize_models()
        
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        
        for name, model in self.models.items():
            logger.info(f"Training {name}...")
            
            # Perform cross-validation
            cv_scores = cross_val_score(
                model, X_train, y_train, 
                cv=cv, 
                scoring='roc_auc',
                n_jobs=-1
            )
            
            self.cv_results[name] = {
                'mean_score': cv_scores.mean(),
                'std_score': cv_scores.std(),
                'scores': cv_scores
            }
            
            logger.info(f"{name} - CV ROC-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
            
            # Fit on full training data
            model.fit(X_train, y_train)
        
        logger.info("Baseline model training complete")
        return self.cv_results
    
    def tune_hyperparameters(self, X_train, y_train, 
                            model_names=None,
                            search_method='grid',
                            cv_folds=5,
                            n_iter=50):
        """
        Tune hyperparameters for specified models.
        
        Parameters:
        -----------
        X_train : array-like
            Training features
        y_train : array-like
            Training target
        model_names : list, optional
            List of model names to tune (if None, tune all)
        search_method : str
            'grid' for GridSearchCV or 'random' for RandomizedSearchCV
        cv_folds : int
            Number of cross-validation folds
        n_iter : int
            Number of iterations for RandomizedSearchCV
            
        Returns:
        --------
        dict : Best models with tuned hyperparameters
        """
        logger.info(f"Starting hyperparameter tuning using {search_method} search...")
        
        if not self.models:
            self.initialize_models()
        
        param_grids = self.get_hyperparameter_grids()
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        
        models_to_tune = model_names if model_names else list(self.models.keys())
        tuned_models = {}
        
        for name in models_to_tune:
            if name not in param_grids:
                logger.warning(f"No parameter grid defined for {name}, skipping...")
                continue
            
            logger.info(f"Tuning {name}...")
            
            if search_method == 'grid':
                search = GridSearchCV(
                    self.models[name],
                    param_grids[name],
                    cv=cv,
                    scoring='roc_auc',
                    n_jobs=-1,
                    verbose=1
                )
            else:
                search = RandomizedSearchCV(
                    self.models[name],
                    param_grids[name],
                    n_iter=n_iter,
                    cv=cv,
                    scoring='roc_auc',
                    n_jobs=-1,
                    random_state=self.random_state,
                    verbose=1
                )
            
            search.fit(X_train, y_train)
            
            tuned_models[name] = {
                'model': search.best_estimator_,
                'best_params': search.best_params_,
                'best_score': search.best_score_
            }
            
            # Update the model in self.models
            self.models[name] = search.best_estimator_
            
            logger.info(f"{name} - Best CV Score: {search.best_score_:.4f}")
            logger.info(f"{name} - Best Parameters: {search.best_params_}")
        
        logger.info("Hyperparameter tuning complete")
        return tuned_models
    
    def select_best_model(self, X_val, y_val, metric='roc_auc'):
        """
        Select the best performing model based on validation data.
        
        Parameters:
        -----------
        X_val : array-like
            Validation features
        y_val : array-like
            Validation target
        metric : str
            Metric to use for selection
            
        Returns:
        --------
        tuple : (best_model_name, best_model, best_score)
        """
        from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
        
        logger.info("Selecting best model...")
        
        best_score = -np.inf
        best_name = None
        
        for name, model in self.models.items():
            y_pred_proba = model.predict_proba(X_val)[:, 1]
            
            if metric == 'roc_auc':
                score = roc_auc_score(y_val, y_pred_proba)
            elif metric == 'accuracy':
                y_pred = model.predict(X_val)
                score = accuracy_score(y_val, y_pred)
            elif metric == 'f1':
                y_pred = model.predict(X_val)
                score = f1_score(y_val, y_pred)
            else:
                raise ValueError(f"Unsupported metric: {metric}")
            
            logger.info(f"{name} - Validation {metric}: {score:.4f}")
            
            if score > best_score:
                best_score = score
                best_name = name
        
        self.best_model = self.models[best_name]
        self.best_model_name = best_name
        
        logger.info(f"Best model: {best_name} with {metric} = {best_score:.4f}")
        
        return best_name, self.best_model, best_score
    
    def save_model(self, model=None, filepath=None, model_name=None):
        """
        Save a trained model to disk.
        
        Parameters:
        -----------
        model : object, optional
            Model to save (if None, saves best_model)
        filepath : str, optional
            Path to save the model
        model_name : str, optional
            Name for the model file
            
        Returns:
        --------
        str : Path where model was saved
        """
        if model is None:
            if self.best_model is None:
                raise ValueError("No model to save. Train a model first.")
            model = self.best_model
            name = self.best_model_name
        else:
            name = model_name or 'model'
        
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = f"../models/{name.replace(' ', '_')}_{timestamp}.pkl"
        
        joblib.dump(model, filepath)
        logger.info(f"Model saved to {filepath}")
        
        return filepath
    
    def load_model(self, filepath):
        """
        Load a saved model from disk.
        
        Parameters:
        -----------
        filepath : str
            Path to the saved model
            
        Returns:
        --------
        object : Loaded model
        """
        model = joblib.load(filepath)
        logger.info(f"Model loaded from {filepath}")
        
        return model
    
    def train_with_pipeline(self, X_train, y_train, X_val, y_val,
                           tune_hyperparameters=True,
                           search_method='grid',
                           cv_folds=5):
        """
        Complete training pipeline.
        
        Parameters:
        -----------
        X_train : array-like
            Training features
        y_train : array-like
            Training target
        X_val : array-like
            Validation features
        y_val : array-like
            Validation target
        tune_hyperparameters : bool
            Whether to perform hyperparameter tuning
        search_method : str
            'grid' or 'random' for hyperparameter search
        cv_folds : int
            Number of cross-validation folds
            
        Returns:
        --------
        tuple : (best_model_name, best_model)
        """
        logger.info("Starting complete training pipeline...")
        
        # Initialize models
        self.initialize_models()
        
        # Train baseline models
        self.train_baseline_models(X_train, y_train, cv_folds=cv_folds)
        
        # Tune hyperparameters if requested
        if tune_hyperparameters:
            # Select top 3 models for tuning based on baseline performance
            top_models = sorted(
                self.cv_results.items(),
                key=lambda x: x[1]['mean_score'],
                reverse=True
            )[:3]
            
            top_model_names = [name for name, _ in top_models]
            logger.info(f"Tuning top models: {top_model_names}")
            
            self.tune_hyperparameters(
                X_train, y_train,
                model_names=top_model_names,
                search_method=search_method,
                cv_folds=cv_folds
            )
        
        # Select best model
        best_name, best_model, _ = self.select_best_model(X_val, y_val)
        
        logger.info("Training pipeline complete")
        
        return best_name, best_model


def split_data(X, y, test_size=0.2, val_size=0.1, random_state=42):
    """
    Split data into train, validation, and test sets.
    
    Parameters:
    -----------
    X : array-like
        Features
    y : array-like
        Target
    test_size : float
        Proportion of data for test set
    val_size : float
        Proportion of training data for validation set
    random_state : int
        Random state for reproducibility
        
    Returns:
    --------
    tuple : (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    # First split: separate test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Second split: separate validation set from training
    val_ratio = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_ratio, random_state=random_state, stratify=y_temp
    )
    
    logger.info(f"Data split - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    print("Model Training Module - Ready for use")
