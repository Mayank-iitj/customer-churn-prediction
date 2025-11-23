"""
Main Pipeline for Customer Churn Prediction

This script orchestrates the complete workflow:
1. Data loading and exploration
2. Data preprocessing
3. Model training
4. Model evaluation
5. Model persistence
"""

import sys
import os
import logging
from datetime import datetime
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from preprocessing import DataPreprocessor, load_data
from model_training import ChurnModelTrainer, split_data
from evaluation import ModelEvaluator
from config import Config


def setup_logging(log_dir='logs'):
    """
    Set up logging configuration.
    
    Parameters:
    -----------
    log_dir : str
        Directory to store log files
    """
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f'churn_prediction_{timestamp}.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info("="*80)
    logger.info("Customer Churn Prediction Pipeline Started")
    logger.info("="*80)
    
    return logger


def load_and_explore_data(filepath, logger):
    """
    Load and perform initial exploration of the data.
    
    Parameters:
    -----------
    filepath : str
        Path to the data file
    logger : logging.Logger
        Logger object
        
    Returns:
    --------
    pandas.DataFrame : Loaded data
    """
    logger.info("Step 1: Loading data...")
    
    df = load_data(filepath)
    
    logger.info(f"Dataset shape: {df.shape}")
    logger.info(f"Columns: {list(df.columns)}")
    logger.info(f"\nData types:\n{df.dtypes}")
    logger.info(f"\nMissing values:\n{df.isnull().sum()}")
    
    if 'Churn' in df.columns:
        churn_distribution = df['Churn'].value_counts()
        logger.info(f"\nChurn distribution:\n{churn_distribution}")
        logger.info(f"Churn rate: {churn_distribution[1] / len(df) * 100:.2f}%")
    
    return df


def preprocess_data(df, config, logger):
    """
    Preprocess the data.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe
    config : Config
        Configuration object
    logger : logging.Logger
        Logger object
        
    Returns:
    --------
    tuple : (X, y, preprocessor)
    """
    logger.info("\nStep 2: Preprocessing data...")
    
    preprocessor = DataPreprocessor(scaling_method=config.SCALING_METHOD)
    
    X, y = preprocessor.preprocess_pipeline(
        df,
        target_column=config.TARGET_COLUMN,
        encoding_method=config.ENCODING_METHOD,
        handle_imbalance=config.HANDLE_IMBALANCE,
        fit=True
    )
    
    logger.info(f"Preprocessing complete. Features shape: {X.shape}")
    
    return X, y, preprocessor


def train_models(X_train, y_train, X_val, y_val, config, logger):
    """
    Train and tune models.
    
    Parameters:
    -----------
    X_train, y_train : array-like
        Training data
    X_val, y_val : array-like
        Validation data
    config : Config
        Configuration object
    logger : logging.Logger
        Logger object
        
    Returns:
    --------
    tuple : (trainer, best_model_name, best_model)
    """
    logger.info("\nStep 3: Training models...")
    
    trainer = ChurnModelTrainer(random_state=config.RANDOM_STATE)
    
    best_model_name, best_model = trainer.train_with_pipeline(
        X_train, y_train, X_val, y_val,
        tune_hyperparameters=config.TUNE_HYPERPARAMETERS,
        search_method=config.SEARCH_METHOD,
        cv_folds=config.CV_FOLDS
    )
    
    logger.info(f"\nBest model: {best_model_name}")
    
    return trainer, best_model_name, best_model


def evaluate_models(trainer, X_test, y_test, feature_names, config, logger):
    """
    Evaluate trained models.
    
    Parameters:
    -----------
    trainer : ChurnModelTrainer
        Trainer object with trained models
    X_test, y_test : array-like
        Test data
    feature_names : list
        Names of features
    config : Config
        Configuration object
    logger : logging.Logger
        Logger object
        
    Returns:
    --------
    ModelEvaluator : Evaluator object with results
    """
    logger.info("\nStep 4: Evaluating models...")
    
    evaluator = ModelEvaluator()
    
    # Create results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join('results', timestamp)
    os.makedirs(results_dir, exist_ok=True)
    
    # Evaluate best model
    logger.info(f"\nEvaluating best model: {trainer.best_model_name}")
    metrics = evaluator.evaluate_model(
        trainer.best_model,
        X_test, y_test,
        feature_names=feature_names,
        save_dir=results_dir,
        model_name=trainer.best_model_name.replace(' ', '_')
    )
    
    # Compare all models
    logger.info("\nComparing all models...")
    comparison_df = evaluator.compare_models(
        trainer.models,
        X_test, y_test,
        save_path=os.path.join(results_dir, 'model_comparison.png')
    )
    
    # Save comparison results
    comparison_df.to_csv(os.path.join(results_dir, 'model_comparison.csv'))
    
    # Calculate permutation importance for best model
    if config.CALCULATE_PERMUTATION_IMPORTANCE:
        logger.info("\nCalculating permutation importance...")
        perm_importance = evaluator.calculate_permutation_importance(
            trainer.best_model,
            X_test, y_test,
            feature_names,
            save_path=os.path.join(results_dir, 'permutation_importance.png')
        )
        perm_importance.to_csv(os.path.join(results_dir, 'permutation_importance.csv'))
    
    # Calculate SHAP values if configured
    if config.CALCULATE_SHAP:
        logger.info("\nCalculating SHAP values...")
        evaluator.calculate_shap_values(
            trainer.best_model,
            X_test,
            feature_names,
            sample_size=config.SHAP_SAMPLE_SIZE,
            save_path=os.path.join(results_dir, 'shap_summary.png')
        )
    
    logger.info(f"\nResults saved to: {results_dir}")
    
    return evaluator, results_dir


def save_artifacts(trainer, preprocessor, results_dir, logger):
    """
    Save trained models and preprocessor.
    
    Parameters:
    -----------
    trainer : ChurnModelTrainer
        Trainer object
    preprocessor : DataPreprocessor
        Preprocessor object
    results_dir : str
        Directory to save artifacts
    logger : logging.Logger
        Logger object
    """
    logger.info("\nStep 5: Saving artifacts...")
    
    # Save best model
    model_path = os.path.join(results_dir, 'best_model.pkl')
    trainer.save_model(filepath=model_path)
    
    # Save preprocessor
    import joblib
    preprocessor_path = os.path.join(results_dir, 'preprocessor.pkl')
    joblib.dump(preprocessor, preprocessor_path)
    logger.info(f"Preprocessor saved to {preprocessor_path}")
    
    # Save all models
    models_dir = os.path.join(results_dir, 'all_models')
    os.makedirs(models_dir, exist_ok=True)
    
    for name, model in trainer.models.items():
        model_file = os.path.join(models_dir, f"{name.replace(' ', '_')}.pkl")
        joblib.dump(model, model_file)
    
    logger.info(f"All models saved to {models_dir}")


def main():
    """Main execution function."""
    
    # Setup logging
    logger = setup_logging()
    
    try:
        # Load configuration
        config = Config()
        
        # Load and explore data
        df = load_and_explore_data(config.DATA_PATH, logger)
        
        # Preprocess data
        X, y, preprocessor = preprocess_data(df, config, logger)
        
        # Split data
        logger.info("\nSplitting data...")
        X_train, X_val, X_test, y_train, y_val, y_test = split_data(
            X, y,
            test_size=config.TEST_SIZE,
            val_size=config.VAL_SIZE,
            random_state=config.RANDOM_STATE
        )
        
        # Get feature names
        if hasattr(X, 'columns'):
            feature_names = list(X.columns)
        else:
            feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        
        # Train models
        trainer, best_model_name, best_model = train_models(
            X_train, y_train, X_val, y_val, config, logger
        )
        
        # Evaluate models
        evaluator, results_dir = evaluate_models(
            trainer, X_test, y_test, feature_names, config, logger
        )
        
        # Save artifacts
        save_artifacts(trainer, preprocessor, results_dir, logger)
        
        # Summary
        logger.info("\n" + "="*80)
        logger.info("Pipeline completed successfully!")
        logger.info(f"Best model: {best_model_name}")
        logger.info(f"Results saved to: {results_dir}")
        logger.info("="*80)
        
    except Exception as e:
        logger.error(f"Pipeline failed with error: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
