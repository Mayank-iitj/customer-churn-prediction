"""
Model Evaluation Module for Customer Churn Prediction

This module handles:
- Model performance evaluation
- Visualization of results (ROC curves, confusion matrices)
- Feature importance analysis
- SHAP values for interpretability
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix,
    classification_report, precision_recall_curve
)
from sklearn.inspection import permutation_importance
import logging
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

# Set style for visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)


class ModelEvaluator:
    """
    Comprehensive model evaluation class for customer churn prediction.
    """
    
    def __init__(self):
        """Initialize the evaluator."""
        self.evaluation_results = {}
        
    def calculate_metrics(self, y_true, y_pred, y_pred_proba=None):
        """
        Calculate comprehensive classification metrics.
        
        Parameters:
        -----------
        y_true : array-like
            True labels
        y_pred : array-like
            Predicted labels
        y_pred_proba : array-like, optional
            Predicted probabilities
            
        Returns:
        --------
        dict : Dictionary of calculated metrics
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1_score': f1_score(y_true, y_pred, zero_division=0)
        }
        
        if y_pred_proba is not None:
            metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
        
        logger.info("Calculated classification metrics")
        
        return metrics
    
    def print_classification_report(self, y_true, y_pred, target_names=None):
        """
        Print detailed classification report.
        
        Parameters:
        -----------
        y_true : array-like
            True labels
        y_pred : array-like
            Predicted labels
        target_names : list, optional
            Names of target classes
        """
        report = classification_report(
            y_true, y_pred, 
            target_names=target_names,
            zero_division=0
        )
        
        print("\n" + "="*60)
        print("CLASSIFICATION REPORT")
        print("="*60)
        print(report)
        print("="*60 + "\n")
        
        return report
    
    def plot_confusion_matrix(self, y_true, y_pred, labels=None, 
                             save_path=None, normalize=False):
        """
        Plot confusion matrix.
        
        Parameters:
        -----------
        y_true : array-like
            True labels
        y_pred : array-like
            Predicted labels
        labels : list, optional
            Class labels
        save_path : str, optional
            Path to save the plot
        normalize : bool
            Whether to normalize the confusion matrix
            
        Returns:
        --------
        matplotlib.figure.Figure : The confusion matrix figure
        """
        cm = confusion_matrix(y_true, y_pred)
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.2%'
            title = 'Normalized Confusion Matrix'
        else:
            fmt = 'd'
            title = 'Confusion Matrix'
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            cm, annot=True, fmt=fmt, cmap='Blues',
            xticklabels=labels or ['No Churn', 'Churn'],
            yticklabels=labels or ['No Churn', 'Churn'],
            ax=ax
        )
        
        ax.set_ylabel('True Label')
        ax.set_xlabel('Predicted Label')
        ax.set_title(title)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Confusion matrix saved to {save_path}")
        
        return fig
    
    def plot_roc_curve(self, y_true, y_pred_proba, save_path=None):
        """
        Plot ROC curve.
        
        Parameters:
        -----------
        y_true : array-like
            True labels
        y_pred_proba : array-like
            Predicted probabilities
        save_path : str, optional
            Path to save the plot
            
        Returns:
        --------
        matplotlib.figure.Figure : The ROC curve figure
        """
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
        roc_auc = roc_auc_score(y_true, y_pred_proba)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.3f})')
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
                label='Random Classifier')
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('Receiver Operating Characteristic (ROC) Curve')
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"ROC curve saved to {save_path}")
        
        return fig
    
    def plot_precision_recall_curve(self, y_true, y_pred_proba, save_path=None):
        """
        Plot Precision-Recall curve.
        
        Parameters:
        -----------
        y_true : array-like
            True labels
        y_pred_proba : array-like
            Predicted probabilities
        save_path : str, optional
            Path to save the plot
            
        Returns:
        --------
        matplotlib.figure.Figure : The PR curve figure
        """
        precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(recall, precision, color='blue', lw=2)
        
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title('Precision-Recall Curve')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Precision-Recall curve saved to {save_path}")
        
        return fig
    
    def compare_models(self, models_dict, X_test, y_test, save_path=None):
        """
        Compare multiple models on test data.
        
        Parameters:
        -----------
        models_dict : dict
            Dictionary of {model_name: model} pairs
        X_test : array-like
            Test features
        y_test : array-like
            Test labels
        save_path : str, optional
            Path to save comparison plot
            
        Returns:
        --------
        pandas.DataFrame : Comparison results
        """
        results = []
        
        for name, model in models_dict.items():
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            metrics = self.calculate_metrics(y_test, y_pred, y_pred_proba)
            metrics['model'] = name
            results.append(metrics)
        
        df_results = pd.DataFrame(results)
        df_results = df_results.set_index('model')
        
        # Plot comparison
        fig, ax = plt.subplots(figsize=(12, 6))
        df_results.plot(kind='bar', ax=ax)
        
        ax.set_title('Model Comparison')
        ax.set_ylabel('Score')
        ax.set_xlabel('Model')
        ax.legend(loc='lower right')
        ax.set_ylim([0, 1])
        plt.xticks(rotation=45, ha='right')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Model comparison plot saved to {save_path}")
        
        logger.info("\nModel Comparison Results:")
        print(df_results.to_string())
        
        return df_results
    
    def plot_feature_importance(self, model, feature_names, top_n=20, save_path=None):
        """
        Plot feature importance for tree-based models.
        
        Parameters:
        -----------
        model : object
            Trained model with feature_importances_ attribute
        feature_names : list
            Names of features
        top_n : int
            Number of top features to display
        save_path : str, optional
            Path to save the plot
            
        Returns:
        --------
        pandas.DataFrame : Feature importance dataframe
        """
        if not hasattr(model, 'feature_importances_'):
            logger.warning("Model does not have feature_importances_ attribute")
            return None
        
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:top_n]
        
        df_importance = pd.DataFrame({
            'feature': [feature_names[i] for i in indices],
            'importance': importances[indices]
        })
        
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.barplot(
            data=df_importance, 
            x='importance', 
            y='feature',
            palette='viridis',
            ax=ax
        )
        
        ax.set_title(f'Top {top_n} Feature Importances')
        ax.set_xlabel('Importance')
        ax.set_ylabel('Feature')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Feature importance plot saved to {save_path}")
        
        return df_importance
    
    def calculate_permutation_importance(self, model, X, y, feature_names, 
                                        n_repeats=10, random_state=42,
                                        top_n=20, save_path=None):
        """
        Calculate and plot permutation importance.
        
        Parameters:
        -----------
        model : object
            Trained model
        X : array-like
            Features
        y : array-like
            Target
        feature_names : list
            Names of features
        n_repeats : int
            Number of times to permute each feature
        random_state : int
            Random state
        top_n : int
            Number of top features to display
        save_path : str, optional
            Path to save the plot
            
        Returns:
        --------
        pandas.DataFrame : Permutation importance dataframe
        """
        logger.info("Calculating permutation importance...")
        
        perm_importance = permutation_importance(
            model, X, y,
            n_repeats=n_repeats,
            random_state=random_state,
            n_jobs=-1
        )
        
        indices = np.argsort(perm_importance.importances_mean)[::-1][:top_n]
        
        df_perm = pd.DataFrame({
            'feature': [feature_names[i] for i in indices],
            'importance_mean': perm_importance.importances_mean[indices],
            'importance_std': perm_importance.importances_std[indices]
        })
        
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.barh(
            range(len(indices)),
            df_perm['importance_mean'],
            xerr=df_perm['importance_std'],
            align='center'
        )
        ax.set_yticks(range(len(indices)))
        ax.set_yticklabels(df_perm['feature'])
        ax.set_xlabel('Permutation Importance')
        ax.set_title(f'Top {top_n} Permutation Feature Importances')
        ax.invert_yaxis()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Permutation importance plot saved to {save_path}")
        
        return df_perm
    
    def calculate_shap_values(self, model, X, feature_names, 
                             sample_size=100, save_path=None):
        """
        Calculate and visualize SHAP values for model interpretability.
        
        Parameters:
        -----------
        model : object
            Trained model
        X : array-like
            Features (will sample from this)
        feature_names : list
            Names of features
        sample_size : int
            Number of samples to use for SHAP calculation
        save_path : str, optional
            Path to save the plot
            
        Returns:
        --------
        object : SHAP explainer object
        """
        try:
            import shap
            
            logger.info("Calculating SHAP values...")
            
            # Sample data for faster computation
            if len(X) > sample_size:
                sample_indices = np.random.choice(len(X), sample_size, replace=False)
                X_sample = X.iloc[sample_indices] if hasattr(X, 'iloc') else X[sample_indices]
            else:
                X_sample = X
            
            # Create explainer based on model type
            if hasattr(model, 'tree_'):
                explainer = shap.TreeExplainer(model)
            else:
                explainer = shap.Explainer(model, X_sample)
            
            shap_values = explainer.shap_values(X_sample)
            
            # Handle multiple outputs (e.g., binary classification)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
            
            # Summary plot
            plt.figure(figsize=(10, 8))
            shap.summary_plot(shap_values, X_sample, feature_names=feature_names, show=False)
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"SHAP summary plot saved to {save_path}")
            
            plt.show()
            
            return explainer
            
        except ImportError:
            logger.warning("SHAP library not installed. Install with: pip install shap")
            return None
        except Exception as e:
            logger.error(f"Error calculating SHAP values: {e}")
            return None
    
    def evaluate_model(self, model, X_test, y_test, feature_names=None,
                      save_dir=None, model_name='model'):
        """
        Comprehensive model evaluation.
        
        Parameters:
        -----------
        model : object
            Trained model
        X_test : array-like
            Test features
        y_test : array-like
            Test labels
        feature_names : list, optional
            Names of features
        save_dir : str, optional
            Directory to save plots
        model_name : str
            Name of the model for saving files
            
        Returns:
        --------
        dict : Evaluation results
        """
        logger.info(f"Evaluating {model_name}...")
        
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        metrics = self.calculate_metrics(y_test, y_pred, y_pred_proba)
        
        # Print classification report
        self.print_classification_report(y_test, y_pred)
        
        # Generate visualizations
        if save_dir:
            import os
            os.makedirs(save_dir, exist_ok=True)
            
            self.plot_confusion_matrix(
                y_test, y_pred,
                save_path=f"{save_dir}/{model_name}_confusion_matrix.png"
            )
            
            self.plot_roc_curve(
                y_test, y_pred_proba,
                save_path=f"{save_dir}/{model_name}_roc_curve.png"
            )
            
            self.plot_precision_recall_curve(
                y_test, y_pred_proba,
                save_path=f"{save_dir}/{model_name}_pr_curve.png"
            )
            
            if feature_names and hasattr(model, 'feature_importances_'):
                self.plot_feature_importance(
                    model, feature_names,
                    save_path=f"{save_dir}/{model_name}_feature_importance.png"
                )
        else:
            self.plot_confusion_matrix(y_test, y_pred)
            plt.show()
            
            self.plot_roc_curve(y_test, y_pred_proba)
            plt.show()
        
        # Store results
        self.evaluation_results[model_name] = {
            'metrics': metrics,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }
        
        logger.info(f"Evaluation complete for {model_name}")
        
        return metrics


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    print("Model Evaluation Module - Ready for use")
