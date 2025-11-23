"""
Customer Churn Prediction Package
"""

__version__ = "1.0.0"
__author__ = "Your Name"
__description__ = "A comprehensive machine learning solution for customer churn prediction"

from . import preprocessing
from . import model_training
from . import evaluation
from . import config
from . import utils

__all__ = [
    'preprocessing',
    'model_training',
    'evaluation',
    'config',
    'utils'
]
