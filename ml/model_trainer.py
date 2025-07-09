import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import cross_val_score
import json
import time
from pathlib import Path

from .config import MODELS_DIR, MODEL_CONFIGS
from .data_processor import DataProcessor


class ModelTrainer:
    """Handles regression model training and evaluation"""

    def __init__(self):
        """Initialize ModelTrainer with data processor and empty containers for models and results"""
        self.data_processor = DataProcessor()
        self.models = {}
        self.training_results = {}

    def train_model(self, model_type, custom_data=None):
        """Train a regression model for the specified type

        Args:
            model_type (str): Type of model to train
            custom_data (pandas.DataFrame, optional): Custom training data

        Returns:
            dict: Training results including R² score, RMSE, and model info
        """
        if custom_data is None:
            data = self.data_processor.load_training_data(model_type)
        else:
            data = custom_data
        X, y = self.data_processor.prepare_data(data, model_type, self.features, self.target)
        model = self._create_model(MODEL_CONFIGS[model_type])
        model.fit(X, y)
        self._save_model(model_type, model)
        self.training_results[model_type] = self._get_model_info(model_type)
        return self.training_results[model_type]


    def _create_model(self, config):
        """Create regression model based on configuration

        Args:
            config (dict): Model configuration parameters

        Returns:
            sklearn.base.BaseEstimator: Regression model instance
        """
        if model_type == "random_forest":
            model = RandomForestRegressor(**config)
        elif model_type == "svr":
            model = SVR(**config)
        elif model_type == "mlp":
            model = MLPRegressor(**config)
        return model


    def _save_model(self, model_type, model):
        """Save trained model to file

        Args:
            model_type (str): Type of model
            model: Trained model instance
        """
        model_path = Path(MODELS_DIR) / f"{model_type}.joblib"
        joblib.dump(model, model_path)
        print(f"Model saved to {model_path}")

    def load_model(self, model_type):
        """Load a trained model from file

        Args:
            model_type (str): Type of model to load

        Returns:
            sklearn.base.BaseEstimator: Loaded model instance
        """
        model_path = Path(MODELS_DIR) / f"{model_type}.joblib"
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        return joblib.load(model_path)

    def get_model_info(self, model_type):
        """Get information about a trained model

        Args:
            model_type (str): Type of model

        Returns:
            dict: Model information including R² score, RMSE, training time, etc.
        """
        model = self.load_model(model_type)
        return {
            "model_type": model_type,
            "model_name": model.__class__.__name__,
            "features": self.features,
            "target": self.target,
            "training_time": self.training_results[model_type]["training_time"],
        }


    def list_trained_models(self):
        """List all trained models with their information

        Returns:
            list: List of dictionaries containing model information
        """
        model_files = list(Path(MODELS_DIR).glob("*.joblib"))

    def retrain_model(self, model_type, new_data=None):
        """Retrain an existing model with new data

        Args:
            model_type (str): Type of model to retrain
            new_data (pandas.DataFrame, optional): New training data

        Returns:
            dict: Updated training results
        """
        pass

    def evaluate_model(self, model_type, test_data):
        """Evaluate a model on new test data

        Args:
            model_type (str): Type of model to evaluate
            test_data (dict): Test data dictionary

        Returns:
            dict: Evaluation results including predictions and confidence intervals
        """
        pass

    def calculate_prediction_interval(self, model, X_test, confidence_level=0.95):
        """Calculate prediction intervals for regression model

        Args:
            model: Trained regression model
            X_test (numpy.ndarray): Test features
            confidence_level (float): Confidence level for interval

        Returns:
            tuple: (lower_bound, upper_bound) prediction intervals
        """
        pass
