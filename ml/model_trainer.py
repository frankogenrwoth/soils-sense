import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    RandomForestClassifier,
    GradientBoostingClassifier,
)
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import cross_val_score, GridSearchCV
import json
import time
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

from ml.config import (
    MODELS_DIR,
    MODEL_CONFIGS,
    REGRESSION_ALGORITHMS,
    DEFAULT_ALGORITHMS,
    CLASSIFICATION_ALGORITHMS,
)
from ml.data_processor import DataProcessor


class ModelTrainer:
    """Train and evaluate models"""

    def __init__(self):
        self.data_processor = DataProcessor()
        self.models = {}
        self.training_results = {}

    def train_model(
        self,
        model_type,
        algorithm=None,
        custom_data=None,
        version=None,
    ):
        """Train a regression model for the specified type algorithm and save it with a version number

        Args:
            model_type (str): Type of model to train (e.g., 'soil_moisture_predictor')
            algorithm (str, optional): Algorithm to use (e.g., 'random_forest', 'gradient_boosting')
            custom_data (pandas.DataFrame, optional): Custom training data
            version (int, optional): Version of the model to train
        Returns:
            tuple: (dict, dict or None) - Training results and inspection findings
        """
        if algorithm is None:
            algorithm = DEFAULT_ALGORITHMS.get(model_type, "gradient_boosting")

        if custom_data is None:
            data = self.data_processor.load_training_data(model_type)
        else:
            data = custom_data

        # save data inspection findings to training_logs
        self.data_processor.training_logs["data_inspection"] = (
            self.data_processor.inspect_dataframe(data)
        )

        try:
            X_train, X_test, y_train, y_test, feature_names, preprocessor = (
                self.data_processor.prepare_data(data=data, model_type=model_type)
            )
        except Exception as e:
            raise Exception(f"Error preparing data: {e}")

        start_time = time.time()

        task_type = MODEL_CONFIGS[model_type].get("task_type", "regression")

        if task_type == "classification":
            algorithm_config = CLASSIFICATION_ALGORITHMS.get(algorithm, {})
        else:
            algorithm_config = REGRESSION_ALGORITHMS.get(algorithm, {})

        model = self._create_model(algorithm, algorithm_config, task_type)
        model.fit(X_train, y_train)

        training_time = time.time() - start_time

        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)

        if task_type == "classification":
            from sklearn.metrics import accuracy_score, f1_score

            train_accuracy = accuracy_score(y_train, y_pred_train)
            test_accuracy = accuracy_score(y_test, y_pred_test)
            train_f1 = f1_score(y_train, y_pred_train, average="weighted")
            test_f1 = f1_score(y_test, y_pred_test, average="weighted")

            cv_scores = cross_val_score(
                model, X_train, y_train, cv=5, scoring="accuracy"
            )
            cv_mean = cv_scores.mean()
            cv_std = cv_scores.std()

            train_r2 = test_r2 = train_rmse = test_rmse = train_mae = test_mae = None
        else:
            train_r2 = r2_score(y_train, y_pred_train)
            test_r2 = r2_score(y_test, y_pred_test)
            train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
            test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
            train_mae = mean_absolute_error(y_train, y_pred_train)
            test_mae = mean_absolute_error(y_test, y_pred_test)

            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="r2")
            cv_mean = cv_scores.mean()
            cv_std = cv_scores.std()

            train_accuracy = test_accuracy = train_f1 = test_f1 = None

        model_key = f"{model_type}_{algorithm}"
        self.models[model_key] = model
        self.training_results[model_key] = {
            "model_type": model_type,
            "algorithm": algorithm,
            "task_type": task_type,
            "training_time": training_time,
            "n_samples": len(X_train),
            "n_features": len(feature_names),
            "feature_names": feature_names,
            "feature_columns": self.data_processor.feature_columns.get(model_type, []),
            "model_name": model.__class__.__name__,
        }

        if task_type == "classification":
            self.training_results[model_key].update(
                {
                    "train_accuracy": train_accuracy,
                    "test_accuracy": test_accuracy,
                    "train_f1": train_f1,
                    "test_f1": test_f1,
                    "cv_mean": cv_mean,
                    "cv_std": cv_std,
                }
            )
        else:
            self.training_results[model_key].update(
                {
                    "train_r2": train_r2,
                    "test_r2": test_r2,
                    "train_rmse": train_rmse,
                    "test_rmse": test_rmse,
                    "train_mae": train_mae,
                    "test_mae": test_mae,
                    "cv_mean": cv_mean,
                    "cv_std": cv_std,
                }
            )

        # save training results to training_logs
        self.training_results[model_key][
            "training_logs"
        ] = self.data_processor.training_logs

        self._save_model(model_key, model, preprocessor, version)

        print(f"Training Results for {model_type} ({algorithm}) - {task_type.upper()}:")
        if task_type == "classification":
            print(f"Training Accuracy: {train_accuracy:.4f}")
            print(f"Test Accuracy: {test_accuracy:.4f}")
            print(f"Training F1: {train_f1:.4f}")
            print(f"Test F1: {test_f1:.4f}")
            print(f"Cross-validation Accuracy: {cv_mean:.4f} (±{cv_std:.4f})")
        else:
            print(f"Training R²: {train_r2:.4f}")
            print(f"Test R²: {test_r2:.4f}")
            print(f"Training RMSE: {train_rmse:.4f}")
            print(f"Test RMSE: {test_rmse:.4f}")
            print(f"Cross-validation R²: {cv_mean:.4f} (±{cv_std:.4f})")

        print(f"Training time: {training_time:.2f} seconds")

        return self.training_results[model_key]

    def _create_model(self, algorithm, config, task_type="regression"):
        """Create model based on algorithm, configuration, and task type

        Args:
            algorithm (str): Algorithm name
            config (dict): Model configuration parameters
            task_type (str): Type of task ("regression" or "classification")

        Returns:
            sklearn.base.BaseEstimator: Model instance
        """
        if task_type == "classification":
            if algorithm == "random_forest":
                return RandomForestClassifier(**config)
            elif algorithm == "gradient_boosting":
                return GradientBoostingClassifier(**config)
            elif algorithm == "logistic_regression":
                return LogisticRegression(**config)
            else:
                raise ValueError(f"Unknown classification algorithm: {algorithm}")
        else:  # regression
            if algorithm == "random_forest":
                return RandomForestRegressor(**config)
            elif algorithm == "gradient_boosting":
                return GradientBoostingRegressor(**config)
            elif algorithm == "svr":
                return SVR(**config)
            elif algorithm == "mlp":
                return MLPRegressor(**config)
            elif algorithm == "linear_regression":
                return LinearRegression(**config)
            else:
                raise ValueError(f"Unknown regression algorithm: {algorithm}")

    def _save_model(self, model_key, model, preprocessor, version):
        """Save trained model and preprocessor to file

        Args:
            model_key (str): Unique model identifier
            model: Trained model instance
            preprocessor: Fitted preprocessor
        """
        if version is None:
            model_path = MODELS_DIR / f"{model_key}.joblib"
        else:
            model_path = MODELS_DIR / f"{model_key}_version_{version}.joblib"

        joblib.dump(model, model_path)

        if version is None:
            preprocessor_path = MODELS_DIR / f"{model_key}_preprocessor.joblib"
        else:
            preprocessor_path = (
                MODELS_DIR / f"{model_key}_version_{version}_preprocessor.joblib"
            )

        joblib.dump(preprocessor, preprocessor_path)

        if version is None:
            results_path = MODELS_DIR / f"{model_key}_results.json"
        else:
            results_path = MODELS_DIR / f"{model_key}_version_{version}_results.json"

        with open(results_path, "w") as f:
            json.dump(self.training_results[model_key], f, indent=2, default=str)

    def load_model(self, model_key):
        """Load a trained model from file

        Args:
            model_key (str): Model identifier

        Returns:
            tuple: (model, preprocessor, results)
        """
        model_path = MODELS_DIR / f"{model_key}.joblib"
        preprocessor_path = MODELS_DIR / f"{model_key}_preprocessor.joblib"
        results_path = MODELS_DIR / f"{model_key}_results.json"

        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        model = joblib.load(model_path)
        preprocessor = (
            joblib.load(preprocessor_path) if preprocessor_path.exists() else None
        )

        results = {}
        if results_path.exists():
            with open(results_path, "r") as f:
                results = json.load(f)

        return model, preprocessor, results

    def get_model_info(self, model_key):
        """Get information about a trained model

        Args:
            model_key (str): Model identifier

        Returns:
            dict: Model information
        """
        if model_key in self.training_results:
            return self.training_results[model_key]

        try:
            _, _, results = self.load_model(model_key)
            return results
        except FileNotFoundError:
            return {"error": f"Model {model_key} not found"}

    def list_trained_models(self):
        """List all trained models with their information

        Returns:
            list: List of dictionaries containing model information
        """
        model_files = list(MODELS_DIR.glob("*_results.json"))
        models = []
        for file in model_files:
            model_key = file.stem.replace("_results", "")
            model_info = self.get_model_info(model_key)
            models.append(model_info)
        return models

    def get_available_algorithms(self, model_type):
        """Get available algorithms for a specific model type

        Args:
            model_type (str): Type of model

        Returns:
            list: List of available algorithms
        """
        if MODEL_CONFIGS[model_type]["task_type"] == "regression":
            return list(REGRESSION_ALGORITHMS.keys())
        elif MODEL_CONFIGS[model_type]["task_type"] == "classification":
            return list(CLASSIFICATION_ALGORITHMS.keys())
        else:
            raise ValueError(
                f"Unknown task type: {MODEL_CONFIGS[model_type]['task_type']}"
            )

    def get_best_model(self, model_type):
        """Get the best performing model for a specific type

        Args:
            model_type (str): Type of model

        Returns:
            str: Best model key
        """
        pass

    def evaluate_model(self, model_key, test_data):
        """Evaluate a model on new test data

        Args:
            model_key (str): Model identifier
            test_data (pandas.DataFrame): Test data

        Returns:
            dict: Evaluation results
        """
        pass
