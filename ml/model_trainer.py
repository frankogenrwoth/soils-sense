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

# Handle imports for both direct execution and module import
try:
    from .config import (
        MODELS_DIR,
        MODEL_CONFIGS,
        MODEL_ALGORITHMS,
        DEFAULT_ALGORITHMS,
        TRAINING_CONFIG,
        CLASSIFICATION_ALGORITHMS,
    )
    from .data_processor import DataProcessor
except ImportError:
    from config import (
        MODELS_DIR,
        MODEL_CONFIGS,
        MODEL_ALGORITHMS,
        DEFAULT_ALGORITHMS,
        TRAINING_CONFIG,
        CLASSIFICATION_ALGORITHMS,
    )
    from data_processor import DataProcessor


class ModelTrainer:
    """Handles regression model training and evaluation with comprehensive feature engineering"""

    def __init__(self):
        """Initialize ModelTrainer with data processor and empty containers for models and results"""
        self.data_processor = DataProcessor()
        self.models = {}
        self.training_results = {}
        self.best_models = {}

    def train_model(
        self, model_type, algorithm=None, custom_data=None, hyperparameter_tuning=False
    ):
        """Train a regression model for the specified type

        Args:
            model_type (str): Type of model to train (e.g., 'soil_moisture_predictor')
            algorithm (str, optional): Algorithm to use (e.g., 'random_forest', 'gradient_boosting')
            custom_data (pandas.DataFrame, optional): Custom training data
            hyperparameter_tuning (bool): Whether to perform hyperparameter tuning

        Returns:
            dict: Training results including R² score, RMSE, and model info
        """
        print(f"🚀 Starting training for {model_type}...")

        # Use default algorithm if not specified
        if algorithm is None:
            algorithm = DEFAULT_ALGORITHMS.get(model_type, "gradient_boosting")

        print(f"📊 Using algorithm: {algorithm}")

        # Load or use custom data
        if custom_data is None:
            data = self.data_processor.load_training_data(model_type)
        else:
            data = custom_data

        print(f"📈 Loaded {len(data)} training samples")

        # Prepare data using the new data processor
        try:
            X_train, X_test, y_train, y_test, feature_names, preprocessor = (
                self.data_processor.prepare_data(data=data, model_type=model_type)
            )
            print(f"✅ Data preparation completed. Features: {len(feature_names)}")
        except Exception as e:
            print(f"❌ Data preparation failed: {e}")
            raise

        # Create and train model
        start_time = time.time()

        # Get task type and algorithm configuration
        task_type = MODEL_CONFIGS[model_type].get("task_type", "regression")
        if task_type == "classification":
            algorithm_config = CLASSIFICATION_ALGORITHMS.get(algorithm, {})
        else:
            algorithm_config = MODEL_ALGORITHMS.get(algorithm, {})

        if hyperparameter_tuning:
            model = self._train_with_hyperparameter_tuning(
                algorithm, X_train, y_train, model_type, task_type
            )
        else:
            model = self._create_model(algorithm, algorithm_config, task_type)
            model.fit(X_train, y_train)

        training_time = time.time() - start_time

        # Evaluate model
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)

        # Calculate metrics based on task type
        if task_type == "classification":
            from sklearn.metrics import accuracy_score, f1_score

            # Classification metrics
            train_accuracy = accuracy_score(y_train, y_pred_train)
            test_accuracy = accuracy_score(y_test, y_pred_test)
            train_f1 = f1_score(y_train, y_pred_train, average="weighted")
            test_f1 = f1_score(y_test, y_pred_test, average="weighted")

            # Cross-validation score
            cv_scores = cross_val_score(
                model, X_train, y_train, cv=5, scoring="accuracy"
            )
            cv_mean = cv_scores.mean()
            cv_std = cv_scores.std()

            # For compatibility, set regression metrics to None
            train_r2 = test_r2 = train_rmse = test_rmse = train_mae = test_mae = None
        else:
            # Regression metrics
            train_r2 = r2_score(y_train, y_pred_train)
            test_r2 = r2_score(y_test, y_pred_test)
            train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
            test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
            train_mae = mean_absolute_error(y_train, y_pred_train)
            test_mae = mean_absolute_error(y_test, y_pred_test)

            # Cross-validation score
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="r2")
            cv_mean = cv_scores.mean()
            cv_std = cv_scores.std()

            # For compatibility, set classification metrics to None
            train_accuracy = test_accuracy = train_f1 = test_f1 = None

        # Store results
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

        # Add task-specific metrics
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

        # Save model and preprocessor
        self._save_model(model_key, model, preprocessor)

        # Print results
        print(
            f"\n📊 Training Results for {model_type} ({algorithm}) - {task_type.upper()}:"
        )
        if task_type == "classification":
            print(f"   Training Accuracy: {train_accuracy:.4f}")
            print(f"   Test Accuracy: {test_accuracy:.4f}")
            print(f"   Training F1: {train_f1:.4f}")
            print(f"   Test F1: {test_f1:.4f}")
            print(f"   Cross-validation Accuracy: {cv_mean:.4f} (±{cv_std:.4f})")
        else:
            print(f"   Training R²: {train_r2:.4f}")
            print(f"   Test R²: {test_r2:.4f}")
            print(f"   Training RMSE: {train_rmse:.4f}")
            print(f"   Test RMSE: {test_rmse:.4f}")
            print(f"   Cross-validation R²: {cv_mean:.4f} (±{cv_std:.4f})")
        print(f"   Training time: {training_time:.2f} seconds")

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

    def _train_with_hyperparameter_tuning(
        self, algorithm, X_train, y_train, model_type, task_type="regression"
    ):
        """Train model with hyperparameter tuning using GridSearchCV

        Args:
            algorithm (str): Algorithm name
            X_train (numpy.ndarray): Training features
            y_train (numpy.ndarray): Training targets
            model_type (str): Model type

        Returns:
            sklearn.base.BaseEstimator: Best trained model
        """
        print(f"🔍 Performing hyperparameter tuning for {algorithm}...")

        # Define parameter grids for different algorithms
        param_grids = {
            "random_forest": {
                "n_estimators": [50, 100, 200],
                "max_depth": [5, 10, 15, None],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
            },
            "gradient_boosting": {
                "n_estimators": [50, 100, 200],
                "learning_rate": [0.05, 0.1, 0.2],
                "max_depth": [3, 6, 9],
                "min_samples_split": [2, 5, 10],
            },
            "svr": {
                "C": [0.1, 1, 10],
                "gamma": ["scale", "auto"],
                "epsilon": [0.01, 0.1, 0.2],
            },
            "mlp": {
                "hidden_layer_sizes": [(50,), (100,), (100, 50)],
                "alpha": [0.0001, 0.001, 0.01],
                "learning_rate": ["constant", "adaptive"],
            },
        }

        if algorithm not in param_grids:
            print(
                f"⚠️ No hyperparameter grid defined for {algorithm}, using default parameters"
            )
            if task_type == "classification":
                return self._create_model(
                    algorithm, CLASSIFICATION_ALGORITHMS[algorithm], task_type
                )
            else:
                return self._create_model(
                    algorithm, MODEL_ALGORITHMS[algorithm], task_type
                )

        # Create base model
        if task_type == "classification":
            base_model = self._create_model(
                algorithm, CLASSIFICATION_ALGORITHMS[algorithm], task_type
            )
            scoring = "accuracy"
        else:
            base_model = self._create_model(
                algorithm, MODEL_ALGORITHMS[algorithm], task_type
            )
            scoring = "r2"

        # Perform grid search
        grid_search = GridSearchCV(
            base_model,
            param_grids[algorithm],
            cv=3,
            scoring=scoring,
            n_jobs=-1,
            verbose=1,
        )

        grid_search.fit(X_train, y_train)

        print(f"✅ Best parameters: {grid_search.best_params_}")
        print(f"✅ Best CV score: {grid_search.best_score_:.4f}")

        return grid_search.best_estimator_

    def _save_model(self, model_key, model, preprocessor):
        """Save trained model and preprocessor to file

        Args:
            model_key (str): Unique model identifier
            model: Trained model instance
            preprocessor: Fitted preprocessor
        """
        # Save model
        model_path = MODELS_DIR / f"{model_key}.joblib"
        joblib.dump(model, model_path)

        # Save preprocessor
        preprocessor_path = MODELS_DIR / f"{model_key}_preprocessor.joblib"
        joblib.dump(preprocessor, preprocessor_path)

        # Save training results
        results_path = MODELS_DIR / f"{model_key}_results.json"
        with open(results_path, "w") as f:
            json.dump(self.training_results[model_key], f, indent=2, default=str)

        print(f"💾 Model saved to {model_path}")
        print(f"💾 Preprocessor saved to {preprocessor_path}")
        print(f"💾 Results saved to {results_path}")

    def load_model(self, model_key):
        """Load a trained model from file

        Args:
            model_key (str): Model identifier (e.g., 'soil_moisture_predictor_random_forest')

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

        # Try to load from file
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

    def compare_models(self, model_type, algorithms=None):
        """Compare different algorithms for a specific model type (DEPRECATED)

        This method is deprecated as we now use only one algorithm per model type
        to reduce complexity and overfitting.

        Args:
            model_type (str): Type of model to compare
            algorithms (list): List of algorithms to compare

        Returns:
            dict: Comparison results (only for the default algorithm)
        """
        print("⚠️  compare_models is deprecated. Using default algorithm only.")
        default_algorithm = DEFAULT_ALGORITHMS.get(model_type, "gradient_boosting")
        result = self.train_model(model_type, default_algorithm)
        return {default_algorithm: result}

    def get_best_model(self, model_type):
        """Get the best performing model for a specific type

        Args:
            model_type (str): Type of model

        Returns:
            str: Best model key
        """
        models = self.list_trained_models()
        model_type_models = [
            m for m in models if m.get("model_type") == model_type and "error" not in m
        ]

        if not model_type_models:
            return None

        # Find model with highest test R² score
        best_model = max(model_type_models, key=lambda x: x.get("test_r2", -1))
        return f"{best_model['model_type']}_{best_model['algorithm']}"

    def evaluate_model(self, model_key, test_data):
        """Evaluate a model on new test data

        Args:
            model_key (str): Model identifier
            test_data (pandas.DataFrame): Test data

        Returns:
            dict: Evaluation results
        """
        try:
            model, preprocessor, _ = self.load_model(model_key)

            # Prepare test data
            if isinstance(test_data, dict):
                test_df = pd.DataFrame([test_data])
            else:
                test_df = test_data

            # Use the data processor to preprocess
            X_test = self.data_processor.preprocess_input(
                test_df.to_dict("records")[0],
                test_df.iloc[0].get("model_type", "soil_moisture_predictor"),
            )

            # Make predictions
            predictions = model.predict(X_test)

            # Calculate metrics if target is available
            results = {"predictions": predictions.tolist()}

            target_col = None
            for col in test_df.columns:
                if "moisture" in col.lower() or "irrigation" in col.lower():
                    target_col = col
                    break

            if target_col and target_col in test_df.columns:
                y_true = test_df[target_col].values
                results["r2"] = r2_score(y_true, predictions)
                results["rmse"] = np.sqrt(mean_squared_error(y_true, predictions))
                results["mae"] = mean_absolute_error(y_true, predictions)

            return results

        except Exception as e:
            return {"error": str(e)}
