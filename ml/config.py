# Configuration for ML model directories and model feature/target definitions
from pathlib import Path

# Base directory for the project
BASE_DIR = Path(__file__).resolve().parent.parent

# Model and data storage paths
MODELS_DIR = BASE_DIR / "ml" / "models"
DATA_DIR = BASE_DIR / "ml" / "data"

MODEL_CONFIGS = {
    "soil_moisture_predictor": {
        "features": [
            "temperature_celsius",
            "humidity_percent",
            "battery_voltage",
            "hour_of_day",
            "month",
            "is_growing_season",
            "temp_humidity_interaction",
            "low_battery",
            "status",
            "irrigation_action",
        ],
        "target": "soil_moisture_percent",
        "task_type": "regression",
    },
    "irrigation_recommendation": {
        "features": [
            "soil_moisture_percent",
            "temperature_celsius",
            "humidity_percent",
            "battery_voltage",
            "hour_of_day",
            "month",
            "is_growing_season",
            "temp_humidity_interaction",
            "low_battery",
            "status",
        ],
        "target": "irrigation_action",
        "task_type": "classification",
    },
}

REGRESSION_ALGORITHMS = {
    "random_forest": {
        "n_estimators": 100,
        "max_depth": 10,
        "min_samples_split": 5,
        "min_samples_leaf": 2,
        "random_state": 42,
        "n_jobs": -1,
    },
    "gradient_boosting": {
        "n_estimators": 100,
        "learning_rate": 0.1,
        "max_depth": 6,
        "min_samples_split": 5,
        "min_samples_leaf": 2,
        "random_state": 42,
    },
    "svr": {"kernel": "rbf", "C": 1.0, "gamma": "scale", "epsilon": 0.1},
    "mlp": {
        "hidden_layer_sizes": (100, 50),
        "activation": "relu",
        "solver": "adam",
        "alpha": 0.0001,
        "learning_rate": "adaptive",
        "max_iter": 500,
        "random_state": 42,
    },
    "linear_regression": {
        "fit_intercept": True,
    },
}

CLASSIFICATION_ALGORITHMS = {
    "random_forest": {
        "n_estimators": 100,
        "max_depth": 10,
        "min_samples_split": 5,
        "min_samples_leaf": 2,
        "random_state": 42,
        "n_jobs": -1,
    },
    "gradient_boosting": {
        "n_estimators": 100,
        "learning_rate": 0.1,
        "max_depth": 6,
        "min_samples_split": 5,
        "min_samples_leaf": 2,
        "random_state": 42,
    },
    "logistic_regression": {
        "C": 1.0,
        "max_iter": 1000,
        "random_state": 42,
    },
}

DEFAULT_ALGORITHMS = {
    "soil_moisture_predictor": "random_forest",
    "irrigation_recommendation": "random_forest",
}

# Training parameters
TRAINING_CONFIG = {
    "test_size": 0.2,
    "random_state": 42,
    "n_splits": 5,
    "scoring": "neg_mean_squared_error",
    "shuffle": True,
    "early_stopping": True,
}

# Prediction settings
PREDICTION_CONFIG = {
    "confidence_threshold": 0.7,
    "default_confidence_level": 0.95,
    "output_format": "dict",
    "max_batch_size": 128,
    "round_predictions": 2,
}
