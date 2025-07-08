from pathlib import Path

_all_ = [
    "BASE_DIR",
    "MODELS_DIR",
    "DATA_DIR",
    "MODEL_CONFIGS",
    "TRAINING_CONFIG",
    "PREDICTION_CONFIG",
]
# Base directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Model storage paths
MODELS_DIR = BASE_DIR / "ml"
DATA_DIR = BASE_DIR / "ml" / "data"

# Model configurations - define structure for different prediction models
MODEL_CONFIGS = {
    "soil_moisture_predictor": {
        "features": [],
        "target": "",
    },
    "irrigation_recommendation": {
        "features": [],
        "target": "",
    },
    "moisture_forecast": {
        "features": [],
        "target": "",
    },
}

# Training parameters
TRAINING_CONFIG = {
    "test_size": 0.2,
}

# Prediction settings
PREDICTION_CONFIG = {
    "confidence_threshold": 0.7,
}
