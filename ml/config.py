# Configuration for ML model directories and model feature/target definitions
from pathlib import Path

# Base directory for the project
BASE_DIR = Path(__file__).resolve().parent.parent

# Model and data storage paths
MODELS_DIR = BASE_DIR / "ml"
DATA_DIR = BASE_DIR / "ml" / "data"

# Model configurations - define structure for different prediction models
# Each model should specify its input features and target variable
MODEL_CONFIGS = {
    "soil_moisture_predictor": {
        "features": [
            "location",
            "status",
            "temperature_celsius",
            "humidity_percent",
            "battery_voltage",
        ],
        "target": "soil_moisture_percent",
    },
    "irrigation_recommendation": {
        "features": [
            "moisture_level",
            "temperature",
            "humidity",
            "rainfall",
            "crop_type",
            "growth_stage",
        ],
        "target": "irrigation_amount",
    },
   
}
