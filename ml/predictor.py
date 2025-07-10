import joblib
import numpy as np
import time
import json
from pathlib import Path

from .config import MODELS_DIR, PREDICTION_CONFIG
from .data_processor import DataProcessor


class Predictor:
    """Handles regression model predictions"""

    def __init__(self):
        """Initialize Predictor with data processor and empty models container"""
        self.data_processor = DataProcessor()
        self.models = {}

    def predict(self, model_type, input_data):
        """Make prediction using trained regression model

        Args:
            model_type (str): Type of model to use for prediction
            input_data (dict): Input data dictionary

        Returns:
            dict: Prediction results including predicted value, confidence interval, and uncertainty
        """
        # Load the model
        model = self._load_model(model_type)
        if model is None:
            return {"success": False, "message": f"Model '{model_type}' not found."}

        # Preprocess the input data
        try:
            X = self.data_processor.preprocess_input(input_data, model_type)
        except Exception as e:
            return {"success": False, "message": f"Input preprocessing failed: {str(e)}"}

        # Make prediction
        try:
            prediction = model.predict(X)
        except Exception as e:
            return {"success": False, "message": f"Prediction failed: {str(e)}"}

        # Calculate confidence interval (optional)
        try:
            lower, upper = self.calculate_confidence_interval(model, X)
        except Exception:
            lower, upper = None, None

        result = {
            "success": True,
            "predicted_value": float(prediction[0]) if hasattr(prediction, '__getitem__') else float(prediction),
            "confidence_interval": (float(lower), float(upper)) if lower is not None and upper is not None else None,
            "uncertainty": float(upper - lower) if lower is not None and upper is not None else None
        }
        return result


    def _load_model(self, model_type):
        """Load a trained model from file

        Args:
            model_type (str): Type of model to load
        """
        if model_type in self.models:
            return self.models[model_type]
        model_path = MODELS_DIR / f"{model_type}.joblib"
        if not model_path.exists():
            return None
        try:
            model = joblib.load(model_path)
            self.models[model_type] = model
            return model
        except Exception as e:
            print(f"Failed to load model '{model_type}': {e}")
            return None

    def predict_multiple(self, input_data):
        """Make predictions using all available models

        Args:
            input_data (dict): Input data dictionary

        Returns:
            dict: Dictionary of predictions from all available models
        """
        results = {}
        model_types = self.get_available_models()
        for model_type in model_types:
            results[model_type] = self.predict(model_type, input_data)
        return results

    def get_available_models(self):
        """Get list of available trained models

        Returns:
            list: List of available model types
        """
        if not MODELS_DIR.exists():
            return []
        model_files = [f for f in MODELS_DIR.iterdir() if f.is_file() and f.suffix == '.joblib']
        model_types = [f.stem for f in model_files]
        return model_types

    def validate_input(self, model_type, input_data):
        """Validate input data for a specific model

        Args:
            model_type (str): Type of model
            input_data (dict): Input data to validate

        Returns:
            dict: Validation result with 'valid' boolean and 'message' string
        """
        from .config import MODEL_CONFIGS
        if model_type not in MODEL_CONFIGS:
            return {"valid": False, "message": f"Unknown model type: {model_type}"}
        required_features = MODEL_CONFIGS[model_type].get("features", [])
        missing = [f for f in required_features if f not in input_data]
        if missing:
            return {
                "valid": False,
                "message": f"Missing required features: {', '.join(missing)}"
            }
        return {"valid": True, "message": "All required features are present."}

    def calculate_confidence_interval(self, model, X, confidence_level=0.95):
        """Calculate confidence interval for prediction

        Args:
            model: Trained regression model
            X (numpy.ndarray): Input features
            confidence_level (float): Confidence level

        Returns:
            tuple: (lower_bound, upper_bound) confidence interval
        """
        import scipy.stats
        # RandomForestRegressor: use predictions from all estimators
        if hasattr(model, "estimators_"):
            all_preds = np.array([est.predict(X)[0] for est in model.estimators_])
            mean_pred = np.mean(all_preds)
            std_pred = np.std(all_preds)
            z = scipy.stats.norm.ppf(1 - (1 - confidence_level) / 2)
            lower = mean_pred - z * std_pred
            upper = mean_pred + z * std_pred
            return lower, upper
        # For other models, return None (not implemented)
        return None, None


class SoilMoisturePredictor:
    """Specialized predictor for soil moisture level prediction"""

    def __init__(self):
        """Initialize SoilMoisturePredictor with a Predictor instance"""
        self.predictor = Predictor()

    def predict_moisture(
        self, location, status, temperature_celsius, humidity_percent, battery_voltage
    ):
    #  temperature (float): Temperature in Celsius
    #         ph_level (float): Soil pH level
    #         humidity (float): Air humidity percentage
    #         rainfall (float): Rainfall amount in mm
    #         previous_moisture (float): Previous moisture level percentage
        """Predict soil moisture level based on environmental factors

        Args:
            location (str): Location identifier
            status (str): Sensor status
            temperature_celsius (float): Temperature in Celsius
            humidity_percent (float): Air humidity percentage
            battery_voltage (float): Sensor battery voltage

        Returns:
            dict: Soil moisture prediction result with value and confidence interval
        """
        input_data = {
            "location": location,
            "status": status,
            "temperature_celsius": temperature_celsius,
            "humidity_percent": humidity_percent,
            "battery_voltage": battery_voltage,
        }
        return self.predictor.predict("soil_moisture_predictor", input_data)


class IrrigationRecommender:
    """Specialized predictor for irrigation recommendations"""

    def __init__(self):
        """Initialize IrrigationRecommender with a Predictor instance"""
        self.predictor = Predictor()

    def recommend_irrigation(
        self, moisture_level, temperature, humidity, rainfall, crop_type, growth_stage
    ):
        """Recommend irrigation amount based on current conditions

        Args:
            moisture_level (float): Current soil moisture percentage
            temperature (float): Temperature in Celsius
            humidity (float): Air humidity percentage
            rainfall (float): Rainfall amount in mm
            crop_type (str): Type of crop (e.g., 'maize', 'beans', 'rice')
            growth_stage (str): Growth stage (e.g., 'seedling', 'vegetative', 'flowering', 'mature')

        Returns:
            dict: Irrigation recommendation with amount and confidence interval
        """
        input_data = {
            
        }
        


class MoistureForecaster:
    """Specialized predictor for moisture level forecasting"""

    def __init__(self):
        """Initialize MoistureForecaster with a Predictor instance"""
        self.predictor = Predictor()

    def forecast_moisture(
        self,
        current_moisture,
        temperature,
        humidity,
        rainfall_forecast,
        evaporation_rate,
        days_ahead=7,
    ):
        """Forecast future moisture levels

        Args:
            current_moisture (float): Current soil moisture percentage
            temperature (float): Temperature in Celsius
            humidity (float): Air humidity percentage
            rainfall_forecast (float): Forecasted rainfall amount
            evaporation_rate (float): Evaporation rate in mm/day
            days_ahead (int, optional): Number of days to forecast. Defaults to 7.

        Returns:
            dict: Moisture forecast with predicted values and confidence intervals
        """
        pass
