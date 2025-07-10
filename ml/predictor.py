import joblib
import numpy as np
import json
from pathlib import Path

# Handle imports for both direct execution and module import
try:
    from .config import MODELS_DIR, DEFAULT_ALGORITHMS
    from .data_processor import DataProcessor
except ImportError:
    from config import MODELS_DIR, DEFAULT_ALGORITHMS
    from data_processor import DataProcessor


class Predictor:
    """Handles regression model predictions"""

    def __init__(self):
        """Initialize Predictor with data processor and empty models container"""
        self.data_processor = DataProcessor()
        self.models = {}
        self.preprocessors = {}

    def predict(self, model_type, input_data, algorithm=None):
        """Make prediction using trained regression model

        Args:
            model_type (str): Type of model to use for prediction (e.g., 'soil_moisture_predictor')
            input_data (dict): Input data dictionary
            algorithm (str, optional): Specific algorithm to use (e.g., 'random_forest')

        Returns:
            dict: Prediction results including predicted value, confidence interval, and uncertainty
        """
        # Determine model key
        if algorithm is None:
            algorithm = DEFAULT_ALGORITHMS.get(model_type, "random_forest")

        model_key = f"{model_type}_{algorithm}"

        # Load the model and preprocessor
        model, preprocessor, model_info = self._load_model_and_preprocessor(model_key)
        if model is None:
            return {"success": False, "message": f"Model '{model_key}' not found."}

        # Preprocess the input data
        # try:
        self.data_processor.load_encoders(model_type)
        X = self.data_processor.preprocess_input(input_data, model_type)
        # except Exception as e:
        #     return {
        #         "success": False,
        #         "message": f"Input preprocessing failed: {str(e)}",
        #     }

        # Make prediction
        try:
            prediction = model.predict(X)

            # Handle different prediction types based on task type
            task_type = (
                model_info.get("task_type", "regression")
                if model_info
                else "regression"
            )

            if task_type == "classification":
                # For classification, return the predicted class label
                prediction_value = (
                    prediction[0] if hasattr(prediction, "__getitem__") else prediction
                )
                # Also get prediction probabilities if available
                try:
                    prediction_proba = model.predict_proba(X)
                    class_probabilities = (
                        prediction_proba[0]
                        if hasattr(prediction_proba, "__getitem__")
                        else prediction_proba
                    )
                except:
                    class_probabilities = None
            else:
                # For regression, convert to float
                prediction_value = (
                    float(prediction[0])
                    if hasattr(prediction, "__getitem__")
                    else float(prediction)
                )
                class_probabilities = None

        except Exception as e:
            return {"success": False, "message": f"Prediction failed: {str(e)}"}

        # Calculate confidence interval (optional)
        try:
            lower, upper = self.calculate_confidence_interval(model, X)
            confidence_interval = (
                (float(lower), float(upper))
                if lower is not None and upper is not None
                else None
            )
            uncertainty = (
                float(upper - lower)
                if lower is not None and upper is not None
                else None
            )
        except Exception:
            confidence_interval = None
            uncertainty = None

        result = {
            "success": True,
            "model_type": model_type,
            "algorithm": algorithm,
            "predicted_value": prediction_value,
            "confidence_interval": confidence_interval,
            "uncertainty": uncertainty,
            "model_info": model_info,
        }

        # Add classification-specific information
        if task_type == "classification" and class_probabilities is not None:
            result["class_probabilities"] = (
                class_probabilities.tolist()
                if hasattr(class_probabilities, "tolist")
                else class_probabilities
            )
        return result

    def _load_model_and_preprocessor(self, model_key):
        """Load a trained model and preprocessor from file

        Args:
            model_key (str): Model identifier (e.g., 'soil_moisture_predictor_random_forest')

        Returns:
            tuple: (model, preprocessor, model_info)
        """
        if model_key in self.models:
            return self.models[model_key], self.preprocessors.get(model_key), None

        model_path = MODELS_DIR / f"{model_key}.joblib"
        preprocessor_path = MODELS_DIR / f"{model_key}_preprocessor.joblib"
        results_path = MODELS_DIR / f"{model_key}_results.json"

        if not model_path.exists():
            return None, None, None

        try:
            model = joblib.load(model_path)
            preprocessor = (
                joblib.load(preprocessor_path) if preprocessor_path.exists() else None
            )

            model_info = {}
            if results_path.exists():
                with open(results_path, "r") as f:
                    model_info = json.load(f)

            # Cache the model and preprocessor
            self.models[model_key] = model
            if preprocessor:
                self.preprocessors[model_key] = preprocessor

            return model, preprocessor, model_info
        except Exception as e:
            print(f"Failed to load model '{model_key}': {e}")
            return None, None, None

    def predict_multiple(self, input_data, model_type=None):
        """Make predictions using all available models or specific model type

        Args:
            input_data (dict): Input data dictionary
            model_type (str, optional): Specific model type to use

        Returns:
            dict: Dictionary of predictions from available models
        """
        results = {}

        if model_type:
            # Predict with specific model type using default algorithm
            algorithm = DEFAULT_ALGORITHMS.get(model_type, "random_forest")
            model_key = f"{model_type}_{algorithm}"
            results[model_key] = self.predict(model_type, input_data, algorithm)
        else:
            # Predict with all available models
            available_models = self.get_available_models()
            for model_key in available_models:
                # Extract model_type and algorithm from model_key
                parts = model_key.split("_", 1)
                if len(parts) == 2:
                    model_type, algorithm = parts
                    results[model_key] = self.predict(model_type, input_data, algorithm)

        return results

    def get_available_models(self):
        """Get list of available trained models

        Returns:
            list: List of available model keys
        """
        if not MODELS_DIR.exists():
            return []

        model_files = [
            f
            for f in MODELS_DIR.iterdir()
            if f.is_file()
            and f.suffix == ".joblib"
            and not f.name.endswith("_preprocessor.joblib")
        ]
        model_keys = [f.stem for f in model_files]
        return model_keys

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

        # Get the features that would be used after feature engineering
        # This is a simplified validation - the actual validation happens in preprocess_input
        required_features = MODEL_CONFIGS[model_type].get("features", [])

        # Check for basic required features (before engineering)
        basic_features = [
            "sensor_id",
            "location",
            "temperature_celsius",
            "humidity_percent",
            "battery_voltage",
            "status",
            "irrigation_action",
            "timestamp",
        ]
        missing = [
            f for f in basic_features if f not in input_data and f in required_features
        ]

        if missing:
            return {
                "valid": False,
                "message": f"Missing required features: {', '.join(missing)}",
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

        # GradientBoostingRegressor: use predictions from all estimators
        elif hasattr(model, "estimators_"):
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
        self,
        sensor_id,
        location,
        temperature_celsius,
        humidity_percent,
        battery_voltage,
        status,
        irrigation_action,
        timestamp,
        algorithm=None,
    ):
        """Predict soil moisture level based on environmental factors

        Args:
            sensor_id (str): Sensor identifier
            location (str): Location identifier
            temperature_celsius (float): Temperature in Celsius
            humidity_percent (float): Air humidity percentage
            battery_voltage (float): Sensor battery voltage
            status (str): Sensor status
            irrigation_action (str): Irrigation action
            timestamp (str): Timestamp
            algorithm (str, optional): Specific algorithm to use

        Returns:
            dict: Soil moisture prediction result with value and confidence interval
        """
        input_data = {
            "sensor_id": sensor_id,
            "location": location,
            "temperature_celsius": temperature_celsius,
            "humidity_percent": humidity_percent,
            "battery_voltage": battery_voltage,
            "status": status,
            "irrigation_action": irrigation_action,
            "timestamp": timestamp,
        }
        return self.predictor.predict("soil_moisture_predictor", input_data, algorithm)


class IrrigationRecommender:
    """Specialized predictor for irrigation recommendations"""

    def __init__(self):
        """Initialize IrrigationRecommender with a Predictor instance"""
        self.predictor = Predictor()

    def recommend_irrigation(
        self,
        soil_moisture_percent,
        temperature_celsius,
        humidity_percent,
        battery_voltage=3.8,
        status="Normal",
        timestamp=None,
        algorithm=None,
    ):
        """Recommend irrigation action based on sensor data

        Args:
            soil_moisture_percent (float): Current soil moisture percentage
            temperature_celsius (float): Temperature in Celsius
            humidity_percent (float): Air humidity percentage
            battery_voltage (float): Sensor battery voltage
            status (str): Sensor status
            timestamp (str, optional): Timestamp (if None, uses current time)
            algorithm (str, optional): Specific algorithm to use

        Returns:
            dict: Irrigation recommendation result
        """
        import datetime

        if timestamp is None:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        input_data = {
            "soil_moisture_percent": soil_moisture_percent,
            "temperature_celsius": temperature_celsius,
            "humidity_percent": humidity_percent,
            "battery_voltage": battery_voltage,
            "status": status,
            "timestamp": timestamp,
        }
        return self.predictor.predict(
            "irrigation_recommendation", input_data, algorithm
        )
