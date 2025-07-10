"""
Main interface for the ML Engine
This file provides easy-to-use functions for training and prediction
"""

from .model_trainer import ModelTrainer
from .predictor import (
    Predictor,
    SoilMoisturePredictor,
    IrrigationRecommender,
)
from .data_processor import DataProcessor


class MLEngine:
    """Main interface for the ML Engine"""

    def __init__(self):
        """Initialize MLEngine with trainer, predictor, and data processor instances"""
        self.trainer = ModelTrainer()
        self.predictor = Predictor()
        self.data_processor = DataProcessor()

    # Training functions
    def train_soil_moisture_predictor(self, custom_data=None):
        """Train soil moisture prediction model

        Args:
            custom_data (pandas.DataFrame, optional): Custom training data

        Returns:
            dict: Training results
        """
        return self.trainer.train_model(
            "soil_moisture_predictor", custom_data=custom_data
        )

    def train_irrigation_recommender(self, custom_data=None):
        """Train irrigation recommendation model

        Args:
            custom_data (pandas.DataFrame, optional): Custom training data

        Returns:
            dict: Training results
        """
        return self.trainer.train_model(
            "irrigation_recommendation", custom_data=custom_data
        )

    def train_all_models(self, custom_data=None):
        """Train all available prediction models

        Args:
            custom_data (pandas.DataFrame, optional): Custom training data

        Returns:
            dict: Dictionary of training results for all models
        """
        results = {
            "soil_moisture_predictor": self.train_soil_moisture_predictor(custom_data),
            "irrigation_recommendation": self.train_irrigation_recommender(custom_data),
        }
        return results

    # Prediction functions
    def predict_soil_moisture(
        self,
        sensor_id,
        location,
        temperature_celsius,
        humidity_percent,
        battery_voltage,
        status,
        irrigation_action,
        timestamp,
    ):
        """Predict soil moisture level

        Args:
            sensor_id (str): Sensor identifier
            location (str): Location identifier
            temperature_celsius (float): Temperature in Celsius
            humidity_percent (float): Air humidity percentage
            battery_voltage (float): Sensor battery voltage
            status (str): Sensor status
            irrigation_action (str): Irrigation action
            timestamp (str): Timestamp

        Returns:
            dict: Soil moisture prediction result
        """
        predictor = SoilMoisturePredictor()

        return predictor.predict_moisture(
            sensor_id,
            location,
            temperature_celsius,
            humidity_percent,
            battery_voltage,
            status,
            irrigation_action,
            timestamp,
        )

    def recommend_irrigation(
        self,
        soil_moisture_percent,
        temperature_celsius,
        humidity_percent,
        battery_voltage=3.8,
        status="Normal",
        timestamp=None,
    ):
        """Recommend irrigation action

        Args:
            soil_moisture_percent (float): Current soil moisture percentage
            temperature_celsius (float): Temperature in Celsius
            humidity_percent (float): Air humidity percentage
            battery_voltage (float): Sensor battery voltage
            status (str): Sensor status
            timestamp (str, optional): Timestamp (if None, uses current time)

        Returns:
            dict: Irrigation recommendation result
        """
        recommender = IrrigationRecommender()
        return recommender.recommend_irrigation(
            soil_moisture_percent,
            temperature_celsius,
            humidity_percent,
            battery_voltage,
            status,
            timestamp,
        )

    def predict_all(self, input_data):
        """Make predictions using all available models

        Args:
            input_data (dict): Input data dictionary

        Returns:
            dict: Dictionary of predictions from all models
        """
        return self.predictor.predict_multiple(input_data)

    # Utility functions
    def get_available_models(self):
        """Get list of available trained models

        Returns:
            list: List of available model types
        """
        return self.predictor.get_available_models()

    def get_model_info(self, model_type):
        """Get information about a trained model

        Args:
            model_type (str): Type of model

        Returns:
            dict: Model information
        """
        return self.trainer.get_model_info(model_type)

    def list_all_models(self):
        """List all trained models with their information

        Returns:
            list: List of model information dictionaries
        """
        return self.trainer.list_trained_models()

    def retrain_model(self, model_type, new_data=None):
        """Retrain an existing model

        Args:
            model_type (str): Type of model to retrain
            new_data (pandas.DataFrame, optional): New training data

        Returns:
            dict: Updated training results
        """
        return self.trainer.train_model(model_type, custom_data=new_data)

    def save_training_data(self, data, model_type):
        """Save training data to file

        Args:
            data (pandas.DataFrame): Data to save
            model_type (str): Type of model
        """
        return self.data_processor.save_training_data(data, model_type)

    def load_training_data(self, model_type):
        """Load training data from file

        Args:
            model_type (str): Type of model

        Returns:
            pandas.DataFrame: Training data
        """
        return self.data_processor.load_training_data(model_type)


# Convenience functions for quick access
def train_model(model_type, custom_data=None):
    """Quick function to train a model

    Args:
        model_type (str): Type of model to train
        custom_data (pandas.DataFrame, optional): Custom training data

    Returns:
        dict: Training results
    """
    engine = MLEngine()
    train_methods = {
        "soil_moisture_predictor": engine.train_soil_moisture_predictor,
        "irrigation_recommendation": engine.train_irrigation_recommender,
    }
    if model_type not in train_methods:
        raise ValueError(f"Unknown model type: {model_type}")
    return train_methods[model_type](custom_data=custom_data)


def predict_soil_moisture(
    sensor_id,
    location,
    temperature_celsius,
    humidity_percent,
    battery_voltage,
    status,
    irrigation_action,
    timestamp,
):
    """Quick function to predict soil moisture level

    Args:
        sensor_id (str): Sensor identifier
        location (str): Location identifier
        temperature_celsius (float): Temperature in Celsius
        humidity_percent (float): Air humidity percentage
        battery_voltage (float): Sensor battery voltage
        status (str): Sensor status
        irrigation_action (str): Irrigation action
        timestamp (str): Timestamp

    Returns:
        dict: Soil moisture prediction result
    """
    engine = MLEngine()
    return engine.predict_soil_moisture(
        sensor_id,
        location,
        temperature_celsius,
        humidity_percent,
        battery_voltage,
        status,
        irrigation_action,
        timestamp,
    )


def recommend_irrigation(
    soil_moisture_percent,
    temperature_celsius,
    humidity_percent,
    battery_voltage=3.8,
    status="Normal",
    timestamp=None,
):
    """Quick function to recommend irrigation action

    Args:
        soil_moisture_percent (float): Current soil moisture percentage
        temperature_celsius (float): Temperature in Celsius
        humidity_percent (float): Air humidity percentage
        battery_voltage (float): Sensor battery voltage
        status (str): Sensor status
        timestamp (str, optional): Timestamp (if None, uses current time)

    Returns:
        dict: Irrigation recommendation result
    """
    engine = MLEngine()
    return engine.recommend_irrigation(
        soil_moisture_percent,
        temperature_celsius,
        humidity_percent,
        battery_voltage,
        status,
        timestamp,
    )


def get_available_models():
    """Quick function to get available models

    Returns:
        list: List of available model types
    """
    engine = MLEngine()
    return engine.get_available_models()
