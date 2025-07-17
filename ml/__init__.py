"""
Main interface for the ML Engine
 - easy-to-use functions for training and prediction
"""

from typing import Literal
import pandas as pd

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

    def train_soil_moisture_predictor(
        self, custom_data: pd.DataFrame | None = None, version: int | None = None
    ) -> dict:
        """Train soil moisture prediction model

        Args:
            custom_data (pandas.DataFrame, optional): Custom training data

        Returns:
            dict: Training results
        """
        return self.trainer.train_model(
            "soil_moisture_predictor", custom_data=custom_data, version=version
        )

    def train_model(
        self,
        model_type: Literal["soil_moisture_predictor", "irrigation_recommendation"],
        custom_data: pd.DataFrame | None = None,
        version: int | None = None,
        algorithm: str | None = None,
    ) -> dict:
        """Train a model

        Args:
            model_type (Literal["soil_moisture_predictor", "irrigation_recommendation"]): Type of model to train
            custom_data (pandas.DataFrame, optional): Custom training data
            version (int, optional): Version of the model
        """
        assert model_type in [
            "soil_moisture_predictor",
            "irrigation_recommendation",
        ], "Invalid model type"
        return self.trainer.train_model(
            model_type,
            custom_data=custom_data,
            version=version,
            algorithm=algorithm,
        )

    def predict(
        self,
        model_type: Literal["soil_moisture_predictor", "irrigation_recommendation"],
        data: dict,
        version: int | None = None,
    ) -> dict:
        """Predict a model

        Args:
            model_type (Literal["soil_moisture_predictor", "irrigation_recommendation"]): Type of model to predict
            data (dict): Data to predict
            version (int, optional): Version of the model
        """
        assert model_type in [
            "soil_moisture_predictor",
            "irrigation_recommendation",
        ], "Invalid model type"
        return self.predictor.predict(model_type, data, version)

    def train_irrigation_recommender(
        self, custom_data: pd.DataFrame | None = None, version: int | None = None
    ) -> dict:
        """Train irrigation recommendation model

        Args:
            custom_data (pandas.DataFrame, optional): Custom training data

        Returns:
            dict: Training results
        """
        return self.trainer.train_model(
            "irrigation_recommendation", custom_data=custom_data, version=version
        )

    def train_all_models(self, custom_data: pd.DataFrame | None = None) -> dict:
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
    ) -> dict:
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

    def get_available_models(self) -> list[str]:
        """Get list of available trained models

        Returns:
            list: List of available model types
        """
        return self.predictor.get_available_models()

    def get_model_info(self, model_type: str) -> dict:
        """Get information about a trained model

        Args:
            model_type (str): Type of model

        Returns:
            dict: Model information
        """
        return self.trainer.get_model_info(model_type)

    def list_all_models(self) -> list[dict]:
        """List all trained models with their information

        Returns:
            list: List of model information dictionaries
        """
        return self.trainer.list_trained_models()

    def retrain_model(
        self, model_type: str, new_data: pd.DataFrame | None = None
    ) -> dict:
        """Retrain an existing model

        Args:
            model_type (str): Type of model to retrain
            new_data (pandas.DataFrame, optional): New training data

        Returns:
            dict: Updated training results
        """
        return self.trainer.train_model(model_type, custom_data=new_data)

    def save_training_data(self, data: pd.DataFrame, model_type: str):
        """Save training data to file

        Args:
            data (pandas.DataFrame): Data to save
            model_type (str): Type of model
        """
        return self.data_processor.save_training_data(data, model_type)

    def load_training_data(self, model_type: str) -> pd.DataFrame:
        """Load training data from file

        Args:
            model_type (str): Type of model

        Returns:
            pandas.DataFrame: Training data
        """
        return self.data_processor.load_training_data(model_type)

    def train_model_on_all_algorithms(
        self, model_type: str, custom_data: pd.DataFrame | None = None
    ) -> dict:
        """Train a model on all available algorithms

        Args:
            model_type (str): Type of model to train
            custom_data (pandas.DataFrame, optional): Custom training data

        Returns:
            dict: Dictionary of training results for all algorithms
        """
        results = {}

        for algorithm in self.trainer.get_available_algorithms(model_type):
            results[algorithm] = self.trainer.train_model(
                model_type, algorithm, custom_data
            )

        return results

    def predict_model_on_all_algorithms(self, model_type: str, data: dict) -> dict:
        """Predict a model on all available algorithms

        Args:
            model_type (str): Type of model to predict
            data (dict): Data to predict

        Returns:
            dict: Dictionary of predictions for all algorithms
        """
        results = {}
        for algorithm in self.trainer.get_available_algorithms(model_type):
            prediction = self.predictor.predict(model_type, data, algorithm)
            del prediction["model_info"]
            results[algorithm] = prediction

        return results


def get_available_models() -> list[str]:
    """Quick function to get available models

    Returns:
        list: List of available model types
    """
    engine = MLEngine()
    return engine.get_available_models()
