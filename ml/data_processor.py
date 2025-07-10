import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import json
import os
from .config import DATA_DIR, TRAINING_CONFIG, MODEL_CONFIGS


class DataProcessor:
    """Handles data loading, preprocessing, and preparation for ML prediction models"""

    def __init__(self):
        """Initialize DataProcessor with empty containers for scalers, encoders, and feature columns"""
        self.scalers = {}
        self.label_encoders = {}
        self.feature_columns = {}

    def load_training_data(self, model_type):
        """Load training data from CSV or JSON files

        Args:
            model_type (str): Type of model (e.g., 'soil_moisture_predictor', 'irrigation_recommendation')

        Returns:
            pandas.DataFrame: Training data
        """
        data_path = DATA_DIR / f"{model_type}.csv"
        if data_path.exists():
            return pd.read_csv(data_path)
        else:
            raise FileNotFoundError(f"Training data for {model_type} not found at {data_path}")

    def _generate_sample_data(self, model_type):
        """Generate sample training data for testing

        Args:
            model_type (str): Type of model to generate data for

        Returns:
            pandas.DataFrame: Sample training data
        """
        # Generate sample data for each supported model type
        if model_type == "soil_moisture_predictor":
            n_samples = 100
            data = pd.DataFrame({
                "location": np.random.choice(["A", "B", "C"], n_samples),
                "status": np.random.choice(["active", "inactive"], n_samples),
                "temperature_celsius": np.random.normal(25, 5, n_samples),
                "humidity_percent": np.random.uniform(30, 90, n_samples),
                "battery_voltage": np.random.uniform(3.5, 4.2, n_samples),
                "soil_moisture_percent": np.random.uniform(10, 45, n_samples),
            })
        elif model_type == "irrigation_recommendation":
            n_samples = 100
            data = pd.DataFrame({
                "moisture_level": np.random.uniform(10, 45, n_samples),
                "temperature": np.random.normal(25, 5, n_samples),
                "humidity": np.random.uniform(30, 90, n_samples),
                "rainfall": np.random.uniform(0, 20, n_samples),
                "crop_type": np.random.choice(["wheat", "corn", "rice"], n_samples),
                "growth_stage": np.random.choice(["seedling", "vegetative", "flowering", "maturity"], n_samples),
                "irrigation_amount": np.random.uniform(0, 15, n_samples),
            })
        elif model_type == "moisture_forecast":
            n_samples = 100
            data = pd.DataFrame({
                "current_moisture": np.random.uniform(10, 45, n_samples),
                "temperature": np.random.normal(25, 5, n_samples),
                "humidity": np.random.uniform(30, 90, n_samples),
                "rainfall_forecast": np.random.uniform(0, 20, n_samples),
                "evaporation_rate": np.random.uniform(2, 8, n_samples),
                "days_ahead": np.random.randint(1, 15, n_samples),
                "forecasted_moisture": np.random.uniform(10, 45, n_samples),
            })
        else:
            raise ValueError(f"Unknown model_type '{model_type}' for sample data generation.")
        return data

    def prepare_data(self, data, model_type, features, target):
        """Prepare data for training regression models

        Args:
            data (pandas.DataFrame): Raw training data
            model_type (str): Type of model
            features (list): List of feature column names
            target (str): Target column name (continuous value)

        Returns:
            tuple: (X_train_scaled, X_test_scaled, y_train, y_test, feature_names)
        """
        data = self.load_training_data(model_type)
        df =pd.DataFrame(data)
        x_train, x_test, y_train, y_test = train_test_split(
            df[features],
            df[target],
            test_size=0.2,
            random_state=42
        )
        # Scale features
        self.scalers[model_type] = StandardScaler()
        self.label_encoders[model_type] = LabelEncoder()
        x_train_scaled = self.scalers[model_type].fit_transform(x_train)
        x_test_scaled = self.scalers[model_type].transform(x_test)
        return (
            x_train_scaled,
            x_test_scaled,
            y_train.values,
            y_test.values,
            features
        )
      

    def save_training_data(self, data, model_type):
        """Save training data to file

        Args:
            data (pandas.DataFrame): Data to save
            model_type (str): Type of model
        """
        data_path = DATA_DIR / f"{model_type}.csv"
        data.to_csv(data_path, index=False)
        print(f"Training data saved to {data_path}")
        return data_path
        
        

    def load_encoders(self, model_type):
        """Load saved encoders and scalers from file

        Args:
            model_type (str): Type of model
        """
        encoder_path = DATA_DIR / f"{model_type}_encoders.pkl"
        scaler_path = DATA_DIR / f"{model_type}_scaler.pkl"
        with open(encoder_path, 'rb') as f:
            self.label_encoders[model_type] = pickle.load(f)
        with open(scaler_path, 'rb') as f:
            self.scalers[model_type] = pickle.load(f)
        return self.label_encoders[model_type], self.scalers[model_type]

    def save_encoders(self, model_type):
        """Save encoders and scalers to file

        Args:
            model_type (str): Type of model
        """
        encoder_path = DATA_DIR / f"{model_type}_encoders.pkl"
        scaler_path = DATA_DIR / f"{model_type}_scaler.pkl"
        with open(encoder_path, 'wb') as f:
            pickle.dump(self.label_encoders[model_type], f)
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scalers[model_type], f)
        print(f"Encoders and scalers saved to {encoder_path} and {scaler_path}")
        return encoder_path, scaler_path

    def preprocess_input(self, input_data, model_type):
        """Preprocess input data for prediction

        Args:
            input_data (dict): Input data dictionary
            model_type (str): Type of model

        Returns:
            numpy.ndarray: Preprocessed input data
        """
        if model_type not in self.label_encoders:
            self.load_encoders(model_type)
            df = pd.DataFrame(input_data)
            required_features = MODEL_CONFIGS[model_type]['features']
            assert required_features.sort() == list(input_data.keys()).sort(), \
                f"Required features {required_features} not found in input data"
                #encode the feature status
                if "status" in df.columns:
                    encoder = self.label_encoders[model_type].get("status")
                    if encoder:
                        df["status"] = encoder.transform(df["status"])

                #encode the feature location
                if "location" in df.columns:
                    encoder = self.label_encoders[model_type].get("location")
                    if encoder:
                        df["location"] = encoder.transform(df["location"])
                        
               #scale the features
               if model_type not in self.scalers:
                self.load_encoders(model_type)
                df_scaled = self.scalers[model_type].transform(df)
                return df_scaled.values


    def validate_prediction_range(self, prediction, model_type):
        """Validate that prediction is within expected range

        Args:
            prediction (float): Predicted value
            model_type (str): Type of model

        Returns:
            dict: Validation result with 'valid' boolean and 'message' string
        """
        #get the std of the target column in the dataset
        data = self.load_training_data(model_type)
        df = pd.DataFrame(data)
        std = df[MODEL_CONFIGS[model_type]['target']].std()
        mean = df[MODEL_CONFIGS[model_type]['target']].mean()
        #validate the prediction
        if prediction < mean - 3*std or prediction > mean + 3*std:
            return {
                "valid": False,
                "message": f"Prediction {prediction} is outside the expected range of {mean - 3*std} to {mean + 3*std}"
            }
        else:
        return {
            "valid": True,
            "message": f"Prediction {prediction} is within the expected range of {mean - 3*std} to {mean + 3*std}"
        }
