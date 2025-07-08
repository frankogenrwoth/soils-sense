import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import json
import os
from .config import DATA_DIR, TRAINING_CONFIG


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
        pass

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
        x_train_scaled = self.scalers[model_type].transform(x_test)
        return x_train_scaled, x_test_scaled, y_train, y_test, features

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
            required_features = MODEL_CONFIG[model_type]['features']
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
        std = df[MODEL_CONFIG[model_type]['target']].std()
        mean = df[MODEL_CONFIG[model_type]['target']].mean()
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
