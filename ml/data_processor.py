import pickle
import pandas as pd
from sklearn.preprocessing import (
    StandardScaler,
    OrdinalEncoder,
)
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import json

from ml.config import DATA_DIR, TRAINING_CONFIG, MODEL_CONFIGS


class DataProcessor:
    """Handles data loading, preprocessing, and preparation for ML prediction models"""

    def __init__(self):
        """Initialize DataProcessor with empty containers for scalers, encoders, and feature columns"""
        self.scalers = {}
        self.label_encoders = {}
        self.onehot_encoders = {}
        self.feature_columns = {}
        self.preprocessing_pipelines = {}
        self.feature_names = {}

    def load_training_data(self, model_type):
        """Load training data from CSV or JSON files

        Args:
            model_type (str): Type of model (e.g., 'soil_moisture_predictor', 'irrigation_recommendation')

        Returns:
            pandas.DataFrame: Training data
        """
        data_path = DATA_DIR / f"{model_type}.csv"

        if data_path.exists():
            data = pd.read_csv(data_path)
            return data
        else:
            raise FileNotFoundError(
                f"Training data for {model_type} not found at {data_path}"
            )

    def _engineer_features(self, data, model_type):
        """Engineer features for soil moisture prediction and irrigation recommendation

        Args:
            data (pandas.DataFrame): Raw data
            model_type (str): Type of model

        Returns:
            pandas.DataFrame: Data with engineered features
        """
        df = data.copy()

        if model_type == "soil_moisture_predictor":
            # Convert timestamp to datetime if it's a string
            if "timestamp" in df.columns:
                if df["timestamp"].dtype == "object":
                    df["timestamp"] = pd.to_datetime(df["timestamp"])

                # Extract essential time-based features only
                df["hour_of_day"] = df["timestamp"].dt.hour
                df["month"] = df["timestamp"].dt.month

                # Seasonal features (growing season)
                df["is_growing_season"] = (
                    df["month"].isin([3, 4, 5, 6, 7, 8, 9]).astype(int)
                )

            # Create interaction features (most important)
            if "temperature_celsius" in df.columns and "humidity_percent" in df.columns:
                df["temp_humidity_interaction"] = (
                    df["temperature_celsius"] * df["humidity_percent"] / 100
                )

            # Battery health indicator (simplified)
            if "battery_voltage" in df.columns:
                df["low_battery"] = (df["battery_voltage"] < 3.6).astype(int)

            # Remove high-cardinality categorical features that cause overfitting
            high_cardinality_features = ["sensor_id", "location", "record_id"]
            for feature in high_cardinality_features:
                if feature in df.columns:
                    df = df.drop(feature, axis=1)

        elif model_type == "irrigation_recommendation":
            # Remove rows with NaN values in irrigation_action column
            if "irrigation_action" in df.columns:
                initial_count = len(df)
                df = df.dropna(subset=["irrigation_action"])
                removed_count = initial_count - len(df)
                print(f"Removed {removed_count} rows with NaN irrigation_action values")
                print(f"Remaining rows: {len(df)}")

            # Convert timestamp to datetime if it's a string
            if "timestamp" in df.columns:
                if df["timestamp"].dtype == "object":
                    df["timestamp"] = pd.to_datetime(df["timestamp"])

                # Extract essential time-based features only
                df["hour_of_day"] = df["timestamp"].dt.hour
                df["month"] = df["timestamp"].dt.month

                # Seasonal features (growing season)
                df["is_growing_season"] = (
                    df["month"].isin([3, 4, 5, 6, 7, 8, 9]).astype(int)
                )

            # Create interaction features (most important)
            if "temperature_celsius" in df.columns and "humidity_percent" in df.columns:
                df["temp_humidity_interaction"] = (
                    df["temperature_celsius"] * df["humidity_percent"] / 100
                )

            # Battery health indicator (simplified)
            if "battery_voltage" in df.columns:
                df["low_battery"] = (df["battery_voltage"] < 3.6).astype(int)

            # Remove high-cardinality categorical features that cause overfitting
            high_cardinality_features = ["sensor_id", "location", "record_id"]
            for feature in high_cardinality_features:
                if feature in df.columns:
                    df = df.drop(feature, axis=1)

        return df

    def _create_preprocessing_pipeline(self, data, model_type):
        """Create a preprocessing pipeline for the given model type

        Args:
            data (pandas.DataFrame): Training data
            model_type (str): Type of model

        Returns:
            sklearn.pipeline.Pipeline: Preprocessing pipeline
        """
        if model_type == "soil_moisture_predictor":
            # Define numeric and categorical features (reduced set)
            numeric_features = [
                "temperature_celsius",
                "humidity_percent",
                "battery_voltage",
                "hour_of_day",
                "month",
                "is_growing_season",
                "temp_humidity_interaction",
                "low_battery",
            ]

            categorical_features = [
                "status",
                "irrigation_action",
            ]

            # Filter to only include features that exist in the data
            numeric_features = [f for f in numeric_features if f in data.columns]
            categorical_features = [
                f for f in categorical_features if f in data.columns
            ]

            # Create preprocessing steps
            numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])

            # Use ordinal encoding for categorical features to reduce dimensionality
            categorical_transformer = Pipeline(
                steps=[
                    (
                        "imputer",
                        SimpleImputer(strategy="constant", fill_value="None"),
                    ),
                    (
                        "ordinal_encoder",
                        OrdinalEncoder(
                            handle_unknown="use_encoded_value", unknown_value=-1
                        ),
                    ),
                ]
            )

            # Combine transformers
            preprocessor = ColumnTransformer(
                transformers=[
                    ("num", numeric_transformer, numeric_features),
                    ("cat", categorical_transformer, categorical_features),
                ],
                remainder="drop",
            )

            return preprocessor, numeric_features, categorical_features

        elif model_type == "irrigation_recommendation":
            # Define numeric and categorical features for irrigation recommendation
            numeric_features = [
                "soil_moisture_percent",
                "temperature_celsius",
                "humidity_percent",
                "battery_voltage",
                "hour_of_day",
                "month",
                "is_growing_season",
                "temp_humidity_interaction",
                "low_battery",
            ]

            categorical_features = [
                "status",
            ]

            # Filter to only include features that exist in the data
            numeric_features = [f for f in numeric_features if f in data.columns]
            categorical_features = [
                f for f in categorical_features if f in data.columns
            ]

            # Create preprocessing steps
            numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])

            # Use ordinal encoding for categorical features to reduce dimensionality
            categorical_transformer = Pipeline(
                steps=[
                    (
                        "imputer",
                        SimpleImputer(strategy="constant", fill_value="None"),
                    ),
                    (
                        "ordinal_encoder",
                        OrdinalEncoder(
                            handle_unknown="use_encoded_value", unknown_value=-1
                        ),
                    ),
                ]
            )

            # Combine transformers
            preprocessor = ColumnTransformer(
                transformers=[
                    ("num", numeric_transformer, numeric_features),
                    ("cat", categorical_transformer, categorical_features),
                ],
                remainder="drop",
            )

            return preprocessor, numeric_features, categorical_features

        # For other model types, use simple scaling
        else:
            numeric_features = [
                col for col in data.columns if data[col].dtype in ["int64", "float64"]
            ]
            numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])

            preprocessor = ColumnTransformer(
                transformers=[("num", numeric_transformer, numeric_features)],
                remainder="drop",
            )

            return preprocessor, numeric_features, []

    def prepare_data(self, data, model_type, features=None, target=None):
        """Prepare data for training regression models with comprehensive feature engineering

        Args:
            data (pandas.DataFrame): Raw training data
            model_type (str): Type of model
            features (list, optional): List of feature column names (if None, will use all available features)
            target (str, optional): Target column name (if None, will use default from config)

        Returns:
            tuple: (X_train_scaled, X_test_scaled, y_train, y_test, feature_names, preprocessor)
        """
        if data is None:
            data = self.load_training_data(model_type)
        else:
            data = pd.DataFrame(data)

        df = pd.DataFrame(data)
        if df.empty:
            raise ValueError(f"No data found for model type {model_type}")

        # Use default target if not specified
        if target is None:
            target = MODEL_CONFIGS[model_type]["target"]

        # Engineer features based on model type
        df_engineered = self._engineer_features(df, model_type)

        # Create preprocessing pipeline
        preprocessor, numeric_features, categorical_features = (
            self._create_preprocessing_pipeline(df_engineered, model_type)
        )

        # Store the preprocessor for later use
        self.preprocessing_pipelines[model_type] = preprocessor

        # Determine features to use
        if features is None:
            # Use all features except the target
            available_features = [col for col in df_engineered.columns if col != target]
            features = available_features

        # Ensure target is not in features
        if target in features:
            features.remove(target)

        # Split the data
        X = df_engineered[features]
        y = df_engineered[target]

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=TRAINING_CONFIG["test_size"],
            random_state=TRAINING_CONFIG["random_state"],
        )

        # Fit and transform the preprocessing pipeline
        print(f"Fitting preprocessing pipeline for model type {model_type}")
        print(f"Features being used: {features}")
        print(f"Target variable: {target}")

        # Fit the preprocessor on training data
        X_train_processed = preprocessor.fit_transform(X_train)
        X_test_processed = preprocessor.transform(X_test)

        # Get feature names after preprocessing
        feature_names = []
        if hasattr(preprocessor, "get_feature_names_out"):
            feature_names = preprocessor.get_feature_names_out().tolist()
        else:
            # Fallback for older sklearn versions
            feature_names = [f"feature_{i}" for i in range(X_train_processed.shape[1])]  # type: ignore

        # Store feature information
        self.feature_names[model_type] = feature_names
        self.feature_columns[model_type] = features

        print(f"Preprocessed training data shape: {X_train_processed.shape}")
        print(f"Preprocessed test data shape: {X_test_processed.shape}")
        print(f"Number of features after preprocessing: {len(feature_names)}")

        return (
            X_train_processed,
            X_test_processed,
            y_train.values if hasattr(y_train, "values") else y_train,  # type: ignore
            y_test.values if hasattr(y_test, "values") else y_test,  # type: ignore
            feature_names,
            preprocessor,
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

    def save_encoders(self, model_type):
        """Save preprocessing pipeline to file

        Args:
            model_type (str): Type of model
        """
        if model_type not in self.preprocessing_pipelines:
            raise ValueError(
                f"No preprocessing pipeline found for model type {model_type}"
            )

        # Save to models directory to match ModelTrainer
        from pathlib import Path

        MODELS_DIR = Path(__file__).parent / "models"

        pipeline_path = MODELS_DIR / f"{model_type}_preprocessor.pkl"
        feature_info_path = MODELS_DIR / f"{model_type}_feature_info.json"

        # Save the preprocessing pipeline
        with open(pipeline_path, "wb") as f:
            pickle.dump(self.preprocessing_pipelines[model_type], f)

        # Save feature information
        feature_info = {
            "feature_columns": self.feature_columns.get(model_type, []),
            "feature_names": self.feature_names.get(model_type, []),
        }
        with open(feature_info_path, "w") as f:
            json.dump(feature_info, f, indent=2)

        print(
            f"Preprocessing pipeline and feature info saved to {pipeline_path} and {feature_info_path}"
        )
        return pipeline_path, feature_info_path

    def load_encoders(self, model_type):
        """Load saved preprocessing pipeline from file

        Args:
            model_type (str): Type of model
        """
        # Try different naming conventions for the preprocessor file
        from pathlib import Path

        MODELS_DIR = Path(__file__).parent / "models"

        possible_paths = [
            DATA_DIR / f"{model_type}_preprocessor.pkl",
            DATA_DIR / f"{model_type}_preprocessor.joblib",
            MODELS_DIR / f"{model_type}_preprocessor.pkl",
            MODELS_DIR / f"{model_type}_preprocessor.joblib",
        ]

        # Also try to find any preprocessor file for this model type
        import glob
        import os

        pattern = f"{model_type}_*_preprocessor.*"
        matching_files = list(DATA_DIR.glob(pattern)) + list(MODELS_DIR.glob(pattern))

        if matching_files:
            # Use the first matching file
            pipeline_path = matching_files[0]
        else:
            # Try the standard paths
            pipeline_path = None
            for path in possible_paths:
                if path.exists():
                    pipeline_path = path
                    break

        if pipeline_path is None or not pipeline_path.exists():
            raise FileNotFoundError(
                f"Preprocessing pipeline not found for model type {model_type}. "
                f"Tried paths: {[str(p) for p in possible_paths]} and pattern: {pattern}"
            )

        # Load the preprocessing pipeline
        if pipeline_path.suffix == ".joblib":
            import joblib

            self.preprocessing_pipelines[model_type] = joblib.load(pipeline_path)
        else:
            with open(pipeline_path, "rb") as f:
                self.preprocessing_pipelines[model_type] = pickle.load(f)

        # Try to load feature information from model trainer results
        results_pattern = f"{model_type}_*_results.json"
        results_files = list(MODELS_DIR.glob(results_pattern))
        if results_files:
            with open(results_files[0], "r") as f:
                results = json.load(f)
                self.feature_columns[model_type] = results.get("feature_columns", [])
                self.feature_names[model_type] = results.get("feature_names", [])
        else:
            # Fallback to feature info file if it exists
            feature_info_path = MODELS_DIR / f"{model_type}_feature_info.json"
            if feature_info_path.exists():
                with open(feature_info_path, "r") as f:
                    feature_info = json.load(f)
                    self.feature_columns[model_type] = feature_info.get(
                        "feature_columns", []
                    )
                    self.feature_names[model_type] = feature_info.get(
                        "feature_names", []
                    )

        # Ensure feature columns and names are always lists
        if self.feature_columns.get(model_type) is None:
            self.feature_columns[model_type] = []
        if self.feature_names.get(model_type) is None:
            self.feature_names[model_type] = []

        print(f"Preprocessing pipeline loaded from {pipeline_path}")
        print(f"Feature columns: {self.feature_columns.get(model_type, [])}")
        print(f"Feature names: {self.feature_names.get(model_type, [])}")
        return self.preprocessing_pipelines[model_type]

    def preprocess_input(self, input_data, model_type):
        """Preprocess input data for prediction using the fitted preprocessing pipeline

        Args:
            input_data (dict): Input data dictionary
            model_type (str): Type of model

        Returns:
            numpy.ndarray: Preprocessed input data
        """
        # Try to load the preprocessor from file if not in memory
        if model_type not in self.preprocessing_pipelines:
            try:
                self.load_encoders(model_type)
            except FileNotFoundError:
                raise ValueError(
                    f"No preprocessing pipeline found for model type {model_type}. "
                    f"Please train the model first."
                )

        # Convert input to DataFrame
        df = pd.DataFrame([input_data])

        # Engineer features for the input data
        df_engineered = self._engineer_features(df, model_type)

        # Get the features that were used during training
        if model_type not in self.feature_columns:
            raise ValueError(
                f"No feature columns found for model type {model_type}. "
                f"Please train the model first."
            )

        required_features = self.feature_columns[model_type]

        # Check if all required features are present
        missing_features = [
            f for f in required_features if f not in df_engineered.columns
        ]
        if missing_features:
            raise ValueError(f"Missing required features: {missing_features}")

        # Select only the required features
        X = df_engineered[required_features]

        # Transform using the fitted preprocessor
        X_processed = self.preprocessing_pipelines[model_type].transform(X)

        return X_processed

    def validate_prediction_range(self, prediction, model_type):
        """Validate that prediction is within expected range

        Args:
            prediction (float): Predicted value
            model_type (str): Type of model

        Returns:
            dict: Validation result with 'valid' boolean and 'message' string
        """
        # get the std of the target column in the dataset
        data = self.load_training_data(model_type)
        df = pd.DataFrame(data)
        std = df[MODEL_CONFIGS[model_type]["target"]].std()
        mean = df[MODEL_CONFIGS[model_type]["target"]].mean()
        # validate the prediction
        if prediction < mean - 3 * std or prediction > mean + 3 * std:
            return {
                "valid": False,
                "message": f"Prediction {prediction} is outside the expected range of {mean - 3*std} to {mean + 3*std}",
            }
        else:
            return {
                "valid": True,
                "message": f"Prediction {prediction} is within the expected range of {mean - 3*std} to {mean + 3*std}",
            }

    def get_feature_importance(self, model_type):
        """Get feature importance for a trained model

        Args:
            model_type (str): Type of model

        Returns:
            dict: Feature importance scores
        """
        if model_type not in self.feature_names:
            raise ValueError(f"No feature names found for model type {model_type}")

        # This method should be called after a model is trained
        # The actual implementation would depend on the model type
        # For now, return a placeholder
        feature_names = self.feature_names[model_type]
        return {name: 0.0 for name in feature_names}

    def get_feature_statistics(self, model_type):
        """Get statistics about the engineered features

        Args:
            model_type (str): Type of model

        Returns:
            dict: Feature statistics
        """
        if model_type not in self.feature_columns:
            raise ValueError(f"No feature columns found for model type {model_type}")

        # Load the data to get statistics
        data = self.load_training_data(model_type)
        df_engineered = self._engineer_features(data, model_type)

        stats = {}
        for feature in self.feature_columns[model_type]:
            if feature in df_engineered.columns:
                col_data = df_engineered[feature]
                if col_data.dtype in ["int64", "float64"]:
                    stats[feature] = {
                        "mean": col_data.mean(),
                        "std": col_data.std(),
                        "min": col_data.min(),
                        "max": col_data.max(),
                        "type": "numeric",
                    }
                else:
                    stats[feature] = {
                        "unique_values": col_data.nunique(),
                        "most_common": (
                            col_data.mode().iloc[0]
                            if not col_data.mode().empty
                            else None
                        ),
                        "type": "categorical",
                    }

        return stats
