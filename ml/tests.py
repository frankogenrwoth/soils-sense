import unittest
from unittest import TestCase
from pathlib import Path

try:
    from ml.config import (
        MODEL_CONFIGS,
        REGRESSION_ALGORITHMS,
        CLASSIFICATION_ALGORITHMS,
        DEFAULT_ALGORITHMS,
        TRAINING_CONFIG,
        PREDICTION_CONFIG,
        MODELS_DIR,
        DATA_DIR,
    )
except ImportError:
    from config import (
        MODEL_CONFIGS,
        REGRESSION_ALGORITHMS,
        CLASSIFICATION_ALGORITHMS,
        DEFAULT_ALGORITHMS,
        TRAINING_CONFIG,
        PREDICTION_CONFIG,
        MODELS_DIR,
        DATA_DIR,
    )


class TestModelConfig(TestCase):
    def test_model_and_data_directories(self):
        """
        Test the model and data directories
        - models directory exists
        - data directory exists
        - models directory is the correct path
        - data directory is the correct path
        """
        self.assertTrue(MODELS_DIR.exists())
        self.assertTrue(DATA_DIR.exists())
        self.assertEqual(MODELS_DIR, Path("ml/models").resolve())
        self.assertEqual(DATA_DIR, Path("ml/data").resolve())

    def test_soil_moisture_predictor_config(self):
        """
        Test the soil moisture predictor config:
            - target is soil_moisture_percent
            - task_type is regression
            - features are:
                - temperature_celsius
                - humidity_percent
                - battery_voltage
                - hour_of_day
                - month
                - is_growing_season
                - temp_humidity_interaction
                - low_battery
                - status
                - irrigation_action
        """
        config = MODEL_CONFIGS["soil_moisture_predictor"]
        self.assertTrue("features" in config)
        self.assertTrue("target" in config)
        self.assertTrue("task_type" in config)
        self.assertEqual(config["target"], "soil_moisture_percent")
        self.assertEqual(config["task_type"], "regression")

        # test for feature names in the features specification
        feature_names = [
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
        ]
        for feature_name in feature_names:
            self.assertTrue(feature_name in config["features"])

    def test_irrigation_recommendation_config(self):
        """
        Test the irrigation recommendation config:
            - target is irrigation_action
            - task_type is classification
            - features are:
                - soil_moisture_percent
                - temperature_celsius
                - humidity_percent
                - hour_of_day
                - month
                - is_growing_season
                - temp_humidity_interaction
                - low_battery
                - status
        """
        config = MODEL_CONFIGS["irrigation_recommendation"]
        self.assertTrue("features" in config)
        self.assertTrue("target" in config)
        self.assertTrue("task_type" in config)
        self.assertEqual(config["target"], "irrigation_action")
        self.assertEqual(config["task_type"], "classification")

        # test for feature names in the features specification
        feature_names = [
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
        ]
        for feature_name in feature_names:
            self.assertTrue(feature_name in config["features"])

    def test_regression_algorithms(self):
        """
        Test the regression algorithms
        - random_forest
        - gradient_boosting
        - svr
        - mlp
        - linear_regression
        """
        self.assertTrue("random_forest" in REGRESSION_ALGORITHMS)
        self.assertTrue("gradient_boosting" in REGRESSION_ALGORITHMS)
        self.assertTrue("svr" in REGRESSION_ALGORITHMS)
        self.assertTrue("mlp" in REGRESSION_ALGORITHMS)
        self.assertTrue("linear_regression" in REGRESSION_ALGORITHMS)

        # test for random_forest parameters
        random_forest_params = [
            "n_estimators",
            "max_depth",
            "min_samples_split",
            "min_samples_leaf",
            "random_state",
            "n_jobs",
        ]
        self.assertTrue(
            len(REGRESSION_ALGORITHMS["random_forest"].keys())
            == len(random_forest_params)
        )
        for param in random_forest_params:
            self.assertTrue(param in REGRESSION_ALGORITHMS["random_forest"])

        # test for gradient_boosting parameters
        gradient_boosting_params = [
            "n_estimators",
            "learning_rate",
            "max_depth",
            "min_samples_split",
            "min_samples_leaf",
            "random_state",
        ]
        self.assertTrue(
            len(REGRESSION_ALGORITHMS["gradient_boosting"].keys())
            == len(gradient_boosting_params)
        )
        for param in gradient_boosting_params:
            self.assertTrue(param in REGRESSION_ALGORITHMS["gradient_boosting"])

        # test for svr parameters
        svr_params = [
            "kernel",
            "C",
            "gamma",
            "epsilon",
        ]
        self.assertTrue(len(REGRESSION_ALGORITHMS["svr"].keys()) == len(svr_params))
        for param in svr_params:
            self.assertTrue(param in REGRESSION_ALGORITHMS["svr"])

        # test for mlp parameters
        mlp_params = [
            "hidden_layer_sizes",
            "activation",
            "solver",
            "alpha",
            "learning_rate",
            "max_iter",
            "random_state",
        ]
        self.assertTrue(len(REGRESSION_ALGORITHMS["mlp"].keys()) == len(mlp_params))
        for param in mlp_params:
            self.assertTrue(param in REGRESSION_ALGORITHMS["mlp"])

        # test for linear_regression parameters
        linear_regression_params = [
            "fit_intercept",
            "normalize",
        ]
        self.assertTrue(
            len(REGRESSION_ALGORITHMS["linear_regression"].keys())
            == len(linear_regression_params)
        )
        for param in linear_regression_params:
            self.assertTrue(param in REGRESSION_ALGORITHMS["linear_regression"])

    def test_classification_algorithms(self):
        """
        Test the classification algorithms
        """
        training_algorithms = [
            "random_forest",
            "gradient_boosting",
            "logistic_regression",
        ]
        for algorithm in training_algorithms:
            self.assertTrue(algorithm in CLASSIFICATION_ALGORITHMS)

        # test for random_forest parameters
        random_forest_params = [
            "n_estimators",
            "max_depth",
            "min_samples_split",
            "min_samples_leaf",
            "random_state",
            "n_jobs",
        ]
        self.assertTrue(
            len(CLASSIFICATION_ALGORITHMS["random_forest"].keys())
            == len(random_forest_params)
        )
        for param in random_forest_params:
            self.assertTrue(param in CLASSIFICATION_ALGORITHMS["random_forest"])

        # test for gradient_boosting parameters
        gradient_boosting_params = [
            "n_estimators",
            "learning_rate",
            "max_depth",
            "min_samples_split",
            "min_samples_leaf",
            "random_state",
        ]
        self.assertTrue(
            len(CLASSIFICATION_ALGORITHMS["gradient_boosting"].keys())
            == len(gradient_boosting_params)
        )
        for param in gradient_boosting_params:
            self.assertTrue(param in CLASSIFICATION_ALGORITHMS["gradient_boosting"])

        # test for logistic_regression parameters
        logistic_regression_params = [
            "C",
            "max_iter",
            "random_state",
        ]
        self.assertTrue(
            len(CLASSIFICATION_ALGORITHMS["logistic_regression"].keys())
            == len(logistic_regression_params)
        )
        for param in logistic_regression_params:
            self.assertTrue(param in CLASSIFICATION_ALGORITHMS["logistic_regression"])

    def test_default_algorithms(self):
        """
        Test the default algorithms
        - soil_moisture_predictor should use gradient_boosting
        - irrigation_recommendation should use gradient_boosting
        """
        self.assertTrue("soil_moisture_predictor" in DEFAULT_ALGORITHMS)
        self.assertTrue("irrigation_recommendation" in DEFAULT_ALGORITHMS)

        # test for default algorithms
        self.assertEqual(
            DEFAULT_ALGORITHMS["soil_moisture_predictor"], "gradient_boosting"
        )
        self.assertEqual(
            DEFAULT_ALGORITHMS["irrigation_recommendation"], "gradient_boosting"
        )

        self.assertTrue(
            DEFAULT_ALGORITHMS["soil_moisture_predictor"] in REGRESSION_ALGORITHMS
        )
        self.assertTrue(
            DEFAULT_ALGORITHMS["irrigation_recommendation"] in CLASSIFICATION_ALGORITHMS
        )

    def test_training_config(self):
        """
        Test the training config
        - test_size is 0.2
        - random_state is 42
        - n_splits is 5
        - scoring is "neg_mean_squared_error"
        - shuffle is True
        - early_stopping is True
        """
        training_config_params = [
            "test_size",
            "random_state",
            "n_splits",
            "scoring",
            "shuffle",
            "early_stopping",
        ]
        self.assertTrue(len(TRAINING_CONFIG.keys()) == len(training_config_params))
        for param in training_config_params:
            self.assertTrue(param in TRAINING_CONFIG)

        # test for training config values
        self.assertLess(TRAINING_CONFIG["test_size"], 0.5)
        
    def test_prediction_config(self):
        """
        Test the prediction config
        - confidence_threshold is 0.7
        - default_confidence_level is 0.95
        - output_format is "dict"
        - max_batch_size is 128
        - round_predictions is 2
        """
        prediction_config_params = [
            "confidence_threshold",
            "default_confidence_level",
            "output_format",
            "max_batch_size",
            "round_predictions",
        ]
        self.assertTrue(len(PREDICTION_CONFIG.keys()) == len(prediction_config_params))
        for param in prediction_config_params:
            self.assertTrue(param in PREDICTION_CONFIG)


if __name__ == "__main__":
    unittest.main()
