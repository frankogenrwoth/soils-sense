"""
SoilSense ML Engine Playground
==============================

This file contains comprehensive examples demonstrating how to use the ML Engine
interface for training, prediction, and model management.

Run this file to see the ML Engine in action!
"""

import pandas as pd
from ml import MLEngine


def setup_example():
    """Demonstrate setup of the ML Engine"""
    print("=" * 60)
    print("SETUP - TRAINING ALL MODELS")
    print("=" * 60)
    ml_engine = MLEngine()
    ml_engine.train_model_on_all_algorithms("irrigation_recommendation")
    ml_engine.train_model_on_all_algorithms("soil_moisture_predictor")

    print("=" * 60)


def basic_usage_example():
    """Demonstrate basic ML Engine usage"""
    print("=" * 60)
    print("BASIC USAGE EXAMPLE")
    print("=" * 60)

    # Initialize the ML engine
    ml_engine = MLEngine()

    # Get available models
    available_models = ml_engine.get_available_models()
    print(f"Available models: {available_models}")

    # List all trained models
    all_models = ml_engine.list_all_models()
    print(f"\nFound {len(all_models)} trained models:")
    for model in all_models:
        print(f"  - {model['model_type']} ({model['algorithm']})")

    print("\n" + "=" * 60)


def soil_moisture_prediction_example():
    """Demonstrate soil moisture prediction"""
    print("SOIL MOISTURE PREDICTION EXAMPLE")
    print("=" * 60)

    ml_engine = MLEngine()

    # Example sensor data for prediction
    test_data = {
        "sensor_id": "SENSOR_001",
        "location": "Field_A",
        "temperature_celsius": 25.5,
        "humidity_percent": 65.0,
        "battery_voltage": 3.8,
        "status": "Normal",
        "irrigation_action": "None",
        "timestamp": "2024-01-15 14:30:00",
    }

    print("Input data:")
    for key, value in test_data.items():
        print(f"  {key}: {value}")

    # Predict soil moisture using the main interface
    prediction = ml_engine.predict_soil_moisture(**test_data)
    print(f"\nPredicted soil moisture: {prediction['predicted_value']}%")
    print(f"Confidence: {prediction.get('confidence', 'N/A')}")

    # Compare predictions across all algorithms
    print("\nComparing predictions across all algorithms:")
    all_predictions = ml_engine.predict_model_on_all_algorithms(
        "soil_moisture_predictor", test_data
    )

    for algorithm, result in all_predictions.items():
        print(f"  {algorithm}: {result['predicted_value']}%")

    print("\n" + "=" * 60)


def irrigation_recommendation_example():
    """Demonstrate irrigation recommendations"""
    print("IRRIGATION RECOMMENDATION EXAMPLE")
    print("=" * 60)

    ml_engine = MLEngine()

    # Example data for irrigation recommendation
    ir_input = {
        "soil_moisture_percent": 35.0,
        "temperature_celsius": 28.0,
        "humidity_percent": 45.0,
        "battery_voltage": 3.8,
        "status": "Normal",
        "timestamp": "2024-01-15 14:30:00",
    }

    print("Input data:")
    for key, value in ir_input.items():
        print(f"  {key}: {value}")

    # Get irrigation recommendation using the main interface
    recommendation = ml_engine.recommend_irrigation(**ir_input)
    print(f"\nIrrigation recommendation: {recommendation['predicted_value']}")
    print(f"Confidence: {recommendation.get('confidence', 'N/A')}")

    # Compare recommendations across all algorithms
    print("\nComparing recommendations across all algorithms:")
    all_recommendations = ml_engine.predict_model_on_all_algorithms(
        "irrigation_recommendation", ir_input
    )

    for algorithm, result in all_recommendations.items():
        print(f"  {algorithm}: {result['predicted_value']}")

    print("\n" + "=" * 60)


def model_training_example():
    """Demonstrate model training capabilities"""
    print("MODEL TRAINING EXAMPLE")
    print("=" * 60)

    ml_engine = MLEngine()

    # Train individual models
    print("Training soil moisture predictor...")
    soil_results = ml_engine.train_soil_moisture_predictor()
    print(
        f"Training completed! Best algorithm: {soil_results.get('best_algorithm', 'N/A')}"
    )

    print("\nTraining irrigation recommender...")
    irrigation_results = ml_engine.train_irrigation_recommender()
    print(
        f"Training completed! Best algorithm: {irrigation_results.get('best_algorithm', 'N/A')}"
    )

    # Train on multiple algorithms
    print("\nTraining soil moisture predictor on all algorithms...")
    soil_algorithms = ml_engine.train_model_on_all_algorithms("soil_moisture_predictor")

    print("Results for each algorithm:")
    for algorithm, result in soil_algorithms.items():
        performance = result.get("performance", {})
        score = performance.get("test_score", "N/A")
        print(f"  {algorithm}: {score}")

    print("\n" + "=" * 60)


def model_management_example():
    """Demonstrate model management features"""
    print("MODEL MANAGEMENT EXAMPLE")
    print("=" * 60)

    ml_engine = MLEngine()

    # Get model information
    print("Getting model information...")
    model_info = ml_engine.get_model_info("soil_moisture_predictor")
    print(f"Model type: {model_info.get('model_type', 'N/A')}")
    print(f"Features: {model_info.get('features', [])}")
    print(f"Target: {model_info.get('target', 'N/A')}")

    # List all models with details
    print("\nAll trained models:")
    all_models = ml_engine.list_all_models()
    for model in all_models:
        print(f"  Model: {model['model_type']}")
        print(f"    Algorithm: {model['algorithm']}")
        print(f"    Performance: {model.get('performance', 'N/A')}")
        print(f"    Trained: {model.get('trained_at', 'N/A')}")
        print()

    print("=" * 60)


def comprehensive_test():
    """Run a comprehensive test of all features"""
    print("COMPREHENSIVE ML ENGINE TEST")
    print("=" * 60)

    ml_engine = MLEngine()

    # Test soil moisture prediction
    print("1. Testing soil moisture prediction...")
    soil_prediction = ml_engine.predict_soil_moisture(
        sensor_id="TEST_001",
        location="Test_Field",
        temperature_celsius=24.0,
        humidity_percent=60.0,
        battery_voltage=3.8,
        status="Normal",
        irrigation_action="None",
        timestamp="2024-01-15 12:00:00",
    )
    print(f"   Result: {soil_prediction['predicted_value']}%")

    # Test irrigation recommendation
    print("2. Testing irrigation recommendation...")
    ir_recommendation = ml_engine.recommend_irrigation(
        soil_moisture_percent=32.0,
        temperature_celsius=26.0,
        humidity_percent=50.0,
        battery_voltage=3.8,
        status="Normal",
    )
    print(f"   Result: {ir_recommendation['predicted_value']}")

    # Test model listing
    print("3. Testing model listing...")
    models = ml_engine.list_all_models()
    print(f"   Found {len(models)} trained models")

    # Test available models
    print("4. Testing available models...")
    available = ml_engine.get_available_models()
    print(f"   Available model types: {available}")

    print("\nAll tests completed successfully!")
    print("=" * 60)


def main():
    """Run all examples"""
    print("SoilSense ML Engine Playground")
    print("Running comprehensive examples...\n")

    try:
        # Run all examples
        basic_usage_example()
        soil_moisture_prediction_example()
        irrigation_recommendation_example()
        model_training_example()
        model_management_example()

        comprehensive_test()

        print("\n🎉 All examples completed successfully!")
        print("\nYou can now use the ML Engine in your applications!")

    except Exception as e:
        import traceback

        print(traceback.format_exc())
        print(f"\n❌ Error running examples: {e}")
        print("Make sure all models are trained and dependencies are installed.")


if __name__ == "__main__":
    main()
