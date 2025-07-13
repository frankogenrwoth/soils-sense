# SoilSense ML Engine

The ML Engine is the main interface for interacting with machine learning models trained on the SoilSense platform. It provides easy-to-use methods for training, prediction, and model management for soil moisture prediction and irrigation recommendations.

## Features

- **Soil Moisture Prediction**: Predict soil moisture levels based on environmental and sensor data
- **Irrigation Recommendations**: Get intelligent irrigation recommendations based on current conditions
- **Multiple Algorithms**: Support for various ML algorithms (Random Forest, Gradient Boosting, SVR, MLP, Linear/Logistic Regression)
- **Model Management**: Train, retrain, and manage multiple model versions
- **Data Processing**: Built-in data preprocessing and feature engineering
- **Easy Integration**: Simple API for integration with web applications

## Quick Start

### Basic Usage

```python
from ml import MLEngine

# Initialize the ML engine
ml_engine = MLEngine()

# Train all models
results = ml_engine.train_all_models()
print(results)
```

### Soil Moisture Prediction

```python
# Predict soil moisture level
prediction = ml_engine.predict_soil_moisture(
    sensor_id="SENSOR_001",
    location="Field_A",
    temperature_celsius=25.5,
    humidity_percent=65.0,
    battery_voltage=3.8,
    status="Normal",
    irrigation_action="None",
    timestamp="2024-01-15 14:30:00"
)

print(f"Predicted soil moisture: {prediction['prediction']}%")
```

### Irrigation Recommendations

```python
# Get irrigation recommendation
recommendation = ml_engine.recommend_irrigation(
    soil_moisture_percent=35.0,
    temperature_celsius=28.0,
    humidity_percent=45.0,
    battery_voltage=3.8,
    status="Normal"
)

print(f"Irrigation recommendation: {recommendation['prediction']}")
```

## Model Training

### Train Individual Models

```python
# Train soil moisture predictor
soil_results = ml_engine.train_soil_moisture_predictor()

# Train irrigation recommender
irrigation_results = ml_engine.train_irrigation_recommender()

# Train with custom data
import pandas as pd
custom_data = pd.read_csv("your_data.csv")
results = ml_engine.train_soil_moisture_predictor(custom_data=custom_data)
```

### Train All Models

```python
# Train all available models
all_results = ml_engine.train_all_models()

# Train all models with custom data
all_results = ml_engine.train_all_models(custom_data=custom_data)
```

### Train on Multiple Algorithms

```python
# Train soil moisture predictor on all algorithms
soil_algorithms = ml_engine.train_model_on_all_algorithms("soil_moisture_predictor")

# Train irrigation recommender on all algorithms
irrigation_algorithms = ml_engine.train_model_on_all_algorithms("irrigation_recommendation")
```

## Model Management

### List Available Models

```python
# Get list of available model types
available_models = ml_engine.get_available_models()
print(available_models)  # ['soil_moisture_predictor', 'irrigation_recommendation']

# List all trained models with details
all_models = ml_engine.list_all_models()
for model in all_models:
    print(f"Model: {model['model_type']}")
    print(f"Algorithm: {model['algorithm']}")
    print(f"Performance: {model['performance']}")
```

### Get Model Information

```python
# Get specific model info
model_info = ml_engine.get_model_info("soil_moisture_predictor")
print(model_info)
```

### Retrain Models

```python
# Retrain existing model
retrain_results = ml_engine.retrain_model("soil_moisture_predictor")

# Retrain with new data
retrain_results = ml_engine.retrain_model(
    "soil_moisture_predictor", 
    new_data=custom_data
)
```

## Data Management

### Save and Load Training Data

```python
# Save training data
ml_engine.save_training_data(data, "soil_moisture_predictor")

# Load training data
loaded_data = ml_engine.load_training_data("soil_moisture_predictor")
```

## Advanced Usage

### Compare Multiple Algorithms

```python
# Test prediction across all algorithms
test_data = {
    "sensor_id": "SENSOR_001",
    "location": "Field_A",
    "temperature_celsius": 25.5,
    "humidity_percent": 65.0,
    "battery_voltage": 3.8,
    "status": "Normal",
    "irrigation_action": "None",
    "timestamp": "2024-01-15 14:30:00"
}

# Compare soil moisture predictions
soil_predictions = ml_engine.predict_model_on_all_algorithms(
    "soil_moisture_predictor", 
    test_data
)

# Compare irrigation recommendations
irrigation_input = {
    "soil_moisture_percent": 35.0,
    "temperature_celsius": 28.0,
    "humidity_percent": 45.0,
    "battery_voltage": 3.8,
    "status": "Normal"
}

irrigation_predictions = ml_engine.predict_model_on_all_algorithms(
    "irrigation_recommendation", 
    irrigation_input
)
```

## Available Models

### Soil Moisture Predictor
- **Type**: Regression
- **Target**: `soil_moisture_percent`
- **Features**: Temperature, humidity, battery voltage, time features, sensor status, irrigation action
- **Algorithms**: Random Forest, Gradient Boosting, SVR, MLP, Linear Regression

### Irrigation Recommender
- **Type**: Classification
- **Target**: `irrigation_action`
- **Features**: Soil moisture, temperature, humidity, battery voltage, time features, sensor status
- **Algorithms**: Random Forest, Gradient Boosting, Logistic Regression

## Configuration

The ML engine uses configuration files to define:
- Model features and targets
- Algorithm hyperparameters
- Training parameters
- Prediction settings

Key configuration files:
- `config.py`: Main configuration with model definitions and algorithm parameters
- `data_processor.py`: Data preprocessing and feature engineering
- `model_trainer.py`: Training pipeline and model management
- `predictor.py`: Prediction interface and model loading

## Requirements

- Python 3.8+
- pandas
- scikit-learn
- numpy
- joblib

## File Structure

```
ml/
├── __init__.py          # Main MLEngine interface
├── config.py            # Configuration and model definitions
├── data_processor.py    # Data preprocessing and feature engineering
├── model_trainer.py     # Model training pipeline
├── predictor.py         # Prediction interface
├── playground.py        # Example usage and testing
├── models/              # Trained model files
├── data/                # Training and test data
└── README.md           # This file
```

## Examples

See `playground.py` for complete working examples of:
- Model training
- Prediction testing
- Algorithm comparison
- Data management

## Integration

The ML Engine is designed to integrate seamlessly with the SoilSense web application. Models can be:
- Trained via admin interface
- Used for real-time predictions
- Updated with new sensor data
- Managed through the technician dashboard

