from ml import MLEngine


def test_ml_engine():
    ml_engine = MLEngine()

    training = ml_engine.train_soil_moisture_predictor()



    print(training, "\n" * 5)

    test_data = {
        "sensor_id": "1234567890",
        "location": "Garden",
        "temperature_celsius": 22,
        "humidity_percent": 50,
        "battery_voltage": 3.8,
        "status": "Normal",
        "irrigation_action": "None",
        "timestamp": "2025-01-01 12:00:00",
    }
    # manual
    # forms in our frontend

    # csv
    # we can upload csv files

    # sensor data
    # api that can accept the parameters

    # test_res = ml_engine.predict_soil_moisture(**test_data)

    train = ml_engine.train_irrigation_recommender()

    # Make data for irrigation recommendation input
    ir_input = {
        "soil_moisture_percent": 35,
        "temperature_celsius": 22,
        "humidity_percent": 50,
        "battery_voltage": 3.8,
        "status": "Normal",
        "timestamp": "2025-01-01 12:00:00",
    }
    
    test_ir = ml_engine.recommend_irrigation(**ir_input)


    print(test_ir, "\n" * 5)
