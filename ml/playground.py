from ml import MLEngine


def set_up_ml_engine():
    ml_engine = MLEngine()

    training_soil = ml_engine.train_model_on_all_algorithms("soil_moisture_predictor")
    training_ir = ml_engine.train_model_on_all_algorithms("irrigation_recommendation")

    print(training_soil, "\n", "*" * 50, "\n")
    print(training_ir, "\n", "*" * 50, "\n")


def test_ml_engine():
    ml_engine = MLEngine()

    # training = ml_engine.train_soil_moisture_predictor()
    # training_soil = ml_engine.train_model_on_all_algorithms("soil_moisture_predictor")

    # training = ml_engine.train_irrigation_recommender()
    # training_ir = ml_engine.train_model_on_all_algorithms("irrigation_recommendation")

    # print(training_soil, "\n", "*" * 50, "\n")
    # print(training_ir, "\n", "*" * 50, "\n")
    # test_data = {
    #     "sensor_id": "1234567890",
    #     "location": "Garden",
    #     "temperature_celsius": 22,
    #     "humidity_percent": 50,
    #     "battery_voltage": 3.8,
    #     "status": "Normal",
    #     "irrigation_action": "None",
    #     "timestamp": "2025-01-01 12:00:00",
    # }

    # prediction = ml_engine.predict_model_on_all_algorithms(
    #     "soil_moisture_predictor", test_data
    # )
    # print(prediction, "\n", "*" * 50, "\n")

    # # manual
    # # forms in our frontend

    # # csv
    # # we can upload csv files

    # # sensor data
    # # api that can accept the parameters

    # # test_res = ml_engine.predict_soil_moisture(**test_data)

    # train = ml_engine.train_irrigation_recommender()

    # # Make data for irrigation recommendation input
    ir_input = {
        "soil_moisture_percent": 35,
        "temperature_celsius": 22,
        "humidity_percent": 50,
        "battery_voltage": 3.8,
        "status": "Normal",
        "timestamp": "2025-01-01 12:00:00",
    }

    prediction_ir = ml_engine.predict_model_on_all_algorithms(
        "irrigation_recommendation", ir_input
    )

    print(prediction_ir, "\n", "*" * 50, "\n")
