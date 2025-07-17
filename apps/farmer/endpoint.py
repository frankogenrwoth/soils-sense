from rest_framework.decorators import api_view
from django.shortcuts import get_object_or_404
from apps.farmer.models import SoilMoistureReading, Farm
from apps.technician.models import Sensor
from rest_framework.response import Response

from rest_framework import status


@api_view(["POST"])
def get_soil_data(request):
    """
    Accepts:
    {
        temperature: float,
        humidity: float,
        moisture: float,
        status: str,
        farm_id: int,
        sensor_id: int,
        timestamp: datetime,
    }
    """
    required_fields = [
        "temperature",
        "humidity",
        "status",
        "farm_id",
        "sensor_id",
        "timestamp",
    ]
    data = request.data

    missing_fields = [field for field in required_fields if field not in data]

    farm = get_object_or_404(Farm, id=data["farm_id"])
    sensor = get_object_or_404(Sensor, sensor_id=data["sensor_id"])

    if sensor.farm_id != farm.id:
        return Response(
            {"error": "Sensor does not belong to farm"},
            status=status.HTTP_400_BAD_REQUEST,
        )

    if missing_fields:
        return Response(
            {"error": f"Missing required fields: {', '.join(missing_fields)}"},
            status=status.HTTP_400_BAD_REQUEST,
        )

    # Optionally, you could validate types here, but the prompt only asks for existence
    sr_o = SoilMoistureReading.objects.create(
        timestamp=data["timestamp"],
        temperature_celsius=data["temperature"],
        humidity_percent=data["humidity"],
        status=data["status"],
        farm_id=data["farm_id"],
        battery_voltage=data["battery_voltage"],
        reading_source="sensor",
        sensor_id=data["sensor_id"],
    )

    from ml import MLEngine

    ml_engine = MLEngine()

    sr_o.soil_moisture_percent = ml_engine.predict_soil_moisture(
        sensor_id=data["sensor_id"],
        location=farm.location,
        temperature_celsius=data["temperature"],
        humidity_percent=data["humidity"],
        battery_voltage=data["battery_voltage"],
        status=data["status"],
        irrigation_action=None,
        timestamp=data["timestamp"],
    ).get("predicted_value")

    sr_o.save()

    # Print the data on the page (return in response)
    return Response(
        {
            "message": "All required fields received.",
            "data": {field: data[field] for field in required_fields},
        },
        status=status.HTTP_200_OK,
    )
