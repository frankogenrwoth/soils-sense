import os
import django
import random
import time
from django.utils.timezone import now

# Setup Django environment
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'soilsense.settings')
django.setup()

from apps.farmer.models import Farm, SoilMoistureReading
from apps.technician.models import Sensor


def generate_reading(farm, sensor_id):
    soil_moisture = round(random.uniform(10.0, 65.0), 2)
    temperature = round(random.uniform(18.0, 40.0), 2)
    humidity = round(random.uniform(30.0, 90.0), 2)
    battery_voltage = round(random.uniform(3.1, 3.6), 2)

    # Determine status
    if soil_moisture < 15:
        status = 'Critical Low'
        irrigation_action = 'Irrigate'
    elif soil_moisture < 25:
        status = 'Dry'
        irrigation_action = 'Irrigate'
    elif soil_moisture > 60:
        status = 'Critical High'
        irrigation_action = 'Reduce Irrigation'
    elif soil_moisture > 40:
        status = 'Wet'
        irrigation_action = 'Reduce Irrigation'
    else:
        status = 'Normal'
        irrigation_action = 'None'

    try:
        reading = SoilMoistureReading.objects.create(
            farm=farm,
            sensor_id=sensor_id,
            timestamp=now(),
            soil_moisture_percent=soil_moisture,
            temperature_celsius=temperature,
            humidity_percent=humidity,
            status=status,
            battery_voltage=battery_voltage,
            irrigation_action=irrigation_action,
            reading_source='sensor'
        )

        print(f"[{sensor_id}] 🌱 Moisture: {soil_moisture}% | 🌡 Temp: {temperature}°C | 💧 Humidity: {humidity}% | 🔋 Battery: {battery_voltage}V → {status} / {irrigation_action}")
    except Exception as e:
        print(f"❌ Error saving reading for {sensor_id}: {e}")

def simulate():
    while True:
        farms = list(Farm.objects.all())

        if not farms:
            print("⚠️  No farms found in database. Please add at least one farm.")
            time.sleep(30)
            continue

        active_sensors = Sensor.objects.filter(is_active=True).select_related('farm')

        if not active_sensors.exists():
            print("⚠️  No active sensors found. Please add and activate sensors.")
            time.sleep(30)
            continue

        for sensor in active_sensors:
            generate_reading(sensor.farm, sensor.sensor_id)

        print("✅ Readings generated successfully. Waiting for next cycle...")
        time.sleep(30)  # Delay between cycles


if __name__ == '__main__':
    simulate()
