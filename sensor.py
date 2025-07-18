import requests
import random
import time
from datetime import datetime
from typing import Dict, Any
from dataclasses import dataclass

# Configuration
URL = "https://soils-sense-production.up.railway.app/farmer/get-soil-data/"
SOIL_STATUS = ["Normal", "Dry", "Wet", "Critical Low", "Critical High"]
FARM_ID = 2
SENSOR_ID = "SENSOR_40"


@dataclass
class SensorRanges:
    """Define valid ranges for sensor readings"""

    temperature: tuple = (-10.0, 50.0)  # Celsius
    humidity: tuple = (0.0, 100.0)  # Percentage
    moisture: tuple = (0.0, 100.0)  # Percentage
    battery: tuple = (2.7, 4.2)  # Voltage (typical Li-ion battery range)


def validate_reading(value: float, valid_range: tuple) -> bool:
    """Check if a reading is within valid range"""
    min_val, max_val = valid_range
    return min_val <= value <= max_val


def validate_payload(payload: Dict[str, Any]) -> tuple[bool, str]:
    """Validate all sensor readings in the payload"""
    ranges = SensorRanges()

    # Required fields check
    required_fields = [
        "temperature",
        "humidity",
        "battery_voltage",
        "farm_id",
        "sensor_id",
        "timestamp",
        "status",
    ]

    missing_fields = [field for field in required_fields if field not in payload]
    if missing_fields:
        return False, f"Missing required fields: {', '.join(missing_fields)}"

    # Validate numeric ranges
    validations = [
        ("Temperature", payload["temperature"], ranges.temperature),
        ("Humidity", payload["humidity"], ranges.humidity),
        ("Battery voltage", payload["battery_voltage"], ranges.battery),
    ]

    for name, value, valid_range in validations:
        if not validate_reading(value, valid_range):
            return False, f"{name} reading {value} is outside valid range {valid_range}"

    # Validate status
    if payload["status"] not in SOIL_STATUS:
        return False, f"Invalid status: {payload['status']}"

    # Validate IDs
    if not isinstance(payload["farm_id"], int) or payload["farm_id"] < 1:
        return False, f"Invalid farm_id: {payload['farm_id']}"
    if not isinstance(payload["sensor_id"], str):
        return False, f"Invalid sensor_id: {payload['sensor_id']}"

    # Validate timestamp format
    try:
        datetime.strptime(payload["timestamp"], "%Y-%m-%dT%H:%M:%S")
    except ValueError:
        return False, "Invalid timestamp format"

    return True, "Validation successful"


class SensorState:
    """Keeps track of the previous sensor values for smooth, uniform changes."""

    sensor_id: str
    farm_id: int

    def __init__(self):
        ranges = SensorRanges()
        # Start with mid-range values
        self.temperature = (ranges.temperature[0] + ranges.temperature[1]) / 2
        self.humidity = (ranges.humidity[0] + ranges.humidity[1]) / 2
        self.battery_voltage = ranges.battery[1]  # Start fully charged

    def update(self):
        """Update each value by a small random offset, keeping within range."""
        ranges = SensorRanges()
        # Small random offset for each value
        self.temperature = self._offset(self.temperature, ranges.temperature, 0.2)
        self.humidity = self._offset(self.humidity, ranges.humidity, 0.5)
        # Simulate battery drain
        self.battery_voltage = self._offset(
            self.battery_voltage, ranges.battery, -0.01, 0.0
        )

    def _offset(self, value, valid_range, max_delta, min_delta=None):
        """Offset value by a small random delta, clamp to valid range."""
        if min_delta is None:
            min_delta = -max_delta
        delta = random.uniform(min_delta, max_delta)
        new_value = value + delta
        # Clamp to valid range
        new_value = max(valid_range[0], min(valid_range[1], new_value))
        return round(new_value, 2)

    def get_status(self):
        """Determine status based on moisture (example logic)."""
        return random.choice(SOIL_STATUS)

    def to_payload(self):
        return {
            "temperature": self.temperature,
            "humidity": self.humidity,
            "battery_voltage": self.battery_voltage,
            "farm_id": self.farm_id,
            "sensor_id": self.sensor_id,
            "timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
            "status": self.get_status(),
        }


def main():
    print("Starting soil sensor simulation (uniform drift)...")
    print(f"Sending data to: {URL}")
    print(f"Farm ID: {FARM_ID}, Sensor ID: {SENSOR_ID}")
    print("Press Ctrl+C to stop...")
    print("--------------------------------")

    sensor_id = input("Enter sensor ID (default SENSOR_40): ") or SENSOR_ID
    farm_id = input("Enter farm ID (default 2): ") or FARM_ID

    sensor = SensorState()
    sensor.sensor_id = sensor_id
    sensor.farm_id = int(farm_id)

    try:
        while True:
            # Update sensor values with small random offset
            sensor.update()
            payload = sensor.to_payload()
            is_valid, message = validate_payload(payload)

            if not is_valid:
                print(f"Validation Error: {message}")
                print("Skipping invalid reading...")
                print("--------------------------------")
                time.sleep(5)
                continue

            print(f"Sending validated data:")
            for key, value in payload.items():
                print(f"{key}: {value}")

            try:
                response = requests.post(URL, json=payload)
                print(f"\nResponse [{response.status_code}]: {response.text}")
            except requests.RequestException as e:
                print(f"\nError sending request: {e}")

            print("--------------------------------")
            time.sleep(30)

    except KeyboardInterrupt:
        print("\nScript stopped by user.")
        print("Shutting down sensor simulation...")


if __name__ == "__main__":
    main()
