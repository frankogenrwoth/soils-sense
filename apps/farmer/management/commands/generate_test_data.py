from django.core.management.base import BaseCommand
from django.utils import timezone
from apps.farmer.models import SoilMoistureReading, Farm
from datetime import timedelta
import random

class Command(BaseCommand):
    help = 'Generates test soil moisture readings data for the past 7 days'

    def handle(self, *args, **options):
        # Get all farms
        farms = Farm.objects.all()
        if not farms.exists():
            self.stdout.write(self.style.ERROR('No farms found. Please create a farm first.'))
            return

        # Generate readings for each farm
        for farm in farms:
            # Generate data for the past 7 days
            end_date = timezone.now()
            start_date = end_date - timedelta(days=7)
            current_date = start_date

            # Base values for realistic data generation
            base_moisture = random.uniform(45, 55)
            base_temperature = random.uniform(20, 25)
            base_humidity = random.uniform(60, 70)

            while current_date <= end_date:
                # Add some random variation to the base values
                moisture = max(0, min(100, base_moisture + random.uniform(-5, 5)))
                temperature = base_temperature + random.uniform(-2, 2)
                humidity = max(0, min(100, base_humidity + random.uniform(-5, 5)))

                # Create the reading
                SoilMoistureReading.objects.create(
                    farm=farm,
                    timestamp=current_date,
                    soil_moisture_percent=moisture,
                    temperature_celsius=temperature,
                    humidity_percent=humidity,
                    sensor_id='TEST_SENSOR_01',
                    reading_source='sensor'
                )

                # Move to next hour
                current_date += timedelta(hours=1)
                
                # Slightly adjust base values for natural variation
                base_moisture += random.uniform(-0.5, 0.5)
                base_temperature += random.uniform(-0.2, 0.2)
                base_humidity += random.uniform(-0.5, 0.5)

            self.stdout.write(
                self.style.SUCCESS(
                    f'Successfully generated test data for farm: {farm.farm_name}'
                )
            ) 