from django.db import models
from apps.farmer.models import Farm

class Sensor(models.Model):
    sensor_id = models.CharField(max_length=50, unique=True)
    farm = models.ForeignKey(Farm, on_delete=models.CASCADE, related_name='sensors')
    description = models.CharField(max_length=255, blank=True, null=True)
    is_active = models.BooleanField(default=True)
    installed_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.sensor_id} ({getattr(self.farm, 'farm_name', str(self.farm))})"

class SensorThreshold(models.Model):
    farm = models.ForeignKey(Farm, on_delete=models.CASCADE, related_name='sensor_thresholds')
    parameter = models.CharField(max_length=50)  # e.g., 'Soil Moisture', 'pH Level', 'Temperature'
    min_value = models.FloatField()
    max_value = models.FloatField()
    unit = models.CharField(max_length=20, default='')
    status = models.CharField(max_length=20, default='Normal')  # e.g., Normal, Warning, Critical

    def __str__(self):
        return f"{getattr(self.farm, 'farm_name', str(self.farm))} - {self.parameter}"

class Report(models.Model):
    REPORT_TYPES = [
        ('soil_analysis', 'Soil Analysis'),
        ('sensor_maintenance', 'Sensor Maintenance'),
        ('custom', 'Custom'),
        ('prediction', 'Prediction'),
    ]
    farm = models.ForeignKey(Farm, on_delete=models.CASCADE, related_name='reports')
    report_type = models.CharField(max_length=50, choices=REPORT_TYPES, default='custom')
    title = models.CharField(max_length=200)
    description = models.TextField(blank=True, null=True)
    file = models.FileField(upload_to='reports/', blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)
    generated_by = models.CharField(max_length=100, default='System')

    def __str__(self):
        return f"{self.title} - {getattr(self.farm, 'farm_name', str(self.farm))}"
