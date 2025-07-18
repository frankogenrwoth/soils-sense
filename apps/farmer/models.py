from django.db import models
from authentication.models import User
from django.utils import timezone

class Farm(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    farm_name = models.CharField(max_length=100)
    location = models.CharField(max_length=200)
    area_size = models.DecimalField(max_digits=10, decimal_places=2)  # in acres/hectares
    soil_type = models.CharField(max_length=100)
    date_added = models.DateTimeField(auto_now_add=True)
    description = models.TextField(blank=True, null=True)

    def __str__(self):
        return f"{self.farm_name} - {self.location}"

class Crop(models.Model):
    CROP_STATUS_CHOICES = [
        ('planning', 'Planning'),
        ('planted', 'Planted'),
        ('growing', 'Growing'),
        ('harvested', 'Harvested')
    ]

    farm = models.ForeignKey(Farm, on_delete=models.CASCADE, related_name='crops')
    crop_name = models.CharField(max_length=100)
    variety = models.CharField(max_length=100)
    planting_date = models.DateField()
    expected_harvest_date = models.DateField()
    status = models.CharField(max_length=20, choices=CROP_STATUS_CHOICES, default='planning')
    area_planted = models.DecimalField(max_digits=10, decimal_places=2)  # in acres/hectares
    notes = models.TextField(blank=True, null=True)
    date_added = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.crop_name} at {self.farm.farm_name}"

class SoilMoistureReading(models.Model):
    farm = models.ForeignKey(Farm, on_delete=models.CASCADE, related_name='moisture_readings')
    sensor_id = models.CharField(max_length=50, default='SENSOR_DEFAULT')
    timestamp = models.DateTimeField(default=timezone.now)
    soil_moisture_percent = models.DecimalField(max_digits=5, decimal_places=2, default=0.0)
    temperature_celsius = models.DecimalField(max_digits=5, decimal_places=2, default=0.0)
    humidity_percent = models.DecimalField(max_digits=5, decimal_places=2, default=0.0)
    status = models.CharField(max_length=20, choices=[
        ('Normal', 'Normal'),
        ('Dry', 'Dry'),
        ('Wet', 'Wet'),
        ('Critical Low', 'Critical Low'),
        ('Critical High', 'Critical High'),
    ], default='Normal')
    battery_voltage = models.DecimalField(max_digits=3, decimal_places=2, default=3.3)
    reading_source = models.CharField(max_length=50, choices=[
        ('sensor', 'IoT Sensor'),
        ('manual_input', 'Manual Input'),
        ('csv_upload', 'CSV Upload'),
        ('prediction', 'ML Prediction')
    ], default='manual_input')
    irrigation_action = models.CharField(max_length=100, blank=True, null=True)

    class Meta:
        ordering = ['-timestamp']
        indexes = [
            models.Index(fields=['farm', '-timestamp']),
        ]

    def __str__(self):
        return f"{self.farm.farm_name} - {self.sensor_id} - {self.timestamp}"

class WeatherData(models.Model):
    farm = models.ForeignKey(Farm, on_delete=models.CASCADE, related_name='weather_data')
    timestamp = models.DateTimeField()
    temperature = models.DecimalField(max_digits=5, decimal_places=2)  # in Celsius
    humidity = models.DecimalField(max_digits=5, decimal_places=2)  # in percentage
    precipitation = models.DecimalField(max_digits=5, decimal_places=2)  # in mm
    wind_speed = models.DecimalField(max_digits=5, decimal_places=2)  # in km/h
    is_forecast = models.BooleanField(default=False)  # True for forecast data, False for historical

    class Meta:
        ordering = ['-timestamp']
        indexes = [
            models.Index(fields=['farm', '-timestamp']),
        ]

    def __str__(self):
        return f"{self.farm.farm_name} - {self.timestamp}"

class IrrigationEvent(models.Model):
    STATUS_CHOICES = [
        ('scheduled', 'Scheduled'),
        ('in_progress', 'In Progress'),
        ('completed', 'Completed'),
        ('cancelled', 'Cancelled'),
    ]

    farm = models.ForeignKey(Farm, on_delete=models.CASCADE, related_name='irrigation_events')
    start_time = models.DateTimeField()
    end_time = models.DateTimeField(null=True, blank=True)
    duration = models.IntegerField(help_text="Duration in minutes")
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='scheduled')
    water_amount = models.DecimalField(max_digits=8, decimal_places=2, help_text="Amount of water in liters")
    created_by = models.CharField(max_length=50, choices=[
        ('system', 'System Recommended'),
        ('manual', 'Manual Schedule'),
    ])
    notes = models.TextField(blank=True, null=True)

    class Meta:
        ordering = ['-start_time']
        indexes = [
            models.Index(fields=['farm', '-start_time']),
        ]

    def __str__(self):
        return f"{self.farm.farm_name} - {self.start_time}"

class Alert(models.Model):
    ALERT_TYPES = [
        ('low_moisture', 'Low Moisture'),
        ('high_moisture', 'High Moisture'),
        ('system_error', 'System Error'),
        ('irrigation_needed', 'Irrigation Needed'),
        ('weather_warning', 'Weather Warning'),
    ]
    
    SEVERITY_LEVELS = [
        ('info', 'Information'),
        ('warning', 'Warning'),
        ('critical', 'Critical'),
    ]

    farm = models.ForeignKey(Farm, on_delete=models.CASCADE, related_name='alerts', null=True, blank=True)  # Made nullable
    alert_type = models.CharField(max_length=50, choices=ALERT_TYPES, default='system_error')
    severity = models.CharField(max_length=20, choices=SEVERITY_LEVELS, default='info')  # Added default
    message = models.TextField()  # Previously 'description'
    timestamp = models.DateTimeField(auto_now_add=True)  # Previously 'created_at'
    is_read = models.BooleanField(default=False)  # Previously 'read'
    is_resolved = models.BooleanField(default=False)  # New field
    resolved_at = models.DateTimeField(null=True, blank=True)
    
    class Meta:
        ordering = ['-timestamp']
        indexes = [
            models.Index(fields=['farm', '-timestamp']),
        ]

    def __str__(self):
        return f"{self.farm.farm_name} - {self.alert_type} - {self.timestamp}"
    
class PredictionResult(models.Model):
    farm = models.ForeignKey('Farm', on_delete=models.CASCADE)
    created_at = models.DateTimeField(default=timezone.now)
    location = models.CharField(max_length=255)
    temperature = models.FloatField()
    humidity = models.FloatField()
    battery_voltage = models.FloatField()
    soil_moisture_result = models.FloatField()
    irrigation_result = models.TextField()
    algorithm = models.CharField(max_length=100)
    algorithm_irr = models.CharField(max_length=100)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f"Prediction for {self.farm.farm_name} at {self.created_at}"

    def get_moisture_status(self):
        # Define thresholds for soil moisture
        LOW_THRESHOLD = 30.0
        HIGH_THRESHOLD = 70.0
        
        if self.soil_moisture_result < LOW_THRESHOLD:
            return "low", "Warning: Soil moisture is critically low"
        elif self.soil_moisture_result > HIGH_THRESHOLD:
            return "high", "Warning: Soil moisture is too high"
        else:
            return "normal", "Soil moisture is within normal range"

    def save(self, *args, **kwargs):
        is_new = self.pk is None
        super().save(*args, **kwargs)
        
        if is_new:
            # Get moisture status and warning message
            status, warning = self.get_moisture_status()
            
            # Create detailed notification message
            message = (
                f"New prediction for {self.farm.farm_name}:\n"
                f"Soil Moisture: {self.soil_moisture_result:.1f}%\n"
                f"{warning}\n"
                f"Recommendation: {self.irrigation_result}"
            )
            
            # Create notification with appropriate type based on moisture status
            notification_type = 'warning' if status in ['low', 'high'] else 'prediction'
            
            Notification.objects.create(
                user=self.farm.user,
                message=message,
                notification_type=notification_type
            )

class Notification(models.Model):
    user = models.ForeignKey('authentication.User', on_delete=models.CASCADE)
    message = models.TextField()
    notification_type = models.CharField(
        max_length=20,
        choices=[
            ('prediction', 'Prediction'),
            ('warning', 'Warning'),
            ('info', 'Information')
        ],
        default='prediction'
    )
    is_read = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f"{self.message} - {self.created_at}"
