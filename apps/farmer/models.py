from django.db import models
from authentication.models import User

class SoilDataFile(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    file = models.FileField(upload_to='soil_data/')
    file_name = models.CharField(max_length=255)
    file_type = models.CharField(max_length=10)  # csv or json
    file_size = models.IntegerField()  # in bytes
    upload_date = models.DateTimeField(auto_now_add=True)

    def get_file_size_mb(self):
        """Return file size in MB with 2 decimal places"""
        return round(self.file_size / (1024 * 1024), 2)

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
