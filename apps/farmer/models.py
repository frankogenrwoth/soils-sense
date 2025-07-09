from django.db import models
from django.conf import settings

class SoilDataFile(models.Model):
    STATUS_CHOICES = [
        ('uploaded', 'Uploaded'),
        ('processing', 'Processing'),
        ('processed', 'Processed'),
        ('error', 'Error'),
    ]

    FILE_TYPE_CHOICES = [
        ('csv', 'CSV'),
        ('json', 'JSON'),
    ]

    file = models.FileField(upload_to='soil_data/')
    file_name = models.CharField(max_length=255)
    file_type = models.CharField(max_length=4, choices=FILE_TYPE_CHOICES)
    file_size = models.IntegerField()  # Size in bytes
    upload_date = models.DateTimeField(auto_now_add=True)
    status = models.CharField(max_length=10, choices=STATUS_CHOICES, default='uploaded')
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)

    def get_file_size_mb(self):
        """Return file size in MB with 2 decimal places"""
        return f"{self.file_size / (1024 * 1024):.2f}"

    class Meta:
        ordering = ['-upload_date']

    def __str__(self):
        return f"{self.file_name} ({self.get_file_size_mb()} MB)"
