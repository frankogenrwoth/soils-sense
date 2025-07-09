from django.db import models
from django.contrib.auth.models import AbstractUser, UserManager


class Role(models.TextChoices):
    ADMIN = "admin"
    FARMER = "farmer"
    TECHNICIAN = "technician"

# Create your models here.
class User(AbstractUser):
    role = models.CharField(max_length=20, choices=Role.choices, default=Role.FARMER)
    image = models.ImageField(
        upload_to="users/profile_images/",
        default="users/profile_images/default.webp",
        null=True,
        blank=True,
    )
    phone_number = models.CharField(max_length=15, null=True, blank=True)

    objects = UserManager()

    def __str__(self):
        return self.username

    def get_full_name(self):
        return f"{self.first_name} {self.last_name}"

    def get_short_name(self):
        return self.username
