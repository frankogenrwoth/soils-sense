from django.db import models
from django.contrib.auth.models import AbstractUser
from authentication.managers import UserManager


class Role(models.TextChoices):
    ADMINISTRATOR = "administrator"
    FARMER = "farmer"
    TECHNICIAN = "technician"


# Create your models here.
class User(AbstractUser):
    role = models.CharField(max_length=20, choices=Role.choices, default=Role.FARMER)
    image = models.ImageField(
        upload_to="users/profiles/",
        default="users/profiles/default.webp",
        null=True,
        blank=True,
    )
    phone_number = models.CharField(max_length=15, null=True, blank=True)

    objects = UserManager()

    def __str__(self):
        return self.username

    def get_user_name(self):
        if self.username:
            return self.username

        if self.first_name and self.last_name:
            return f"{self.first_name} {self.last_name}"

        return self.email

    def get_full_name(self):
        return f"{self.first_name} {self.last_name}"

    def get_short_name(self):
        return self.username
