from django.db import models
from django.contrib.auth import get_user_model


User = get_user_model()


class Model(models.Model):
    creator = models.ForeignKey(User, on_delete=models.CASCADE)
    name = models.CharField(max_length=255)
    dataset = models.FileField(upload_to="datasets/", null=True, blank=True)
    is_active = models.BooleanField(default=False)

    date_added = models.DateTimeField(auto_now_add=True)
    date_updated = models.DateTimeField(auto_now=True)

    def get_model_name(self):
        return str(self.name) + "_" + str(self.id)

    def get_model_version(self):
        return f"v0.{self.creator.id}.{self.id}"

    def __str__(self):
        return self.get_model_name()


class Training(models.Model):
    model = models.ForeignKey(Model, on_delete=models.CASCADE)
    data = models.JSONField(default=dict, blank=True)

    date_added = models.DateTimeField(auto_now_add=True)
    date_updated = models.DateTimeField(auto_now=True)

    def __str__(self):
        return self.model.get_model_name() + " training history"
