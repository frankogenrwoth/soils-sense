from django.contrib import admin
from .models import Model, Training


# Register your models here.
@admin.register(Model)
class ModelManagementAdmin(admin.ModelAdmin):
    list_display = ["name", "creator", "is_active", "date_added", "date_updated"]
    list_filter = ["is_active", "creator"]
    search_fields = ["name", "creator__username"]


@admin.register(Training)
class TrainingManagementAdmin(admin.ModelAdmin):
    list_display = ["model", "date_added", "date_updated"]
    list_filter = ["model"]
