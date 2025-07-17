from django.contrib import admin

# Register your models here.
from apps.farmer.models import Farm, Crop, PredictionResult



@admin.register(Farm)
class FarmAdmin(admin.ModelAdmin):
    list_display = ('farm_name', 'area_size', 'location')

    search_fields = ('farm_name', 'location')
    list_filter = ('location',)
    ordering = ('farm_name',)
    list_per_page = 10
    list_display_links = ('farm_name', 'location')
    list_editable = ('area_size',)
    list_max_show_all = 10

@admin.register(Crop)
class CropAdmin(admin.ModelAdmin):
    list_display = ('crop_name', 'variety', 'planting_date', 'expected_harvest_date')

@admin.register(PredictionResult)
class PredictionResultAdmin(admin.ModelAdmin):
    list_display = ('temperature', 'humidity', 'soil_moisture_result', "irrigation_result", "algorithm")

