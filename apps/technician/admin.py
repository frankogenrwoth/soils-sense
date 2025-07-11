from django.contrib import admin
from .models import SensorThreshold, Report

# Register your models here.
@admin.register(SensorThreshold)
class SensorThresholdAdmin(admin.ModelAdmin):
    list_display = ('farm', 'parameter', 'min_value', 'max_value', 'unit', 'status')
    list_filter = ('parameter', 'status', 'farm')
    search_fields = ('farm__farm_name', 'parameter')
    list_editable = ('min_value', 'max_value', 'status')

@admin.register(Report)
class ReportAdmin(admin.ModelAdmin):
    list_display = ('title', 'farm', 'report_type', 'created_at', 'generated_by')
    list_filter = ('report_type', 'created_at', 'farm')
    search_fields = ('title', 'farm__farm_name', 'description')
    readonly_fields = ('created_at',)
