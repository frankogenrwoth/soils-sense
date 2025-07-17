from django.views import View
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib import messages
from django.urls import reverse

from apps.farmer.models import (
    Farm,
    Crop,
    SoilMoistureReading,
    WeatherData,
    IrrigationEvent,
    PredictionResult,
    Alert,
    Notification,
)
from apps.technician.models import Sensor

from .data_forms import (
    FarmForm,
    SoilMoistureReadingForm,
    WeatherDataForm,
    IrrigationEventForm,
    PredictionResultForm,
    AlertForm,
    NotificationForm,
    SensorForm,
)


class DataDetailView(View):
    template_name = None
    model = None
    form_class = None
    context_name = None
    success_url = "administrator:data"

    def get_context_data(self, **kwargs):
        context = kwargs
        context["model_name"] = self.model.__name__
        return context

    def get(self, request, pk):
        obj = get_object_or_404(self.model, pk=pk)
        form = self.form_class(instance=obj)
        context = self.get_context_data(**{self.context_name: obj, "form": form})
        return render(request, self.template_name, context)

    def post(self, request, pk):
        obj = get_object_or_404(self.model, pk=pk)
        form = self.form_class(request.POST, request.FILES, instance=obj)

        if form.is_valid():
            form.save()
            messages.success(request, f"{self.model.__name__} updated successfully")
            return redirect(reverse(self.success_url))

        context = self.get_context_data(**{self.context_name: obj, "form": form})
        return render(request, self.template_name, context)


class DataDeleteView(View):
    model = None
    success_url = "administrator:data"

    def post(self, request, pk):
        obj = get_object_or_404(self.model, pk=pk)
        try:
            obj.delete()
            messages.success(request, f"{self.model.__name__} deleted successfully")
        except Exception as e:
            messages.error(request, f"Error deleting {self.model.__name__}: {str(e)}")
        return redirect(reverse(self.success_url))


# Farm Views
class FarmDetailView(DataDetailView):
    model = Farm
    form_class = FarmForm
    context_name = "farm"
    template_name = "administrator/data/farm_detail.html"


class FarmDeleteView(DataDeleteView):
    model = Farm


# Soil Moisture Reading Views
class SoilMoistureReadingDetailView(DataDetailView):
    model = SoilMoistureReading
    form_class = SoilMoistureReadingForm
    context_name = "reading"
    template_name = "administrator/data/soil_moisture_detail.html"


class SoilMoistureReadingDeleteView(DataDeleteView):
    model = SoilMoistureReading


# Weather Data Views
class WeatherDataDetailView(DataDetailView):
    model = WeatherData
    form_class = WeatherDataForm
    context_name = "weather"
    template_name = "administrator/data/weather_detail.html"


class WeatherDataDeleteView(DataDeleteView):
    model = WeatherData


# Irrigation Event Views
class IrrigationEventDetailView(DataDetailView):
    model = IrrigationEvent
    form_class = IrrigationEventForm
    context_name = "event"
    template_name = "administrator/data/irrigation_detail.html"


class IrrigationEventDeleteView(DataDeleteView):
    model = IrrigationEvent


# Prediction Result Views
class PredictionResultDetailView(DataDetailView):
    model = PredictionResult
    form_class = PredictionResultForm
    context_name = "prediction"
    template_name = "administrator/data/prediction_detail.html"


class PredictionResultDeleteView(DataDeleteView):
    model = PredictionResult


# Alert Views
class AlertDetailView(DataDetailView):
    model = Alert
    form_class = AlertForm
    context_name = "alert"
    template_name = "administrator/data/alert_detail.html"


class AlertDeleteView(DataDeleteView):
    model = Alert


# Notification Views
class NotificationDetailView(DataDetailView):
    model = Notification
    form_class = NotificationForm
    context_name = "notification"
    template_name = "administrator/data/notification_detail.html"


class NotificationDeleteView(DataDeleteView):
    model = Notification


class SensorDetailView(DataDetailView):
    model = Sensor
    form_class = SensorForm
    context_name = "sensor"
    template_name = "administrator/data/sensor_detail.html"


class SensorDeleteView(DataDeleteView):
    model = Sensor
