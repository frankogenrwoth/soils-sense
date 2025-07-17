from django import forms
from apps.farmer.models import (
    Farm,
    SoilMoistureReading,
    WeatherData,
    IrrigationEvent,
    PredictionResult,
    Alert,
    Notification,
)


class BaseDataForm(forms.ModelForm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for field_name, field in self.fields.items():
            field.widget.attrs["class"] = (
                "w-full border border-gray-300 rounded px-3 py-2 "
                "focus:outline-none focus:ring-2 focus:ring-green-200"
            )
            if isinstance(field.widget, forms.DateTimeInput):
                field.widget = forms.DateTimeInput(
                    attrs={
                        "type": "datetime-local",
                        "class": "w-full border border-gray-300 rounded px-3 py-2 "
                        "focus:outline-none focus:ring-2 focus:ring-green-200",
                    }
                )


class FarmForm(BaseDataForm):
    class Meta:
        model = Farm
        fields = "__all__"


class SoilMoistureReadingForm(BaseDataForm):
    class Meta:
        model = SoilMoistureReading
        fields = "__all__"


class WeatherDataForm(BaseDataForm):
    class Meta:
        model = WeatherData
        fields = "__all__"


class IrrigationEventForm(BaseDataForm):
    class Meta:
        model = IrrigationEvent
        fields = "__all__"


class PredictionResultForm(BaseDataForm):
    class Meta:
        model = PredictionResult
        fields = "__all__"


class AlertForm(BaseDataForm):
    class Meta:
        model = Alert
        fields = "__all__"


class NotificationForm(BaseDataForm):
    class Meta:
        model = Notification
        fields = "__all__"
