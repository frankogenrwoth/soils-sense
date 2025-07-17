from django import forms
from apps.farmer.models import Farm, SoilMoistureReading
from .models import SensorThreshold, Report
from authentication.models import User
from .models import Sensor

class FarmEditForm(forms.ModelForm):
    class Meta:
        model = Farm
        fields = ['farm_name', 'location', 'area_size', 'description', 'soil_type']
        widgets = {
            'farm_name': forms.TextInput(attrs={'class': 'form-control'}),
            'location': forms.TextInput(attrs={'class': 'form-control'}),
            'area_size': forms.NumberInput(attrs={'class': 'form-control', 'step': '0.01', 'min': '0'}),
            'description': forms.Textarea(attrs={'class': 'form-control', 'rows': '4'}),
            'soil_type': forms.Select(attrs={'class': 'form-select'}),
        }

class SensorThresholdForm(forms.ModelForm):
    farm = forms.ModelChoiceField(
        queryset=Farm.objects.all(),
        widget=forms.Select(attrs={'class': 'form-select'}),
        label='Farm'
    )
    
    class Meta:
        model = SensorThreshold
        fields = ['farm', 'parameter', 'min_value', 'max_value', 'unit', 'status']
        widgets = {
            'parameter': forms.TextInput(attrs={'class': 'form-control'}),
            'min_value': forms.NumberInput(attrs={'class': 'form-control', 'step': '0.01'}),
            'max_value': forms.NumberInput(attrs={'class': 'form-control', 'step': '0.01'}),
            'unit': forms.TextInput(attrs={'class': 'form-control'}),
            'status': forms.Select(attrs={'class': 'form-select'}),
        }

    def clean(self):
        cleaned_data = super().clean()
        min_value = cleaned_data.get('min_value')
        max_value = cleaned_data.get('max_value')
        
        if min_value is not None and max_value is not None and min_value >= max_value:
            raise forms.ValidationError("Minimum value must be less than maximum value.")
        
        return cleaned_data

class ReportForm(forms.ModelForm):
    farm = forms.ModelChoiceField(
        queryset=Farm.objects.all(),
        widget=forms.Select(attrs={'class': 'form-select'}),
        label='Farm'
    )
    
    class Meta:
        model = Report
        fields = ['farm', 'report_type', 'title', 'description', 'file']
        widgets = {
            'report_type': forms.Select(attrs={'class': 'form-select'}),
            'title': forms.TextInput(attrs={'class': 'form-control'}),
            'description': forms.Textarea(attrs={'class': 'form-control', 'rows': '4'}),
            'file': forms.FileInput(attrs={'class': 'form-control'}),
        }

class SoilReadingFilterForm(forms.Form):
    farm = forms.ModelChoiceField(
        queryset=Farm.objects.all(),
        required=False,
        empty_label="All Farms",
        widget=forms.Select(attrs={'class': 'form-select'})
    )
    date_from = forms.DateField(
        required=False,
        widget=forms.DateInput(attrs={'class': 'form-control', 'type': 'date'})
    )
    date_to = forms.DateField(
        required=False,
        widget=forms.DateInput(attrs={'class': 'form-control', 'type': 'date'})
    )
    status = forms.ChoiceField(
        choices=[('', 'All Statuses')] + SoilMoistureReading._meta.get_field('status').choices,
        required=False,
        widget=forms.Select(attrs={'class': 'form-select'})
    ) 

class TechnicianProfileForm(forms.ModelForm):
    class Meta:
        model = User
        fields = ['first_name', 'last_name', 'email', 'image']
        widgets = {
            'first_name': forms.TextInput(attrs={'class': 'form-control'}),
            'last_name': forms.TextInput(attrs={'class': 'form-control'}),
            'email': forms.EmailInput(attrs={'class': 'form-control'}),
            'image': forms.FileInput(attrs={'class': 'form-control'}),
        } 

class SensorForm(forms.ModelForm):
    farm = forms.ModelChoiceField(
        queryset=Farm.objects.filter(user__isnull=False),
        widget=forms.Select(attrs={'class': 'form-select'}),
        label='Farm'
    )

    class Meta:
        model = Sensor
        fields = ['sensor_id', 'farm', 'description', 'is_active']
        widgets = {
            'sensor_id': forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'e.g., SOIL_001'}),
            'description': forms.Textarea(attrs={'class': 'form-control', 'rows': 2, 'placeholder': 'e.g., North field sensor near irrigation point'}),
            'is_active': forms.CheckboxInput(attrs={'class': 'form-check-input'}),
        }
