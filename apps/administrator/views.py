import logging

from django.shortcuts import render, redirect
from django.views import View
from django.contrib.auth import get_user_model
from django import forms
from django.shortcuts import get_object_or_404
from django.contrib import messages
from django.http import JsonResponse
from django.views.decorators.http import require_POST
from django.utils.decorators import method_decorator
from django.views.decorators.csrf import csrf_exempt


from ml import MLEngine
from .forms import UserForm

logger = logging.getLogger(__name__)
User = get_user_model()


class DashboardView(View):
    template_name = "administrator/dashboard.html"

    def get(self, request):
        context = {}
        return render(request, self.template_name, context=context)


class UserManagementView(View):
    """Admin should be able to manage other users on the platform (view, create, edit, delete, assign roles, reset passwords)."""

    template_name = "administrator/user_management.html"
    form_class = UserForm

    def get(self, request):
        users = User.objects.all()
        form = self.form_class()

        context = {
            "users": users,
            "form": form,
        }

        return render(request, self.template_name, context=context)

    def post(self, request):
        users = User.objects.all()
        form = self.form_class(request.POST, request.FILES)

        if form.is_valid():
            user = form.save(commit=False)
            # Hash the password if provided
            if form.cleaned_data.get("password"):
                user.set_password(form.cleaned_data["password"])
            user.save()
            messages.success(request, "User created successfully")
            return redirect("administrator:users")

        print(form.errors)

        context = {
            "users": users,
            "form": form,
        }
        return render(request, self.template_name, context=context)


class UserDetailView(View):
    template_name = "administrator/user_detail.html"

    def get(self, request, pk):
        user = get_object_or_404(User, id=pk)
        context = {
            "user": user,
        }

        return render(request, self.template_name, context=context)


class UserUpdateView(View):
    """View for updating user information"""

    def post(self, request, pk):
        user = get_object_or_404(User, id=pk)
        form = UserForm(request.POST, request.FILES, instance=user)

        if form.is_valid():
            user = form.save(commit=False)
            # Only update password if a new one is provided
            if form.cleaned_data.get("password"):
                user.set_password(form.cleaned_data["password"])
            user.save()
            messages.success(request, f"User {user.get_user_name()} updated successfully")
            return redirect("administrator:users")
        else:
            messages.error(request, "Failed to update user. Please check the form.")
            return redirect("administrator:users")


class UserDeleteView(View):
    """View for deleting users with confirmation"""

    def post(self, request, pk):
        user = get_object_or_404(User, id=pk)
        user_name = user.get_user_name()

        # Prevent admin from deleting themselves
        if user == request.user:
            messages.error(request, "You cannot delete your own account")
            return redirect("administrator:users")

        try:
            user.delete()
            messages.success(request, f"User {user_name} deleted successfully")
        except Exception as e:
            messages.error(request, f"Failed to delete user: {str(e)}")

        return redirect("administrator:users")


class DataManagementView(View):
    template_name = "administrator/data_management.html"

    def get(self, request):
        context = {}
        return render(request, self.template_name, context=context)


class ReportManagementView(View):
    template_name = "administrator/report_management.html"

    def get(self, request):
        context = {}
        return render(request, self.template_name, context=context)


class MLModelManagementView(View):
    template_name = "administrator/ml_model_management.html"

    class SoilMoistureForm(forms.Form):
        sensor_id = forms.CharField()
        location = forms.CharField()
        temperature_celsius = forms.FloatField()
        humidity_percent = forms.FloatField()
        battery_voltage = forms.FloatField()
        status = forms.CharField()
        irrigation_action = forms.CharField()
        timestamp = forms.DateTimeField()

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            for field in self.fields.values():
                field.widget.attrs["class"] = (
                    "block w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-green-500 focus:border-green-500 sm:text-sm"
                )

    class IrrigationRecommendationForm(forms.Form):
        soil_moisture_percent = forms.FloatField()
        temperature_celsius = forms.FloatField()
        humidity_percent = forms.FloatField()
        battery_voltage = forms.FloatField()
        status = forms.CharField()
        timestamp = forms.DateTimeField()

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            for field in self.fields.values():
                field.widget.attrs["class"] = (
                    "block w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-green-500 focus:border-green-500 sm:text-sm"
                )

    def get(self, request):
        # Todo: pass a list of models which are standard
        # Todo: filter particulr user trained models
        # Todo: add modal for showing model training history
        # Todo: add ability to retrain the model
        # Todo: add ability to delete the model
        # Todo: add ability to retrain or train the model

        
        soil_moisture_form = self.SoilMoistureForm()
        irrigation_recommendation_form = self.IrrigationRecommendationForm()

        ml_engine = MLEngine()

        available_models = ml_engine.get_available_models()

        models_data = [ml_engine.get_model_info(model) for model in available_models]

        context = {
            "available_models": models_data,
            "soil_moisture_form": soil_moisture_form,
            "irrigation_recommendation_form": irrigation_recommendation_form,
            "predicted_soil_moisture": None,
            "predicted_irrigation_recommendation": None,
        }
        return render(request, self.template_name, context=context)

    def post(self, request):
        data = request.POST

        ml_engine = MLEngine()

        if "sensor_id" in data:
            soil_moisture_form = self.SoilMoistureForm(data)
            irrigation_recommendation_form = self.IrrigationRecommendationForm()
            if soil_moisture_form.is_valid():

                predicted_soil_moisture = ml_engine.predict_soil_moisture(
                    **soil_moisture_form.cleaned_data
                )

            else:
                predicted_soil_moisture = None

            predicted_irrigation_recommendation = None
        else:
            soil_moisture_form = self.SoilMoistureForm()
            irrigation_recommendation_form = self.IrrigationRecommendationForm(data)

            if irrigation_recommendation_form.is_valid():
                predicted_irrigation_recommendation = ml_engine.recommend_irrigation(
                    **irrigation_recommendation_form.cleaned_data
                )

            else:
                predicted_irrigation_recommendation = None

            predicted_soil_moisture = None

        available_models = ml_engine.get_available_models()

        models_data = [ml_engine.get_model_info(model) for model in available_models]

        context = {
            "available_models": models_data,
            "soil_moisture_form": soil_moisture_form,
            "irrigation_recommendation_form": irrigation_recommendation_form,
            "predicted_soil_moisture": (
                predicted_soil_moisture["predicted_value"]
                if predicted_soil_moisture
                else None
            ),
            "predicted_irrigation_recommendation": (
                predicted_irrigation_recommendation["predicted_value"]
                if predicted_irrigation_recommendation
                else None
            ),
        }
        return render(request, self.template_name, context=context)


class NotificationView(View):
    template_name = "administrator/notification.html"

    def get(self, request):
        context = {}
        return render(request, self.template_name, context=context)


class SensorView(View):
    template_name = "administrator/sensor_management.html"

    def get(self, request):
        context = {}
        return render(request, self.template_name, context=context)
