import logging
import traceback
import pandas as pd
from ml.config import MODEL_CONFIGS
from typing import Literal

from django.shortcuts import render, redirect
from django.views import View
from django.contrib.auth import get_user_model
from django import forms
from django.shortcuts import get_object_or_404
from django.contrib import messages
from django.http import JsonResponse, HttpResponse
from django.views.decorators.http import require_POST
from django.utils.decorators import method_decorator
from django.views.decorators.csrf import csrf_exempt


from ml import MLEngine
from ml.config import MODEL_CONFIGS
from .forms import UserForm
from .models import Model, Training
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
from authentication.models import Role

logger = logging.getLogger(__name__)
User = get_user_model()
ml_engine = MLEngine()


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
        users = User.objects.all().exclude(role=Role.ADMINISTRATOR)
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
            messages.success(
                request, f"User {user.get_user_name()} updated successfully"
            )
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
        # Get search parameters
        search_query = request.GET.get("search", "")
        data_type = request.GET.get("data_type", "all")

        # Base querysets
        models = Model.objects.all()
        datasets = Training.objects.all()
        farms = Farm.objects.all()
        crops = Crop.objects.all()
        soil_moisture_readings = SoilMoistureReading.objects.all()
        weather_data = WeatherData.objects.all()
        irrigation_events = IrrigationEvent.objects.all()
        prediction_results = PredictionResult.objects.all()
        alerts = Alert.objects.all()
        notifications = Notification.objects.all()

        # Apply search filters if search query exists
        if search_query:
            if data_type == "farms":
                farms = farms.filter(farm_name__icontains=search_query)
            elif data_type == "crops":
                crops = crops.filter(name__icontains=search_query)
            elif data_type == "soil_moisture":
                soil_moisture_readings = soil_moisture_readings.filter(
                    sensor_id__icontains=search_query
                )
            elif data_type == "weather":
                weather_data = weather_data.filter(location__icontains=search_query)
            elif data_type == "irrigation":
                irrigation_events = irrigation_events.filter(
                    farm__farm_name__icontains=search_query
                )
            elif data_type == "predictions":
                prediction_results = prediction_results.filter(
                    farm__farm_name__icontains=search_query
                )
            elif data_type == "alerts":
                alerts = alerts.filter(message__icontains=search_query)
            elif data_type == "notifications":
                notifications = notifications.filter(message__icontains=search_query)

        context = {
            "models": models,
            "datasets": datasets,
            "farms": farms,
            "crops": crops,
            "soil_moisture_readings": soil_moisture_readings,
            "weather_data": weather_data,
            "irrigation_events": irrigation_events,
            "prediction_results": prediction_results,
            "alerts": alerts,
            "notifications": notifications,
            "search_query": search_query,
            "data_type": data_type,
        }
        return render(request, self.template_name, context=context)


class ReportManagementView(View):
    template_name = "administrator/report_management.html"

    def get(self, request):
        context = {}
        return render(request, self.template_name, context=context)


class MLModelManagementView(View):
    template_name = "administrator/ml_model_management.html"

    def get(self, request):
        # Todo: pass a list of models which are standard
        # Todo: filter particulr user trained models
        # Todo: add modal for showing model training history
        # Todo: add ability to retrain the model
        # Todo: add ability to delete the model
        # Todo: add ability to retrain or train the model

        ml_engine = MLEngine()

        available_models = ml_engine.get_available_models()

        user_models = Model.objects.filter(creator=request.user)

        user_models_data = [
            {
                **ml_engine.get_model_info(model.get_model_name()),
                "version": model.get_model_version(),
                "model_name": model.get_model_name(),
            }
            for model in user_models
        ]

        standard_models = [
            ml_engine.get_model_info(model)
            for model in available_models
            if model.find("version") == -1
        ]

        from ml.config import REGRESSION_ALGORITHMS, CLASSIFICATION_ALGORITHMS

        algorithms = [
            algorithm
            for algorithm in list(REGRESSION_ALGORITHMS.keys())
            if algorithm in list(CLASSIFICATION_ALGORITHMS.keys())
        ]

        context = {
            # new context
            "user_models": user_models_data,
            "standard_models": standard_models,
            "algorithms": algorithms,
        }
        return render(request, self.template_name, context=context)


class MLModelDetailView(View):
    template_name = "administrator/ml_model_detail.html"

    def get(self, request, model_name):
        model_algorithm = request.GET.get("algorithm", None)
        if model_algorithm is not None:
            model_name = f"{model_name}_{model_algorithm}"

        models = ml_engine.get_available_models()

        if model_name not in models:
            messages.error(request, "Model not found")
            return redirect("administrator:ml")

        model_info = ml_engine.get_model_info(model_name)

        required_fields = model_info.get("features", [])
        features = MODEL_CONFIGS.get(model_info.get("model_type"), {}).get(
            "features", []
        )
        print(features)
        engineered_fields = [
            "hour_of_day",
            "month",
            "is_growing_season",
            "temp_humidity_interaction",
            "low_battery",
            "irrigation_action",
        ]
        input_fields = [field for field in features if field not in engineered_fields]

        print(input_fields)

        version_number = model_name.split("_")[-1]

        context = {
            "model": ml_engine.get_model_info(model_name),
            "version_number": version_number,
            "input_fields": input_fields,
        }

        return render(request, self.template_name, context=context)


class UploadDatasetView(View):
    def post(self, request):
        dataset = request.FILES.get("dataset")
        model_type: Literal["soil_moisture_predictor", "irrigation_recommendation"] = (
            request.POST.get("model_type")
        )
        algorithm = request.POST.get("algorithm", "random_forest")

        assert model_type in [
            "soil_moisture_predictor",
            "irrigation_recommendation",
        ], "Invalid model type"

        try:
            df = self._clean_dataset(request, dataset, model_type)

            if df is None:
                return redirect("administrator:ml")

            model_object = Model.objects.create(
                creator=request.user, name=f"{model_type}_{algorithm}", dataset=dataset
            )

            ml_engine.train_model(
                model_type,
                algorithm=algorithm,
                custom_data=df,
                version=model_object.get_model_version(),
            )

            messages.success(
                request, "Dataset uploaded and model training started successfully!"
            )

            return redirect(
                "administrator:ml_model_detail",
                model_name=model_object.get_model_name(),
            )

        except Exception as e:
            messages.error(request, f"Error processing dataset: {str(e)}")
            print(traceback.format_exc())
            return HttpResponse({"error": str(e)}, status=400)

    def get(self, request, model_type):
        return redirect("administrator:ml_model_detail", model_type=model_type)

    def _clean_dataset(self, request, file, model_type="soil_moisture_predictor"):
        """
        Clean the dataset and return a dataframe
        """
        if file.size > 10 * 1024 * 1024:
            messages.error(request, "File size exceeds 10MB limit")
            return None

        # Try to read the file as CSV or Excel
        df = None
        if file.content_type == "text/csv":
            try:
                df = pd.read_csv(file)
            except Exception:
                messages.error(request, "Invalid CSV file")
                return None
        elif (
            file.content_type
            == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        ):
            try:
                df = pd.read_excel(file)
            except Exception:
                messages.error(request, "Invalid Excel file")
                return None
        else:
            messages.error(request, "Invalid file type")
            return None

        if df is None or df.empty:
            messages.error(request, "Empty or unreadable file")
            return None

        # Check for required columns
        required_columns = MODEL_CONFIGS.get(model_type, {}).get("features", [])

        engineered_columns = [
            "hour_of_day",
            "month",
            "is_growing_season",
            "temp_humidity_interaction",
            "low_battery",
        ]

        # remove engineered columns from required columns
        required_columns = [
            col for col in required_columns if col not in engineered_columns
        ] + ["timestamp"]

        if required_columns and not all(col in df.columns for col in required_columns):
            messages.error(request, "Required columns are missing")
            return None

        # Return the dataframe for further processing
        return df


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
