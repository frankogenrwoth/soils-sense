import logging
import traceback
import pandas as pd
from ml.config import MODEL_CONFIGS
from typing import Literal
from io import BytesIO
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.platypus import Table, TableStyle, SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from django.urls import reverse

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
from apps.technician.models import Sensor
from authentication.models import Role
from django.db.models import Avg
from django.utils import timezone

logger = logging.getLogger(__name__)
User = get_user_model()
ml_engine = MLEngine()


class DashboardView(View):
    template_name = "administrator/dashboard.html"

    def get(self, request):
        # Get user statistics
        total_users = User.objects.count()
        farmers = User.objects.filter(role=Role.FARMER).count()
        technicians = User.objects.filter(role=Role.TECHNICIAN).count()

        # Get sensor statistics
        total_sensors = Sensor.objects.count()
        active_sensors = 0
        total_readings = 0
        avg_moisture = None

        # Calculate sensor stats for the last 24 hours
        recent_readings = SoilMoistureReading.objects.filter(
            timestamp__gte=timezone.now() - timezone.timedelta(hours=24)
        )
        if recent_readings.exists():
            # Count sensors with readings in last 24 hours as active
            active_sensors = recent_readings.values("sensor_id").distinct().count()
            total_readings = recent_readings.count()
            avg_moisture = recent_readings.aggregate(Avg("soil_moisture_percent"))[
                "soil_moisture_percent__avg"
            ]
            if avg_moisture:
                avg_moisture = round(avg_moisture, 1)

        # Get farm statistics
        total_farms = Farm.objects.count()

        # Get notification and alert statistics
        total_notifications = Notification.objects.count()
        unread_notifications = Notification.objects.filter(is_read=False).count()
        active_alerts = Alert.objects.filter(is_read=False).count()

        # Get recent activities (last 10)
        recent_notifications = Notification.objects.all().order_by("-created_at")[:5]
        recent_alerts = Alert.objects.all().order_by("-timestamp")[:5]
        recent_readings = SoilMoistureReading.objects.all().order_by("-timestamp")[:5]

        context = {
            # User stats
            "total_users": total_users,
            "farmers": farmers,
            "technicians": technicians,
            # Sensor stats
            "total_sensors": total_sensors,
            "active_sensors": active_sensors,
            "total_readings": total_readings,
            "avg_moisture": avg_moisture,
            # Farm stats
            "total_farms": total_farms,
            # Notification stats
            "total_notifications": total_notifications,
            "unread_notifications": unread_notifications,
            "active_alerts": active_alerts,
            # Recent activities
            "recent_notifications": recent_notifications,
            "recent_alerts": recent_alerts,
            "recent_readings": recent_readings,
        }
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
        soil_moisture_readings = SoilMoistureReading.objects.all()[:5]
        weather_data = WeatherData.objects.all()
        irrigation_events = IrrigationEvent.objects.all()
        prediction_results = PredictionResult.objects.all()
        alerts = Alert.objects.all()
        notifications = Notification.objects.all()
        sensors = Sensor.objects.all()
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
            "sensors": sensors,
        }
        return render(request, self.template_name, context=context)


class ServerLogView(View):
    template_name = "administrator/server_log.html"

    def get(self, request):
        context = {}

        with open("logs/soilsense.log", "r") as file:
            log_content = file.read()

        context["log_content"] = "\n".join(log_content.splitlines()[-200:])

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
        predicted_value = request.GET.get("predicted_value", None)
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
        engineered_fields = [
            "hour_of_day",
            "month",
            "is_growing_season",
            "temp_humidity_interaction",
            "low_battery",
            "irrigation_action",
        ]
        input_fields = [field for field in features if field not in engineered_fields]

        version_number = model_name.split("_")[-1] if "v0." in model_name else ""

        context = {
            "model": ml_engine.get_model_info(model_name),
            "version_number": version_number,
            "input_fields": input_fields,
            "predicted_value": predicted_value,
        }

        return render(request, self.template_name, context=context)

    def post(self, request, model_name):
        model_super_name = model_name
        model_algorithm = request.GET.get("algorithm", None)
        version = request.GET.get("version", None)
        meta = request.POST.get("meta", None)
        if meta is not None:
            meta = meta.split("&")
            meta = {k: v for k, v in [item.split("=") for item in meta]}

            version = meta.get("version", None)
            model_algorithm = meta.get("algorithm", None)
            model_name = meta.get("model", None)

        data = {k: v for k, v in request.POST.items() if v != ""}
        cleaned_data = {}
        for key, value in data.items():
            if key in ["action", "irrigation_action"]:
                continue
            try:
                cleaned_data[key] = float(value)
            except ValueError:
                cleaned_data[key] = value

        cleaned_data["timestamp"] = datetime.now().isoformat()

        if data.get("action") == "predict":
            # fill in data
            cleaned_data["irrigation_action"] = "Irrigate"

            res = ml_engine.predict(
                model_type=model_name,
                data=cleaned_data,
                version=version,
                algorithm=model_algorithm,
            )
            predicted_value = res.get("predicted_value")
            messages.success(
                request,
                f"Prediction successful: {predicted_value} {model_algorithm.title().replace('_', ' ')}",
            )

            url = reverse("administrator:ml_model_detail", args=[model_super_name])
            if not meta:
                return redirect(
                    f"{url}?algorithm={model_algorithm}&version={version}&predicted_value={predicted_value}"
                )
            else:
                return redirect(f"{url}?predicted_value={predicted_value}")

        elif data.get("action") == "retrain":
            dataset = request.FILES.get("dataset")
            df = self._clean_dataset(request, dataset, model_name)

            if df is None:
                return redirect("administrator:ml")

            res = ml_engine.train_model(
                model_type=model_name, custom_data=df, version=version
            )

            messages.success(request, "Model training successful")
            url = reverse("administrator:ml_model_detail", args=[model_super_name])
            return redirect(f"{url}?algorithm={model_algorithm}&version={version}")

        else:
            messages.error(request, "Invalid action")
            return redirect("administrator:ml")

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
        # Get notifications and alerts, ordered by most recent first
        notifications = Notification.objects.all()
        alerts = Alert.objects.all()

        context = {
            "notifications": notifications,
            "alerts": alerts,
        }
        return render(request, self.template_name, context=context)


class SensorView(View):
    template_name = "administrator/sensor_management.html"

    def get(self, request):
        # Get all sensors and their latest readings
        sensors = Sensor.objects.all()

        # Create a dict to store sensor stats
        sensor_stats = {}
        for sensor in sensors:
            # Get the latest 24 hours of readings for this sensor
            recent_readings = SoilMoistureReading.objects.filter(
                sensor_id=sensor.sensor_id,
                timestamp__gte=timezone.now() - timezone.timedelta(hours=24),
            )

            # Calculate stats
            reading_count = recent_readings.count()
            avg_moisture = recent_readings.aggregate(Avg("soil_moisture_percent"))[
                "soil_moisture_percent__avg"
            ]
            latest_reading = recent_readings.first()

            sensor_stats[sensor.id] = {
                "reading_count": reading_count,
                "avg_moisture": round(avg_moisture, 2) if avg_moisture else None,
                "latest_reading": latest_reading,
                "status": "Active" if reading_count > 0 else "Inactive",
            }

        context = {
            "sensors": sensors,
            "sensor_stats": sensor_stats,
        }
        return render(request, self.template_name, context=context)


class PrintReportView(View):
    def get(self, request, model_name):
        algorithm = request.GET.get("algorithm", None)
        version = request.GET.get("version", None)

        if version is None:
            model_name = f"{model_name}_{algorithm}"
        else:
            model_name = f"{model_name}_{algorithm}_version_{version}"

        model_info = ml_engine.get_model_info(model_name)

        buffer = self._generate_pdf(model_info)

        response = HttpResponse(buffer, content_type="application/pdf")
        response["Content-Disposition"] = 'attachment; filename="model_report.pdf"'
        return response

    def _generate_pdf(self, model_info):
        """
        Generate a visually appealing, greyscale PDF report for the given model_info dictionary.
        Uses PyPDF2 and reportlab for page creation. All content is left-aligned and uses only greyscale.
        Returns a BytesIO object containing the PDF.
        """
        from io import BytesIO
        from reportlab.pdfgen import canvas
        from reportlab.lib.pagesizes import letter
        from reportlab.lib.units import inch
        from reportlab.lib.colors import black, grey, lightgrey, white, HexColor
        from PyPDF2 import PdfReader, PdfWriter

        # Define a palette of greys for visual appeal
        GREY_DARK = HexColor("#222222")
        GREY_MED = HexColor("#888888")
        GREY_LIGHT = HexColor("#DDDDDD")
        GREY_TABLE_ALT = HexColor("#F5F5F5")

        buffer = BytesIO()
        c = canvas.Canvas(buffer, pagesize=letter)
        width, height = letter

        # Margins and layout
        left_margin = 0.7 * inch
        right_margin = 0.7 * inch
        top_margin = 0.8 * inch
        bottom_margin = 0.8 * inch
        usable_width = width - left_margin - right_margin

        y = height - top_margin

        def check_page_break(row_height=0.22 * inch):
            nonlocal y
            if y < bottom_margin + row_height:
                c.showPage()
                y = height - top_margin

        def add_table_spacing():
            nonlocal y
            y -= 0.18 * inch  # Increased spacing between tables

        def draw_title(text):
            nonlocal y
            c.setFont("Helvetica-Bold", 20)
            c.setFillColor(GREY_DARK)
            c.drawString(left_margin, y, text)
            y -= 0.45 * inch
            c.setStrokeColor(GREY_DARK)
            c.setLineWidth(1.2)
            c.line(
                left_margin,
                y + 0.18 * inch,
                left_margin + usable_width,
                y + 0.18 * inch,
            )
            y -= 0.08 * inch

        def draw_section_header(text):
            nonlocal y
            add_table_spacing()
            check_page_break(0.3 * inch)
            c.setFont("Helvetica-Bold", 14)
            c.setFillColor(GREY_MED)
            c.drawString(left_margin, y, text)
            y -= 0.22 * inch
            c.setStrokeColor(GREY_LIGHT)
            c.setLineWidth(0.7)
            c.line(
                left_margin,
                y + 0.13 * inch,
                left_margin + usable_width,
                y + 0.13 * inch,
            )
            y -= 0.06 * inch

        def draw_subheader(text):
            nonlocal y
            add_table_spacing()
            check_page_break(0.18 * inch)
            c.setFont("Helvetica-Bold", 11)
            c.setFillColor(GREY_DARK)
            c.drawString(left_margin, y, text)
            y -= 0.16 * inch

        def draw_text(text, font="Helvetica", size=10, color=GREY_DARK, spacing=0.18):
            nonlocal y
            add_table_spacing()
            check_page_break(spacing * inch)
            c.setFont(font, size)
            c.setFillColor(color)
            c.drawString(left_margin, y, text)
            y -= spacing * inch

        def draw_table(data, col_widths, header_grey=True, alt_rows=True, font_size=10):
            nonlocal y
            add_table_spacing()
            row_height = 0.20 * inch
            table_top = y
            n_cols = len(col_widths)
            c.setFont("Helvetica", font_size)
            for i, row in enumerate(data):
                check_page_break(row_height)
                x = left_margin
                # Header row
                if i == 0 and header_grey:
                    c.setFillColor(GREY_LIGHT)
                    c.rect(
                        left_margin,
                        y - 2,
                        sum(col_widths),
                        row_height,
                        fill=1,
                        stroke=0,
                    )
                    c.setFillColor(GREY_DARK)
                    c.setFont("Helvetica-Bold", font_size)
                else:
                    # Alternate row shading for readability
                    if alt_rows and i % 2 == 1:
                        c.setFillColor(GREY_TABLE_ALT)
                        c.rect(
                            left_margin,
                            y - 2,
                            sum(col_widths),
                            row_height,
                            fill=1,
                            stroke=0,
                        )
                    c.setFillColor(GREY_DARK)
                    c.setFont("Helvetica", font_size)
                for j, cell in enumerate(row):
                    cell_str = str(cell)
                    # Truncate if too long
                    max_chars = int(col_widths[j] // (font_size * 0.5))
                    if len(cell_str) > max_chars:
                        cell_str = cell_str[: max_chars - 3] + "..."
                    c.drawString(x + 4, y, cell_str)
                    x += col_widths[j]
                y -= row_height
                # Draw horizontal line
                c.setStrokeColor(GREY_LIGHT)
                c.setLineWidth(0.5)
                c.line(
                    left_margin,
                    y + row_height - 2,
                    left_margin + sum(col_widths),
                    y + row_height - 2,
                )
            # Draw vertical lines
            x = left_margin
            for w in col_widths:
                c.setStrokeColor(GREY_LIGHT)
                c.setLineWidth(0.5)
                c.line(x, table_top, x, y + row_height)
                x += w
            c.line(
                left_margin + sum(col_widths),
                table_top,
                left_margin + sum(col_widths),
                y + row_height,
            )
            y -= 0.10 * inch

        # Title
        title = f"ML Model Report: {model_info.get('model_name', 'Unknown')}"
        draw_title(title)

        # Section: Model Overview
        draw_section_header("Model Overview")
        overview_data = [
            ["Model Type", model_info.get("model_type", "")],
            ["Algorithm", model_info.get("algorithm", "")],
            ["Task Type", model_info.get("task_type", "")],
            ["Training Time (s)", f"{model_info.get('training_time', 0):.3f}"],
            ["# Samples", model_info.get("n_samples", "")],
            ["# Features", model_info.get("n_features", "")],
        ]
        draw_table(
            overview_data,
            [1.8 * inch, 3.7 * inch],
            header_grey=True,
            alt_rows=True,
            font_size=10,
        )

        # Section: Feature Names
        draw_section_header("Feature Names")
        feature_names = model_info.get("feature_names", [])
        if feature_names:
            draw_table(
                [[f] for f in feature_names],
                [5.5 * inch],
                header_grey=False,
                alt_rows=True,
                font_size=10,
            )
        else:
            draw_text(
                "No feature names available.", font="Helvetica-Oblique", color=GREY_MED
            )

        # Section: Feature Columns
        draw_section_header("Feature Columns Used")
        feature_columns = model_info.get("feature_columns", [])
        if feature_columns:
            draw_table(
                [[f] for f in feature_columns],
                [5.5 * inch],
                header_grey=False,
                alt_rows=True,
                font_size=10,
            )
        else:
            draw_text(
                "No feature columns available.",
                font="Helvetica-Oblique",
                color=GREY_MED,
            )

        # Section: Model Performance
        draw_section_header("Model Performance")
        perf_data = [
            ["Metric", "Train", "Test"],
            [
                "R2",
                (
                    f"{model_info.get('train_r2', 'N/A'):.4f}"
                    if model_info.get("train_r2") is not None
                    else "N/A"
                ),
                (
                    f"{model_info.get('test_r2', 'N/A'):.4f}"
                    if model_info.get("test_r2") is not None
                    else "N/A"
                ),
            ],
            [
                "RMSE",
                (
                    f"{model_info.get('train_rmse', 'N/A'):.4f}"
                    if model_info.get("train_rmse") is not None
                    else "N/A"
                ),
                (
                    f"{model_info.get('test_rmse', 'N/A'):.4f}"
                    if model_info.get("test_rmse") is not None
                    else "N/A"
                ),
            ],
            [
                "MAE",
                (
                    f"{model_info.get('train_mae', 'N/A'):.4f}"
                    if model_info.get("train_mae") is not None
                    else "N/A"
                ),
                (
                    f"{model_info.get('test_mae', 'N/A'):.4f}"
                    if model_info.get("test_mae") is not None
                    else "N/A"
                ),
            ],
            [
                "CV Mean",
                (
                    f"{model_info.get('cv_mean', 'N/A'):.4f}"
                    if model_info.get("cv_mean") is not None
                    else "N/A"
                ),
                "",
            ],
            [
                "CV Std",
                (
                    f"{model_info.get('cv_std', 'N/A'):.4f}"
                    if model_info.get("cv_std") is not None
                    else "N/A"
                ),
                "",
            ],
        ]
        draw_table(
            perf_data,
            [1.2 * inch, 1.2 * inch, 1.2 * inch],
            header_grey=True,
            alt_rows=True,
            font_size=10,
        )

        # Section: Training Logs - Data Inspection
        training_logs = model_info.get("training_logs", {})
        data_inspection = training_logs.get("data_inspection", {})
        if data_inspection:
            draw_section_header("Training Data Inspection")
            # Basic info
            di_data = [
                ["Rows", data_inspection.get("num_rows", "")],
                ["Columns", data_inspection.get("num_columns", "")],
                ["Column Names", ", ".join(data_inspection.get("columns", []))],
            ]
            draw_table(
                di_data,
                [1.8 * inch, 3.7 * inch],
                header_grey=True,
                alt_rows=True,
                font_size=10,
            )

            # Dtypes
            dtypes = data_inspection.get("dtypes", {})
            if dtypes:
                draw_subheader("Column Data Types")
                dtype_table = [["Column", "Type"]] + [[k, v] for k, v in dtypes.items()]
                draw_table(
                    dtype_table,
                    [2.7 * inch, 2.7 * inch],
                    header_grey=True,
                    alt_rows=True,
                    font_size=10,
                )

            # Missing values
            missing = data_inspection.get("missing_values", {})
            if missing:
                draw_subheader("Missing Values")
                missing_table = [["Column", "Missing"]] + [
                    [k, v] for k, v in missing.items()
                ]
                draw_table(
                    missing_table,
                    [2.7 * inch, 2.7 * inch],
                    header_grey=True,
                    alt_rows=True,
                    font_size=10,
                )

            # Unique values
            unique = data_inspection.get("unique_values", {})
            if unique:
                draw_subheader("Unique Values")
                unique_table = [["Column", "Unique"]] + [
                    [k, v] for k, v in unique.items()
                ]
                draw_table(
                    unique_table,
                    [2.7 * inch, 2.7 * inch],
                    header_grey=True,
                    alt_rows=True,
                    font_size=10,
                )

            # Numeric summary
            numeric_summary = data_inspection.get("numeric_summary", {})
            if numeric_summary:
                draw_subheader("Numeric Summary")
                header = [
                    "Column",
                    "Count",
                    "Mean",
                    "Std",
                    "Min",
                    "25%",
                    "50%",
                    "75%",
                    "Max",
                ]
                rows = []
                for col, stats in numeric_summary.items():
                    row = [
                        col,
                        (
                            f"{stats.get('count', ''):.2f}"
                            if stats.get("count") is not None
                            else ""
                        ),
                        (
                            f"{stats.get('mean', ''):.2f}"
                            if stats.get("mean") is not None
                            else ""
                        ),
                        (
                            f"{stats.get('std', ''):.2f}"
                            if stats.get("std") is not None
                            else ""
                        ),
                        (
                            f"{stats.get('min', ''):.2f}"
                            if stats.get("min") is not None
                            else ""
                        ),
                        (
                            f"{stats.get('25%', ''):.2f}"
                            if stats.get("25%") is not None
                            else ""
                        ),
                        (
                            f"{stats.get('50%', ''):.2f}"
                            if stats.get("50%") is not None
                            else ""
                        ),
                        (
                            f"{stats.get('75%', ''):.2f}"
                            if stats.get("75%") is not None
                            else ""
                        ),
                        (
                            f"{stats.get('max', ''):.2f}"
                            if stats.get("max") is not None
                            else ""
                        ),
                    ]
                    rows.append(row)
                draw_table(
                    [header] + rows,
                    [
                        0.9 * inch,
                        0.6 * inch,
                        0.6 * inch,
                        0.6 * inch,
                        0.6 * inch,
                        0.6 * inch,
                        0.6 * inch,
                        0.6 * inch,
                        0.6 * inch,
                    ],
                    header_grey=True,
                    alt_rows=True,
                    font_size=9,
                )

            # Outliers
            outliers = data_inspection.get("outliers", {})
            if outliers:
                draw_subheader("Outliers Detected")
                outlier_table = [["Column", "Outliers"]] + [
                    [k, v] for k, v in outliers.items()
                ]
                draw_table(
                    outlier_table,
                    [2.7 * inch, 2.7 * inch],
                    header_grey=True,
                    alt_rows=True,
                    font_size=10,
                )

            # Duplicates
            num_duplicates = data_inspection.get("num_duplicates", None)
            if num_duplicates is not None:
                draw_text(
                    f"Number of duplicate rows: {num_duplicates}",
                    font="Helvetica",
                    color=GREY_DARK,
                )

        # Section: Cleaning Report
        cleaning_report = training_logs.get("cleaning_report", {})
        if cleaning_report:
            draw_section_header("Data Cleaning Report")
            # Duplicates removed
            if "duplicates_removed" in cleaning_report:
                draw_text(
                    f"Duplicates removed: {cleaning_report['duplicates_removed']}",
                    font="Helvetica",
                    color=GREY_DARK,
                )
            # Outliers capped
            outliers_capped = cleaning_report.get("outliers_capped", {})
            if outliers_capped:
                draw_subheader("Outliers Capped")
                outcap_table = [["Column", "Capped"]] + [
                    [k, v] for k, v in outliers_capped.items()
                ]
                draw_table(
                    outcap_table,
                    [2.7 * inch, 2.7 * inch],
                    header_grey=True,
                    alt_rows=True,
                    font_size=10,
                )
            # Invalid value corrections
            invalids = cleaning_report.get("invalid_value_corrections", {})
            if invalids:
                draw_subheader("Invalid Value Corrections")
                inv_table = [["Type", "Count"]] + [[k, v] for k, v in invalids.items()]
                draw_table(
                    inv_table,
                    [3.7 * inch, 1.7 * inch],
                    header_grey=True,
                    alt_rows=True,
                    font_size=10,
                )

        c.save()
        buffer.seek(0)

        # Use PyPDF2 to read and write the PDF (for demonstration, just pass through)
        reader = PdfReader(buffer)
        output_buffer = BytesIO()
        writer = PdfWriter()
        for page in reader.pages:
            writer.add_page(page)
        writer.write(output_buffer)
        output_buffer.seek(0)
        return output_buffer
