from django.shortcuts import render
from django.views import View
from ml import MLEngine
import logging

logger = logging.getLogger(__name__)


class DashboardView(View):
    template_name = "administrator/dashboard.html"

    def get(self, request):
        context = {}
        return render(request, self.template_name, context=context)


class UserManagementView(View):
    template_name = "administrator/user_management.html"

    def get(self, request):
        context = {}
        return render(request, self.template_name, context=context)


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

    def get(self, request):
        ml_engine = MLEngine()

        available_models = ml_engine.get_available_models()

        print(available_models)

        model_info = ml_engine.get_model_info(
            "soil_moisture_predictor_gradient_boosting"
        )

        print(model_info)

        context = {
            "available_models": available_models,
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
