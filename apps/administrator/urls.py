from django.urls import path
from django.contrib.auth.decorators import login_required
from apps.administrator.views import (
    UserManagementView,
    UserDetailView,
    UserUpdateView,
    UserDeleteView,
    DataManagementView,
    MLModelManagementView,
    NotificationView,
    SensorView,
    DashboardView,
    MLModelDetailView,
    UploadDatasetView,
    PrintReportView,
    ServerLogView,
)
from apps.administrator.data_views import (
    FarmDetailView,
    FarmDeleteView,
    SoilMoistureReadingDetailView,
    SoilMoistureReadingDeleteView,
    WeatherDataDetailView,
    WeatherDataDeleteView,
    IrrigationEventDetailView,
    IrrigationEventDeleteView,
    PredictionResultDetailView,
    PredictionResultDeleteView,
    AlertDetailView,
    AlertDeleteView,
    NotificationDetailView,
    NotificationDeleteView,
)
from apps.administrator.utils import admin_role_required

app_name = "administrator"

urlpatterns = [
    path("", admin_role_required(DashboardView.as_view()), name="dashboard"),
    path("users/", admin_role_required(UserManagementView.as_view()), name="users"),
    path(
        "users/<int:pk>/",
        admin_role_required(UserDetailView.as_view()),
        name="user_detail",
    ),
    path(
        "users/<int:pk>/update/",
        admin_role_required(UserUpdateView.as_view()),
        name="user_update",
    ),
    path(
        "users/<int:pk>/delete/",
        admin_role_required(UserDeleteView.as_view()),
        name="user_delete",
    ),
    path("data/", admin_role_required(DataManagementView.as_view()), name="data"),
    # Farm URLs
    path(
        "data/farms/<int:pk>/",
        admin_role_required(FarmDetailView.as_view()),
        name="farm_detail",
    ),
    path(
        "data/farms/<int:pk>/delete/",
        admin_role_required(FarmDeleteView.as_view()),
        name="farm_delete",
    ),
    # Soil Moisture Reading URLs
    path(
        "data/soil-moisture/<int:pk>/",
        admin_role_required(SoilMoistureReadingDetailView.as_view()),
        name="soil_moisture_detail",
    ),
    path(
        "data/soil-moisture/<int:pk>/delete/",
        admin_role_required(SoilMoistureReadingDeleteView.as_view()),
        name="soil_moisture_delete",
    ),
    # Weather Data URLs
    path(
        "data/weather/<int:pk>/",
        admin_role_required(WeatherDataDetailView.as_view()),
        name="weather_detail",
    ),
    path(
        "data/weather/<int:pk>/delete/",
        admin_role_required(WeatherDataDeleteView.as_view()),
        name="weather_delete",
    ),
    # Irrigation Event URLs
    path(
        "data/irrigation/<int:pk>/",
        admin_role_required(IrrigationEventDetailView.as_view()),
        name="irrigation_detail",
    ),
    path(
        "data/irrigation/<int:pk>/delete/",
        admin_role_required(IrrigationEventDeleteView.as_view()),
        name="irrigation_delete",
    ),
    # Prediction Result URLs
    path(
        "data/predictions/<int:pk>/",
        admin_role_required(PredictionResultDetailView.as_view()),
        name="prediction_detail",
    ),
    path(
        "data/predictions/<int:pk>/delete/",
        admin_role_required(PredictionResultDeleteView.as_view()),
        name="prediction_delete",
    ),
    # Alert URLs
    path(
        "data/alerts/<int:pk>/",
        admin_role_required(AlertDetailView.as_view()),
        name="alert_detail",
    ),
    path(
        "data/alerts/<int:pk>/delete/",
        admin_role_required(AlertDeleteView.as_view()),
        name="alert_delete",
    ),
    # Notification URLs
    path(
        "data/notifications/<int:pk>/",
        admin_role_required(NotificationDetailView.as_view()),
        name="notification_detail",
    ),
    path(
        "data/notifications/<int:pk>/delete/",
        admin_role_required(NotificationDeleteView.as_view()),
        name="notification_delete",
    ),
    path(
        "server-logs/",
        admin_role_required(ServerLogView.as_view()),
        name="server_logs",
    ),
    path("ml-models/", admin_role_required(MLModelManagementView.as_view()), name="ml"),
    path(
        "ml-models/<str:model_name>/",
        admin_role_required(MLModelDetailView.as_view()),
        name="ml_model_detail",
    ),
    path(
        "ml-models/upload/dataset/",
        admin_role_required(UploadDatasetView.as_view()),
        name="upload_dataset",
    ),
    path(
        "notifications/",
        admin_role_required(NotificationView.as_view()),
        name="notifications",
    ),
    path("sensors/", admin_role_required(SensorView.as_view()), name="sensors"),
    path(
        "ml-models/<str:model_name>/print-report/",
        admin_role_required(PrintReportView.as_view()),
        name="print_report",
    ),
]
