from django.urls import path
from apps.administrator.views import (
    UserManagementView,
    DataManagementView,
    ReportManagementView,
    MLModelManagementView,
    NotificationView,
    SensorView,
    DashboardView,
)

app_name = "administrator"

urlpatterns = [
    path("", DashboardView.as_view(), name="dashboard"),
    path("users/", UserManagementView.as_view(), name="users"),
    path("data/", DataManagementView.as_view(), name="data"),
    path("reports/", ReportManagementView.as_view(), name="reports"),
    path("ml-models/", MLModelManagementView.as_view(), name="ml"),
    path("notifications/", NotificationView.as_view(), name="notifications"),
    path("sensors/", SensorView.as_view(), name="sensors"),
]
