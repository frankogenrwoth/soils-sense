from django.urls import path
from apps.administrator.views import (
    UserManagementView,
    UserDetailView,
    UserUpdateView,
    UserDeleteView,
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
    path("users/<int:pk>/", UserDetailView.as_view(), name="user_detail"),
    path("users/<int:pk>/update/", UserUpdateView.as_view(), name="user_update"),
    path("users/<int:pk>/delete/", UserDeleteView.as_view(), name="user_delete"),
    path("data/", DataManagementView.as_view(), name="data"),
    path("reports/", ReportManagementView.as_view(), name="reports"),
    path("ml-models/", MLModelManagementView.as_view(), name="ml"),
    path("notifications/", NotificationView.as_view(), name="notifications"),
    path("sensors/", SensorView.as_view(), name="sensors"),
]
