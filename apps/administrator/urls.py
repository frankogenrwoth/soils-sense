from django.urls import path
from django.contrib.auth.decorators import login_required
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
    path("", login_required(DashboardView.as_view()), name="dashboard"),
    path("users/", login_required(UserManagementView.as_view()), name="users"),
    path("users/<int:pk>/", login_required(UserDetailView.as_view()), name="user_detail"),
    path("users/<int:pk>/update/", login_required(UserUpdateView.as_view()), name="user_update"),
    path("users/<int:pk>/delete/", login_required(UserDeleteView.as_view()), name="user_delete"),
    path("data/", login_required(DataManagementView.as_view()), name="data"),
    path("reports/", login_required(ReportManagementView.as_view()), name="reports"),
    path("ml-models/", login_required(MLModelManagementView.as_view()), name="ml"),
    path("notifications/", login_required(NotificationView.as_view()), name="notifications"),
    path("sensors/", login_required(SensorView.as_view()), name="sensors"),
]
