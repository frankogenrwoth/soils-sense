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
    MLModelDetailView,
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
    path(
        "reports/", admin_role_required(ReportManagementView.as_view()), name="reports"
    ),
    path("ml-models/", admin_role_required(MLModelManagementView.as_view()), name="ml"),
    path(
        "ml-models/<str:model_name>/",
        admin_role_required(MLModelDetailView.as_view()),
        name="ml_model_detail",
    ),
    path(
        "notifications/",
        admin_role_required(NotificationView.as_view()),
        name="notifications",
    ),
    path("sensors/", admin_role_required(SensorView.as_view()), name="sensors"),
]
