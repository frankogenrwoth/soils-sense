from django.urls import path
from apps.administrator.views import (
    UserManagementView,
    DataManagementView,
    ReportManagementView,
    MLModelManagementView,
)

app_name = "administrator"

urlpatterns = [
    path("users/", UserManagementView.as_view(), name="users"),
    path("data/", DataManagementView.as_view(), name="data"),
    path("reports/", ReportManagementView.as_view(), name="reports"),
    path("ml-models/", MLModelManagementView.as_view(), name="ml"),
]
