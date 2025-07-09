from django.urls import path
from apps.administrator.views import UserManagementView

app_name = "administrator"

urlpatterns = [
    path("users/", UserManagementView.as_view(), name="users"),
]
