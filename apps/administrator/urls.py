from django.urls import path
from apps.administrator.views import UserManagementView

urlpatterns = [
    path("users/", UserManagementView.as_view(), name="users"),
]
