from django.urls import path
from .views import LoginView, LogoutView

app_name = "authentication"

url_patterns = [
    path("login/", LoginView.as_view(), name="login"),
    path("logout/", LogoutView.as_view(), name="logout"),
]
