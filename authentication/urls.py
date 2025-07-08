from django.urls import path
from .views import LoginView, LogoutView, SignupView, PasswordResetRequestView, PasswordResetConfirmView

app_name = "authentication"

urlpatterns = [
    path("login/", LoginView.as_view(), name="login"),
    path("logout/", LogoutView.as_view(), name="logout"),
    path("signup/", SignupView.as_view(), name="signup"),
    path("reset/", PasswordResetRequestView.as_view(), name="password_reset_request"),
    path("reset/<uidb64>/<token>/", PasswordResetConfirmView.as_view(), name="password_reset_confirm"),
]
