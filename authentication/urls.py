from django.urls import path
from .views import (
    LoginView,
    LogoutView,
    SignupView,
)
from django.contrib.auth import views as auth_views
from .forms import PasswordResetRequestForm, PasswordResetConfirmForm
from django.urls import reverse_lazy

app_name = "authentication"

urlpatterns = [
    path("login/", LoginView.as_view(), name="login"),
    path("logout/", LogoutView.as_view(), name="logout"),
    path("signup/", SignupView.as_view(), name="signup"),
    path(
        "reset/",
        auth_views.PasswordResetView.as_view(
            template_name="authentication/password_reset_request.html",
            email_template_name="authentication/password_reset_email.html",
            html_email_template_name="authentication/password_reset_email.html",
            form_class=PasswordResetRequestForm,
            success_url=reverse_lazy("authentication:password_reset_done"),
            subject_template_name="authentication/password_reset_subject.txt",  # Optional, for custom subject
        ),
        name="password_reset",
    ),
    path(
        "reset/done/",
        auth_views.PasswordResetDoneView.as_view(
            template_name="authentication/password_reset_done.html"
        ),
        name="password_reset_done",
    ),
    path(
        "reset/<uidb64>/<token>/",
        auth_views.PasswordResetConfirmView.as_view(
            template_name="authentication/password_reset_confirm.html",
            form_class=PasswordResetConfirmForm,
            success_url=reverse_lazy("authentication:password_reset_complete"),
        ),
        name="password_reset_confirm",
    ),
    path(
        "reset/complete/",
        auth_views.PasswordResetCompleteView.as_view(
            template_name="authentication/password_reset_complete.html"
        ),
        name="password_reset_complete",
    ),
]
