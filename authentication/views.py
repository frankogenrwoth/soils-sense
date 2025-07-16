from django.shortcuts import render

from django.views import View
from django.contrib.auth import get_user_model
from django.contrib.auth import login, logout, authenticate
from .forms import LoginForm
from django.urls import reverse
from django.shortcuts import redirect
from django.views.generic.edit import FormView
from .forms import SignupForm
from django.views.generic import FormView
from django.contrib.auth.tokens import default_token_generator
from django.utils.http import urlsafe_base64_encode, urlsafe_base64_decode
from django.utils.encoding import force_bytes, force_str
from django.core.mail import send_mail
from django.conf import settings
from django.contrib.auth import get_user_model
from .forms import PasswordResetForm, PasswordResetConfirmForm
from django.template.loader import render_to_string
from django.contrib import messages
from django.contrib.auth.tokens import PasswordResetTokenGenerator

from .models import Role, User
import random
from datetime import timedelta
from django.utils import timezone


# Create your views here.
class LoginView(View):
    def get(self, request):
        context = {"form": LoginForm()}
        return render(request, "authentication/login.html", context)

    def post(self, request):
        form = LoginForm(request.POST)
        if form.is_valid():
            username = form.cleaned_data["username"]
            password = form.cleaned_data["password"]
            user = authenticate(request, username=username, password=password)

            next_url = request.GET.get("next")
            if next_url and next_url != "/authentication/login/":
                return redirect(next_url)

            if user is not None:
                login(request, user)
                if user.role == Role.ADMINISTRATOR:
                    admin_dashboard_url = reverse("administrator:dashboard")
                    return redirect(admin_dashboard_url)
                if user.role == Role.FARMER:
                    user_dashboard_url = reverse("farmer:dashboard")
                    return redirect(user_dashboard_url)
                if user.role == Role.TECHNICIAN:
                    technician_dashboard_url = reverse("technician:dashboard")
                    return redirect(technician_dashboard_url)
                return render(
                    request,
                    "authentication/login.html",
                    {"message": "Login successful", "form": form},
                )
            else:
                return render(
                    request,
                    "authentication/login.html",
                    {"form": form, "message": "Invalid credentials"},
                )
        else:
            # Add error handling for form validation
            return render(
                request,
                "authentication/login.html",
                {"form": form, "message": "Please check your input and try again."},
            )


class LogoutView(View):
    def get(self, request):
        logout(request)
        return redirect("home")


class SignupView(FormView):
    template_name = "authentication/signup.html"
    form_class = SignupForm
    success_url = "/authentication/login/"

    def form_valid(self, form):
        form.save()
        messages.success(self.request, "Signup successful! Please log in.")
        return super().form_valid(form)


class PasswordResetRequestView(FormView):
    template_name = "authentication/password_reset_request.html"
    form_class = PasswordResetForm
    success_url = "/authentication/reset/confirm/"

    def form_valid(self, form):
        email = form.cleaned_data["email"]
        User = get_user_model()
        try:
            user = User.objects.get(email=email)
            # Generate a 6-digit code
            code = f"{random.randint(100000, 999999)}"
            user.reset_code = code
            user.reset_code_expiry = timezone.now() + timedelta(minutes=15)
            user.save()
            subject = "Your Password Reset Code"
            message = render_to_string(
                "authentication/password_reset_email.txt",
                {
                    "code": code,
                    "user": user,
                },
            )
            send_mail(subject, message, settings.DEFAULT_FROM_EMAIL, [email])
            self.request.session["reset_email"] = email
        except User.DoesNotExist:
            pass  # Do not reveal if email exists
        return super().form_valid(form)


class PasswordResetConfirmView(FormView):
    template_name = "authentication/password_reset_confirm.html"
    form_class = PasswordResetConfirmForm
    success_url = "/authentication/login/"

    def get(self, request, *args, **kwargs):
        return super().get(request, *args, **kwargs)

    def post(self, request, *args, **kwargs):
        form = self.form_class(request.POST)
        if form.is_valid():
            email = request.session.get("reset_email")
            code = form.cleaned_data["code"]
            password = form.cleaned_data["password"]
            User = get_user_model()
            try:
                user = User.objects.get(email=email, reset_code=code)
                if user.reset_code_expiry and user.reset_code_expiry < timezone.now():
                    form.add_error("code", "Reset code has expired.")
                    return self.form_invalid(form)
                user.set_password(password)
                user.reset_code = None
                user.reset_code_expiry = None
                user.save()
                return super().form_valid(form)
            except User.DoesNotExist:
                form.add_error("code", "Invalid reset code or email.")
                return self.form_invalid(form)
        return self.form_invalid(form)
