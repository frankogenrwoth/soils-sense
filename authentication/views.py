from django.shortcuts import render

from django.views import View
from django.contrib.auth import get_user_model
from django.contrib.auth import login, logout, authenticate
from .forms import LoginForm
from django.urls import reverse
from django.shortcuts import redirect
from django.views.generic.edit import FormView
from .forms import SignupForm

from .models import Role, User


# Create your views here.
class LoginView(View):
    def get(self, request):
        context = {
            'form': LoginForm()
        }
        return render(request, 'authentication/login.html', context)
    
    def post(self, request):
        form = LoginForm(request.POST)
        if form.is_valid():
            username = form.cleaned_data['username']
            password = form.cleaned_data['password']
            user = authenticate(request, username=username, password=password)
            if user is not None:
                login(request, user)
                if user.role == Role.ADMIN:
                    admin_dashboard_url = reverse("admin:dashboard")
                    return redirect(admin_dashboard_url)
                if user.role == Role.FARMER:
                    user_dashboard_url = reverse("farmer:dashboard")
                    return redirect(user_dashboard_url)
                if user.role == Role.TECHNICIAN:
                    technician_dashboard_url = reverse("technician:dashboard")
                    return redirect(technician_dashboard_url)
                return render(request, 'authentication/login.html', {'message': 'Login successful', 'form': form})
            else:
                return render(request, 'authentication/login.html', {'form': form, 'message': 'Invalid credentials'})
        return render(request, 'authentication/login.html', {'form': form})

class LogoutView(View):
    def get(self, request):
        logout(request)
        return redirect('authentication:login')

class SignupView(FormView):
    template_name = 'authentication/signup.html'
    form_class = SignupForm
    success_url = '/authentication/login/'

    def form_valid(self, form):
        form.save()
        return super().form_valid(form)