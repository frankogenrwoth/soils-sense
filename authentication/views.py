from django.shortcuts import render

from django.views import View
from django.contrib.auth import get_user_model

from django.contrib.auth.forms import AuthenticationForm
# from django.contrib.auth import login, logout, authenticate
from django.urls import reverse
from django.shortcuts import redirect

from .models import Role, User


# Create your views here.
class LoginView(View):
    def get(self, request):
        context = {
            'form': AuthenticationForm()
        }
        
        return render(request, 'authentication/login.html')
    
    def post(self, request):
        # Handle login logic here
        form = AuthenticationForm(request, data=request.POST)
        if form.is_valid():
            user = form.get_user()
            # Log the user in, set session, etc.
            
            # For example: login(request, user)
            
            if user.role == Role.ADMIN:
                # Redirect to admin dashboard
                admin_dashboard_url = reverse("admin:dashboard")
                return redirect(admin_dashboard_url)
            
            if user.role == Role.FARMER:
                # Redirect to user dashboard
                user_dashboard_url = reverse("farmer:dashboard")
                return redirect(user_dashboard_url)
            
            if user.role == Role.TECHNICIAN:
                # Redirect to technician dashboard
                technician_dashboard_url = reverse("technician:dashboard")
                return redirect(technician_dashboard_url)
            
            
            # redirect the user based on his or her role
            
            return render(request, 'authentication/login.html', {'message': 'Login successful'})
        return render(request, 'authentication/login.html', {'message': 'Login successful'})

class LogoutView(View):
    pass