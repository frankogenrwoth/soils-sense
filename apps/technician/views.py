from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from django.core.exceptions import PermissionDenied
from authentication.models import Role
from django.contrib import messages

# Keeping the decorator definition for future use, but not applying it
def technician_required(view_func):
    """Decorator to check if the user is a technician"""
    def wrapper(request, *args, **kwargs):
        if not request.user.is_authenticated:
            return redirect('authentication:login')
        if request.user.role != Role.TECHNICIAN:
            raise PermissionDenied("You must be a technician to access this page.")
        return view_func(request, *args, **kwargs)
    return wrapper

def dashboard(request):
    """Main technician dashboard view"""
    return render(request, 'technician/dashboard.html')

def profile(request):
    """Profile management view"""
    if request.method == 'POST':
        # Get form data
        first_name = request.POST.get('first_name')
        last_name = request.POST.get('last_name')
        email = request.POST.get('email')
        phone_number = request.POST.get('phone_number')
        
        # Update user object
        user = request.user
        user.first_name = first_name
        user.last_name = last_name
        user.email = email
        user.phone_number = phone_number
        
        # Handle profile image upload
        if 'profile_image' in request.FILES:
            user.image = request.FILES['profile_image']
        
        user.save()
        messages.success(request, 'Profile updated successfully!')
        return redirect('technician:profile')
    
    return render(request, 'technician/profile.html')

def farm_locations(request):
    """Farm locations management view"""
    return render(request, 'technician/farm_locations.html')

def sensor_config(request):
    """Sensor configuration view"""
    return render(request, 'technician/sensor_config.html')

def analytics(request):
    """Analytics dashboard view"""
    return render(request, 'technician/analytics.html')

def reports(request):
    """Reports management view"""
    return render(request, 'technician/reports.html')

def settings(request):
    """Settings configuration view"""
    return render(request, 'technician/settings.html')
