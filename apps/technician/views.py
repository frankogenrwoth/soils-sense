from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.core.exceptions import PermissionDenied
from authentication.models import Role
from django.contrib import messages
from apps.farmer.models import Farm
from django.shortcuts import get_object_or_404, redirect, render
from django.core.exceptions import PermissionDenied


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
    farm_count = Farm.objects.count()
    return render(request, 'technician/dashboard.html', {
        'farm_count': farm_count,
        # ... any other context ...
    })

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
    farms = Farm.objects.all()  # Or filter as needed
    return render(request, 'technician/farm_locations.html', {
        'farms': farms
    })

def delete_farm(request, pk):
    """
    View to delete a farm.
    Only allow POST requests for deletion.
    """
    farm = get_object_or_404(Farm, pk=pk)
    if request.method == 'POST':
        farm.delete()
        messages.success(request, 'Farm deleted successfully!')
        return redirect('technician:farm_locations')
    return render(request, 'technician/confirm_delete_farm.html', {'farm': farm})

def farm_detail(request, pk):
    farm = get_object_or_404(Farm, pk=pk)
    return render(request, 'technician/farm_detail.html', {'farm': farm})

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

def edit_farm(request, pk):
    farm = get_object_or_404(Farm, pk=pk)
    # Add your edit logic here
    return render(request, 'technician/edit_farm.html', {'farm': farm})
