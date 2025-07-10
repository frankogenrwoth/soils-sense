from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required, user_passes_test
from django.core.exceptions import PermissionDenied
from authentication.models import Role
from django.contrib import messages
from apps.farmer.models import Farm, SoilMoistureReading
from django.shortcuts import get_object_or_404, redirect, render
from django.core.exceptions import PermissionDenied
from django.db.models import Avg
from django.utils.timezone import now, timedelta



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
        # ... any other context ...
    selected_farm_id = request.GET.get('farm')
    farms = Farm.objects.all()

    # Filter by farm if selected
    readings = SoilMoistureReading.objects.all()
    if selected_farm_id:
        readings = readings.filter(farm__id=selected_farm_id)

    # Prepare daily average moisture over last 7 days
    days = []
    moisture_per_day = []
    today = now().date()
    for i in range(6, -1, -1):
        day = today - timedelta(days=i)
        daily_readings = readings.filter(timestamp__date=day)
        avg = daily_readings.aggregate(avg=Avg('soil_moisture_percent'))['avg']
        days.append(day.strftime('%b %d'))
        moisture_per_day.append(round(avg, 2) if avg else 0)

    # Recent critical readings
    recent_critical = readings.filter(
        status__in=['Critical Low', 'Critical High']
    ).order_by('-timestamp')[:5]

    context = {
        'farms': farms,
        'selected_farm_id': selected_farm_id,
        'days': days,
        'moisture_per_day': moisture_per_day,
        'recent_critical_readings': recent_critical,
    }
    return render(request, 'technician/dashboard.html', context)

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

def is_technician(user):
    return user.is_authenticated and user.role == 'technician'

@login_required
@user_passes_test(is_technician)
def technician_soil_readings(request):
    farms = Farm.objects.all()
    readings = SoilMoistureReading.objects.select_related('farm')

    # Filters
    farm_id = request.GET.get('farm')
    date_from = request.GET.get('date_from')
    date_to = request.GET.get('date_to')

    if farm_id:
        readings = readings.filter(farm__id=farm_id)
    if date_from:
        readings = readings.filter(timestamp__date__gte=date_from)
    if date_to:
        readings = readings.filter(timestamp__date__lte=date_to)

    context = {
        'farms': farms,
        'readings': readings,
        'selected_farm_id': int(farm_id) if farm_id else None,
        'date_from': date_from,
        'date_to': date_to,
    }
    return render(request, 'technician/soil_readings.html', context)
