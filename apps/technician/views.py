from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required, user_passes_test
from django.core.exceptions import PermissionDenied
from authentication.models import Role
from django.contrib import messages
from apps.farmer.models import Farm, SoilMoistureReading
from django.shortcuts import get_object_or_404, redirect, render
from django.core.exceptions import PermissionDenied
from django.db.models import Avg, Count, Max, Min
from django.utils.timezone import now, timedelta
from .forms import FarmEditForm, SoilReadingFilterForm
from datetime import datetime, timedelta



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
    from django.db.models import Count, Q
    from datetime import datetime, timedelta
    
    # Get date range for recent data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=7)
    
    # Basic statistics
    farm_count = Farm.objects.count()
    total_readings = SoilMoistureReading.objects.count()
    
    # Recent readings (last 7 days)
    recent_readings = SoilMoistureReading.objects.filter(
        timestamp__gte=start_date
    ).count()
    
    # Critical readings count
    critical_readings = SoilMoistureReading.objects.filter(
        status__in=['Critical Low', 'Critical High']
    ).count()
    
    # Get farms with most readings
    farms_with_readings = Farm.objects.annotate(
        reading_count=Count('moisture_readings')
    ).order_by('-reading_count')[:5]
    
    # Recent critical readings for alerts
    recent_critical = SoilMoistureReading.objects.filter(
        status__in=['Critical Low', 'Critical High']
    ).select_related('farm').order_by('-timestamp')[:5]
    
    # Moisture distribution for charts
    moisture_distribution = {
        'Normal': SoilMoistureReading.objects.filter(status='Normal').count(),
        'Dry': SoilMoistureReading.objects.filter(status='Dry').count(),
        'Wet': SoilMoistureReading.objects.filter(status='Wet').count(),
        'Critical Low': SoilMoistureReading.objects.filter(status='Critical Low').count(),
        'Critical High': SoilMoistureReading.objects.filter(status='Critical High').count(),
    }
    
    # Average moisture by farm
    farm_averages = []
    for farm in Farm.objects.all()[:5]:
        avg_moisture = farm.moisture_readings.aggregate(
            avg=Avg('soil_moisture_percent')
        )['avg'] or 0
        farm_averages.append({
            'farm': farm,
            'avg_moisture': round(avg_moisture, 2),
            'reading_count': farm.moisture_readings.count()
        })
    
    # Sensor thresholds summary
    from .models import SensorThreshold
    threshold_count = SensorThreshold.objects.count()
    warning_thresholds = SensorThreshold.objects.filter(status='Warning').count()
    
    # Recent reports
    from .models import Report
    recent_reports = Report.objects.select_related('farm').order_by('-created_at')[:3]
    
    # Daily moisture trend (last 7 days)
    days = []
    moisture_per_day = []
    for i in range(6, -1, -1):
        day = end_date - timedelta(days=i)
        daily_readings = SoilMoistureReading.objects.filter(
            timestamp__date=day
        )
        avg = daily_readings.aggregate(avg=Avg('soil_moisture_percent'))['avg']
        days.append(day.strftime('%b %d'))
        moisture_per_day.append(round(avg, 2) if avg else 0)
    
    # Farm filter for dashboard
    selected_farm_id = request.GET.get('farm')
    farms = Farm.objects.all()
    
    context = {
        'farm_count': farm_count,
        'total_readings': total_readings,
        'recent_readings': recent_readings,
        'critical_readings': critical_readings,
        'farms_with_readings': farms_with_readings,
        'recent_critical_readings': recent_critical,
        'moisture_distribution': moisture_distribution,
        'farm_averages': farm_averages,
        'threshold_count': threshold_count,
        'warning_thresholds': warning_thresholds,
        'recent_reports': recent_reports,
        'days': days,
        'moisture_per_day': moisture_per_day,
        'farms': farms,
        'selected_farm_id': selected_farm_id,
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
    
    # Get recent soil readings for this farm
    recent_readings = SoilMoistureReading.objects.filter(farm=farm).order_by('-timestamp')[:10]
    
    # Get recent alerts (placeholder - you can implement actual alerts later)
    recent_alerts = []
    
    context = {
        'farm': farm,
        'recent_readings': recent_readings,
        'recent_alerts': recent_alerts,
    }
    return render(request, 'technician/farm_detail.html', context)

def sensor_config(request):
    """Sensor configuration view"""
    from .models import SensorThreshold
    from .forms import SensorThresholdForm
    
    if request.method == 'POST':
        form = SensorThresholdForm(request.POST)
        if form.is_valid():
            threshold = form.save()
            messages.success(request, 'Sensor threshold configured successfully!')
            return redirect('technician:sensor_config')
    else:
        form = SensorThresholdForm()
    
    # Get existing thresholds
    thresholds = SensorThreshold.objects.select_related('farm').all()
    
    context = {
        'form': form,
        'thresholds': thresholds,
        'farms': Farm.objects.all(),
    }
    return render(request, 'technician/sensor_config.html', context)

def analytics(request):
    """Analytics dashboard view"""
    
    # Get date range (last 30 days)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    
    # Analytics data
    total_farms = Farm.objects.count()
    total_readings = SoilMoistureReading.objects.count()
    avg_moisture = SoilMoistureReading.objects.aggregate(avg=Avg('soil_moisture_percent'))['avg'] or 0
    
    # Recent readings trend
    recent_readings = SoilMoistureReading.objects.filter(
        timestamp__gte=start_date
    ).order_by('timestamp')
    
    # Critical readings count
    critical_readings = SoilMoistureReading.objects.filter(
        status__in=['Critical Low', 'Critical High']
    ).count()
    
    # Farm with most readings
    farm_with_most_readings = Farm.objects.annotate(
        reading_count=Count('moisture_readings')
    ).order_by('-reading_count').first()
    
    # Moisture distribution
    moisture_distribution = {
        'Normal': SoilMoistureReading.objects.filter(status='Normal').count(),
        'Dry': SoilMoistureReading.objects.filter(status='Dry').count(),
        'Wet': SoilMoistureReading.objects.filter(status='Wet').count(),
        'Critical Low': SoilMoistureReading.objects.filter(status='Critical Low').count(),
        'Critical High': SoilMoistureReading.objects.filter(status='Critical High').count(),
    }
    
    context = {
        'total_farms': total_farms,
        'total_readings': total_readings,
        'avg_moisture': round(avg_moisture, 2),
        'critical_readings': critical_readings,
        'farm_with_most_readings': farm_with_most_readings,
        'moisture_distribution': moisture_distribution,
        'recent_readings': recent_readings[:50],  # Last 50 readings for chart
    }
    return render(request, 'technician/analytics.html', context)

def reports(request):
    """Reports management view"""
    from .models import Report
    from .forms import ReportForm
    
    if request.method == 'POST':
        form = ReportForm(request.POST, request.FILES)
        if form.is_valid():
            report = form.save(commit=False)
            report.generated_by = request.user.get_full_name() or request.user.username
            report.save()
            messages.success(request, 'Report created successfully!')
            return redirect('technician:reports')
    else:
        form = ReportForm()
    
    # Get existing reports
    reports = Report.objects.select_related('farm').all().order_by('-created_at')
    
    context = {
        'form': form,
        'reports': reports,
        'farms': Farm.objects.all(),
    }
    return render(request, 'technician/reports.html', context)

def settings(request):
    """Settings view"""
    from django.contrib.auth.forms import PasswordChangeForm
    from django.contrib.auth import update_session_auth_hash
    
    user = request.user
    password_form = PasswordChangeForm(user)
    
    if request.method == 'POST':
        if 'update_profile' in request.POST:
            # Update user profile
            user.first_name = request.POST.get('first_name', '')
            user.last_name = request.POST.get('last_name', '')
            user.email = request.POST.get('email', '')
            user.save()
            messages.success(request, 'Profile updated successfully!')
            return redirect('technician:settings')
        
        elif 'change_password' in request.POST:
            password_form = PasswordChangeForm(user, request.POST)
            if password_form.is_valid():
                user = password_form.save()
                update_session_auth_hash(request, user)
                messages.success(request, 'Password changed successfully!')
                return redirect('technician:settings')
            else:
                messages.error(request, 'Please correct the errors below.')
    
    context = {
        'user': user,
        'password_form': password_form,
    }
    return render(request, 'technician/settings.html', context)

def edit_farm(request, pk):
    farm = get_object_or_404(Farm, pk=pk)
    
    if request.method == 'POST':
        form = FarmEditForm(request.POST, instance=farm)
        if form.is_valid():
            form.save()
            messages.success(request, 'Farm updated successfully!')
            return redirect('technician:farm_detail', pk=farm.pk)
        else:
            messages.error(request, 'Please correct the errors below.')
    else:
        form = FarmEditForm(instance=farm)
    
    return render(request, 'technician/edit_farm.html', {'farm': farm, 'form': form})

def is_technician(user):
    return user.is_authenticated and user.role == 'technician'

@login_required
@user_passes_test(is_technician)
def technician_soil_readings(request):
    from django.core.paginator import Paginator
    
    form = SoilReadingFilterForm(request.GET)
    readings = SoilMoistureReading.objects.select_related('farm').order_by('-timestamp')

    if form.is_valid():
        farm = form.cleaned_data.get('farm')
        date_from = form.cleaned_data.get('date_from')
        date_to = form.cleaned_data.get('date_to')
        status = form.cleaned_data.get('status')

        if farm:
            readings = readings.filter(farm=farm)
        if date_from:
            readings = readings.filter(timestamp__date__gte=date_from)
        if date_to:
            readings = readings.filter(timestamp__date__lte=date_to)
        if status:
            readings = readings.filter(status=status)

    # Pagination
    paginator = Paginator(readings, 25)  # Show 25 readings per page
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)

    context = {
        'form': form,
        'readings': page_obj,
    }
    return render(request, 'technician/soil_readings.html', context)

def delete_threshold(request, pk):
    """Delete sensor threshold"""
    from .models import SensorThreshold
    
    threshold = get_object_or_404(SensorThreshold, pk=pk)
    if request.method == 'POST':
        threshold.delete()
        messages.success(request, 'Sensor threshold deleted successfully!')
        return redirect('technician:sensor_config')
    
    return render(request, 'technician/confirm_delete_threshold.html', {'threshold': threshold})

def edit_threshold(request, pk):
    """Edit sensor threshold"""
    from .models import SensorThreshold
    from .forms import SensorThresholdForm
    
    threshold = get_object_or_404(SensorThreshold, pk=pk)
    
    if request.method == 'POST':
        form = SensorThresholdForm(request.POST, instance=threshold)
        if form.is_valid():
            form.save()
            messages.success(request, 'Sensor threshold updated successfully!')
            return redirect('technician:sensor_config')
    else:
        form = SensorThresholdForm(instance=threshold)
    
    context = {
        'form': form,
        'threshold': threshold,
        'farms': Farm.objects.all(),
    }
    return render(request, 'technician/edit_threshold.html', context)

def delete_report(request, pk):
    """Delete report"""
    from .models import Report
    
    report = get_object_or_404(Report, pk=pk)
    if request.method == 'POST':
        report.delete()
        messages.success(request, 'Report deleted successfully!')
        return redirect('technician:reports')
    
    return render(request, 'technician/confirm_delete_report.html', {'report': report})

def edit_report(request, pk):
    """Edit report"""
    from .models import Report
    from .forms import ReportForm
    
    report = get_object_or_404(Report, pk=pk)
    
    if request.method == 'POST':
        form = ReportForm(request.POST, request.FILES, instance=report)
        if form.is_valid():
            form.save()
            messages.success(request, 'Report updated successfully!')
            return redirect('technician:reports')
    else:
        form = ReportForm(instance=report)
    
    context = {
        'form': form,
        'report': report,
        'farms': Farm.objects.all(),
    }
    return render(request, 'technician/edit_report.html', context)
