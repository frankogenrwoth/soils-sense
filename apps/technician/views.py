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
from .forms import FarmEditForm, SoilReadingFilterForm, TechnicianProfileForm
from datetime import datetime, timedelta
import csv
from django.http import HttpResponse
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from django.http import Http404
from ml import MLEngine
import datetime



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
    user = request.user
    image_url = user.image.url if user.image else ''
    has_custom_image = bool(user.image and not image_url.endswith('default.webp'))
    return render(request, 'technician/profile.html', {'user': user, 'has_custom_image': has_custom_image})

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
    from apps.farmer.models import Farm
    if request.method == 'POST':
        if request.POST.get('generate_prediction'):
            # Handle prediction form
            farm_id = request.POST.get('farm_id')
            if not farm_id:
                messages.error(request, "Please select a farm.")
                form = ReportForm()
                reports = Report.objects.select_related('farm').all().order_by('-created_at')
                user_farms = Farm.objects.filter(user=request.user)
                context = {
                    'form': form,
                    'reports': reports,
                    'farms': user_farms,
                }
                return render(request, 'technician/reports.html', context)
            try:
                farm = Farm.objects.get(id=farm_id, user=request.user)
            except Farm.DoesNotExist:
                messages.error(request, "Selected farm not found.")
                form = ReportForm()
                reports = Report.objects.select_related('farm').all().order_by('-created_at')
                user_farms = Farm.objects.filter(user=request.user)
                context = {
                    'form': form,
                    'reports': reports,
                    'farms': user_farms,
                }
                return render(request, 'technician/reports.html', context)
            location = request.POST.get('location')
            temperature_celsius = request.POST.get('temperature_celsius')
            humidity_percent = request.POST.get('humidity_percent')
            battery_voltage = request.POST.get('battery_voltage')
            status = request.POST.get('status')
            timestamp = request.POST.get('timestamp')
            if timestamp:
                timestamp = datetime.datetime.strptime(timestamp, '%Y-%m-%dT%H:%M').strftime('%Y-%m-%d %H:%M:%S')
            else:
                timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            # Use dummy values for sensor_id and irrigation_action for soil moisture prediction
            ml = MLEngine()
            soil_pred = ml.predict_soil_moisture(
                sensor_id='manual',
                location=location,
                temperature_celsius=float(temperature_celsius),
                humidity_percent=float(humidity_percent),
                battery_voltage=float(battery_voltage),
                status=status,
                irrigation_action='None',
                timestamp=timestamp,
            )
            predicted_soil_moisture = soil_pred.get('predicted_value')
            # Now use this value to get irrigation recommendation
            ir_pred = ml.recommend_irrigation(
                soil_moisture_percent=predicted_soil_moisture,
                temperature_celsius=float(temperature_celsius),
                humidity_percent=float(humidity_percent),
                battery_voltage=float(battery_voltage),
                status=status,
                timestamp=timestamp,
            )
            irrigation_action = ir_pred.get('predicted_value')
            # Compose description
            description = f"Location: {location}\nTemperature: {temperature_celsius}°C\nHumidity: {humidity_percent}%\nBattery Voltage: {battery_voltage}V\nStatus: {status}\nTimestamp: {timestamp}\n\nPredicted Soil Moisture: {predicted_soil_moisture}%\nRecommended Irrigation Action: {irrigation_action}"
            report = Report.objects.create(
                farm=farm,
                report_type='prediction',
                title=f"Soil Moisture Prediction ({location})",
                description=description,
                generated_by=request.user.get_full_name() or request.user.username
            )
            messages.success(request, 'Prediction report generated and saved!')
            return redirect('technician:reports')
        else:
            form = ReportForm(request.POST, request.FILES)
            if form.is_valid():
                report = form.save(commit=False)
                report.generated_by = request.user.get_full_name() or request.user.username
                report.save()
                messages.success(request, 'Report created successfully!')
                return redirect('technician:reports')
    else:
        form = ReportForm()
    # Only show farms belonging to the current user (technician)
    user_farms = Farm.objects.filter(user=request.user)
    reports = Report.objects.select_related('farm').all().order_by('-created_at')
    context = {
        'form': form,
        'reports': reports,
        'farms': user_farms,
    }
    return render(request, 'technician/reports.html', context)

def export_reports(request):
    from .models import Report
    response = HttpResponse(content_type='text/csv')
    response['Content-Disposition'] = 'attachment; filename="reports.csv"'

    writer = csv.writer(response)
    writer.writerow(['Title', 'Farm', 'Type', 'Generated By', 'Created', 'Description'])
    for report in Report.objects.select_related('farm').all().order_by('-created_at'):
        writer.writerow([
            report.title,
            report.farm.farm_name if report.farm else '',
            report.report_type,
            report.generated_by,
            report.created_at.strftime('%Y-%m-%d %H:%M'),
            report.description or ''
        ])
    return response

def settings(request):
    """Settings view"""
    from django.contrib.auth.forms import PasswordChangeForm
    from django.contrib.auth import update_session_auth_hash

    user = request.user
    password_form = PasswordChangeForm(user)
    profile_form = TechnicianProfileForm(instance=user)

    image_url = user.image.url if user.image else ''
    has_custom_image = bool(user.image and not image_url.endswith('default.webp'))

    if request.method == 'POST':
        if 'update_profile' in request.POST:
            profile_form = TechnicianProfileForm(request.POST, request.FILES, instance=user)
            if profile_form.is_valid():
                profile_form.save()
                messages.success(request, 'Profile updated successfully!')
                return redirect('technician:settings')
            else:
                messages.error(request, 'Please correct the errors below.')
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
        'profile_form': profile_form,
        'has_custom_image': has_custom_image,
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

def download_prediction_pdf(request, pk):
    from .models import Report
    report = get_object_or_404(Report, pk=pk)
    if report.report_type != 'prediction':
        raise Http404('Not a prediction report')

    response = HttpResponse(content_type='application/pdf')
    response['Content-Disposition'] = f'attachment; filename="prediction_report_{report.pk}.pdf"'

    p = canvas.Canvas(response, pagesize=letter)
    width, height = letter
    y = height - 50

    p.setFont('Helvetica-Bold', 16)
    p.drawString(50, y, f'Prediction Report: {report.title}')
    y -= 30
    p.setFont('Helvetica', 12)
    p.drawString(50, y, f'Farm: {report.farm.farm_name if report.farm else "N/A"}')
    y -= 20
    p.drawString(50, y, f'Generated By: {report.generated_by}')
    y -= 20
    p.drawString(50, y, f'Created: {report.created_at.strftime("%Y-%m-%d %H:%M")}')
    y -= 30
    p.setFont('Helvetica-Bold', 12)
    p.drawString(50, y, 'Description:')
    y -= 20
    p.setFont('Helvetica', 12)
    text = p.beginText(50, y)
    for line in (report.description or '').splitlines():
        text.textLine(line)
    p.drawText(text)
    p.showPage()
    p.save()
    return response

def ml_predict_soil_moisture(location, soil_moisture, temperature, humidity):
    # Stub ML function: returns a fake status and irrigation action
    # Replace with real ML model call
    if float(soil_moisture) < 30:
        status = 'Dry'
        irrigation_action = 'Irrigate'
    else:
        status = 'Normal'
        irrigation_action = 'No Action'
    return status, irrigation_action

def add_farm(request):
    """Technician add farm view (POST only)"""
    if request.method == 'POST':
        farm_name = request.POST.get('farm_name')
        location = request.POST.get('location')
        area_size = request.POST.get('area_size')
        soil_type = request.POST.get('soil_type')
        description = request.POST.get('description')
        if not (farm_name and location and area_size and soil_type):
            messages.error(request, 'All fields except description are required.')
            return redirect('technician:farm_locations')
        try:
            Farm.objects.create(
                user=request.user,
                farm_name=farm_name,
                location=location,
                area_size=area_size,
                soil_type=soil_type,
                description=description
            )
            messages.success(request, f'Farm "{farm_name}" added successfully!')
        except Exception as e:
            messages.error(request, f'Error adding farm: {e}')
        return redirect('technician:farm_locations')
    else:
        return redirect('technician:farm_locations')
