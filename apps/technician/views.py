from functools import wraps
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
from django.utils import timezone
import datetime

from django.contrib.auth.decorators import login_required, user_passes_test
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib import messages
from .models import Sensor
from apps.farmer.models import Farm
from .forms import SensorForm  # We will create this next

def is_technician(user):
    """Check if the user is authenticated and has technician role"""
    return user.is_authenticated and user.role == Role.TECHNICIAN

def technician_required(view_func):
    """
    Decorator that ensures a user is logged in and is a technician.
    Combines Django's login_required and user_passes_test decorators.
    """
    return login_required(user_passes_test(is_technician, login_url='authentication:login')(view_func))

@technician_required
def sensor_list(request):
    sensors = Sensor.objects.select_related('farm').all()
    return render(request, 'technician/sensor_list.html', {'sensors': sensors})

@technician_required
def sensor_add(request):
    if request.method == 'POST':
        form = SensorForm(request.POST)
        if form.is_valid():
            form.save()
            messages.success(request, 'Sensor added successfully.')
            return redirect('technician:sensor_list')
    else:
        form = SensorForm()
    
    # Get list of farms that have users associated with them
    farms = Farm.objects.filter(user__isnull=False)
    
    return render(request, 'technician/sensor_form.html', {
        'form': form,
        'title': 'Add Sensor',
        'farms': farms
    })

@technician_required
def sensor_edit(request, pk):
    sensor = get_object_or_404(Sensor, pk=pk)
    if request.method == 'POST':
        form = SensorForm(request.POST, instance=sensor)
        if form.is_valid():
            form.save()
            messages.success(request, 'Sensor updated successfully.')
            return redirect('technician:sensor_list')
    else:
        form = SensorForm(instance=sensor)
    return render(request, 'technician/sensor_form.html', {'form': form, 'title': 'Edit Sensor'})

@technician_required
def sensor_delete(request, pk):
    sensor = get_object_or_404(Sensor, pk=pk)
    if request.method == 'POST':
        sensor.delete()
        messages.success(request, 'Sensor deleted successfully.')
        return redirect('technician:sensor_list')
    return render(request, 'technician/sensor_confirm_delete.html', {'sensor': sensor})

from ml.config import REGRESSION_ALGORITHMS, CLASSIFICATION_ALGORITHMS, DEFAULT_ALGORITHMS
from ml.predictor import SoilMoisturePredictor, IrrigationRecommender
from apps.farmer.models import PredictionResult




# Keeping the decorator definition for future use, but not applying it
def technician_required(view_func):
    @login_required
    @wraps(view_func)
    def wrapper(request, *args, **kwargs):
        user = request.user
        if user.is_authenticated and user.role == Role.TECHNICIAN:
            return view_func(request, *args, **kwargs)
     
    return wrapper

@technician_required
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
    
    # Recent reports (exclude auto-generated farm registration reports)
    from .models import Report
    recent_reports = Report.objects.select_related('farm').exclude(title='Farm Registered').order_by('-created_at')[:3]
    
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

     # New sensor info
    sensors = Sensor.objects.all()  # fetch all sensors
    active_sensors_count = sensors.filter(is_active=True).count()
    total_sensors_count = sensors.count()
    
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
        'sensors': sensors,
        'active_sensors_count': active_sensors_count,
        'total_sensors_count': total_sensors_count,
    }
    return render(request, 'technician/dashboard.html', context)

@technician_required
def profile(request):
    user = request.user
    image_url = user.image.url if user.image else ''
    has_custom_image = bool(user.image and not image_url.endswith('default.webp'))
    return render(request, 'technician/profile.html', {'user': user, 'has_custom_image': has_custom_image})

@technician_required
def farm_locations(request):
    farms = Farm.objects.filter(user__isnull=False)  # Only show farms added by farmers
    return render(request, 'technician/farm_locations.html', {'farms': farms})

@technician_required
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

@technician_required
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

@technician_required
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

@technician_required
def analytics(request):
    """Analytics dashboard view"""
    
    # Get date range (last 30 days)
    end_date = datetime.datetime.now()
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

@technician_required
def reports(request):
    farms = Farm.objects.filter(user__isnull=False)  # Only show farms added by farmers
    predictions = PredictionResult.objects.select_related('farm').order_by('-created_at')

    # Add recent reports for the reports page (exclude auto-generated farm registration reports)
    from .models import Report
    recent_reports = Report.objects.select_related('farm').exclude(title='Farm Registered').order_by('-created_at')[:3]

    if request.method == 'POST':
        try:
            farm_id = request.POST.get('farm')
            farm = get_object_or_404(Farm, id=farm_id)

            # Get form data
            location = request.POST.get('location')
            temperature = float(request.POST.get('temperature'))
            humidity = float(request.POST.get('humidity'))
            battery_voltage = float(request.POST.get('battery_voltage'))
            algorithm = request.POST.get('algorithm')
            algorithm_irr = request.POST.get('algorithm_irr')

            # Make predictions
            soil_predictor = SoilMoisturePredictor()
            irrigation_recommender = IrrigationRecommender()

            # Get current timestamp
            from django.utils import timezone
            current_time = timezone.now()

            # Predict soil moisture
            soil_moisture_result = soil_predictor.predict_moisture(
                sensor_id=1,  # Default sensor ID
                location=location,
                temperature_celsius=temperature,
                humidity_percent=humidity,
                battery_voltage=battery_voltage,
                status="Normal",  # Default status
                irrigation_action="None",  # Default - no irrigation
                timestamp=current_time,
                algorithm=algorithm
            )

            # Get the predicted value
            soil_moisture_value = soil_moisture_result['predicted_value'] if isinstance(soil_moisture_result, dict) else soil_moisture_result

            # Get irrigation recommendation
            irrigation_result = irrigation_recommender.recommend_irrigation(
                soil_moisture_percent=soil_moisture_value,
                temperature_celsius=temperature,
                humidity_percent=humidity,
                battery_voltage=battery_voltage,
                status="Normal",
                timestamp=current_time,
                algorithm=algorithm_irr
            )

            # Get the recommendation value
            irrigation_recommendation = irrigation_result['predicted_value'] if isinstance(irrigation_result, dict) else irrigation_result

            # Save the prediction result
            PredictionResult.objects.create(
                farm=farm,
                location=location,
                temperature=temperature,
                humidity=humidity,
                battery_voltage=battery_voltage,
                soil_moisture_result=soil_moisture_value,
                irrigation_result=irrigation_recommendation,
                algorithm=algorithm,
                algorithm_irr=algorithm_irr
            )

            messages.success(request, 'Prediction created successfully!')
            return redirect('technician:reports')

        except Exception as e:
            messages.error(request, f'Error making prediction: {str(e)}')
            return redirect('technician:reports')

    context = {
        'farms': farms,
        'predictions': predictions,
        'soil_algorithms': REGRESSION_ALGORITHMS,
        'irrigation_algorithms': CLASSIFICATION_ALGORITHMS,
        'default_soil_algorithm': DEFAULT_ALGORITHMS['soil_moisture_predictor'],
        'default_irrigation_algorithm': DEFAULT_ALGORITHMS['irrigation_recommendation'],
        'recent_reports': recent_reports,  # Add this line
    }
    return render(request, 'technician/reports.html', context)

@technician_required
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

@technician_required
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

@technician_required
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

@technician_required
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

@technician_required
def delete_threshold(request, pk):
    """Delete sensor threshold"""
    from .models import SensorThreshold
    
    threshold = get_object_or_404(SensorThreshold, pk=pk)
    if request.method == 'POST':
        threshold.delete()
        messages.success(request, 'Sensor threshold deleted successfully!')
        return redirect('technician:sensor_config')
    
    return render(request, 'technician/confirm_delete_threshold.html', {'threshold': threshold})

@technician_required
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

@technician_required
def delete_report(request, pk):
    """Delete report"""
    from .models import Report
    
    report = get_object_or_404(Report, pk=pk)
    if request.method == 'POST':
        report.delete()
        messages.success(request, 'Report deleted successfully!')
        return redirect('technician:reports')
    
    return render(request, 'technician/confirm_delete_report.html', {'report': report})

@technician_required
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

@technician_required
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
    # Extract algorithm from description
    algorithm = None
    for line in (report.description or '').splitlines():
        if line.startswith('Algorithm:'):
            algorithm = line.replace('Algorithm:', '').strip()
            break
    if algorithm:
        y -= 20
        p.drawString(50, y, f'Algorithm: {algorithm}')
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

@technician_required
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

@technician_required
def models_view(request):
    ml_engine = MLEngine()
    model_data = [ml_engine.get_model_info(model_type) for model_type in ml_engine.get_available_models()]
    return render(request, 'technician/models.html', {'models': model_data})

@technician_required
def delete_prediction(request, pk):
    """Delete a prediction result from Prediction History"""
    prediction = get_object_or_404(PredictionResult, pk=pk)
    if request.method == 'POST':
        prediction.delete()
        messages.success(request, 'Prediction deleted successfully!')
        return redirect('technician:reports')
    return HttpResponse(status=405)  # Method not allowed for GET

@technician_required
def download_predictionresult_pdf(request, pk):
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.units import inch, cm
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.enums import TA_CENTER
        from io import BytesIO
        import os
        from django.conf import settings

        # Get the prediction (no farm__user restriction for technician)
        prediction = get_object_or_404(PredictionResult, id=pk)

        # Create the PDF buffer
        buffer = BytesIO()

        # Set up the document with margins
        doc = SimpleDocTemplate(
            buffer,
            pagesize=A4,
            rightMargin=1.5*cm,
            leftMargin=1.5*cm,
            topMargin=1.5*cm,
            bottomMargin=1.5*cm
        )

        # Styles
        styles = getSampleStyleSheet()
        styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            spaceAfter=30,
            textColor=colors.HexColor('#1a5f7a'),
            alignment=TA_CENTER
        ))
        styles.add(ParagraphStyle(
            name='SubTitle',
            parent=styles['Heading2'],
            fontSize=16,
            textColor=colors.HexColor('#2c3e50'),
            spaceAfter=20,
            spaceBefore=20
        ))
        styles.add(ParagraphStyle(
            name='SectionTitle',
            parent=styles['Heading3'],
            fontSize=14,
            textColor=colors.HexColor('#34495e'),
            spaceAfter=10,
            spaceBefore=10
        ))
        styles.add(ParagraphStyle(
            name='NormalText',
            parent=styles['Normal'],
            fontSize=10,
            textColor=colors.HexColor('#2c3e50'),
            spaceAfter=5
        ))

        # Start building the document
        elements = []

        # Add logo if exists (skip gracefully if not found)
        try:
            logo_path = os.path.join(settings.STATIC_ROOT, 'images', 'logo.png')
            if os.path.exists(logo_path):
                logo = Image(logo_path)
                logo.drawHeight = 1.5*inch
                logo.drawWidth = 1.5*inch
                elements.append(logo)
                elements.append(Spacer(1, 20))
        except Exception:
            pass

        # Title
        elements.append(Paragraph('Soil Sense - Prediction and Irrigation Recommendation Report', styles['CustomTitle']))

        # Farm Information Section (no farmer info)
        elements.append(Paragraph('Farm Information', styles['SubTitle']))
        farm_info = [
            ['Farm Name:', prediction.farm.farm_name],
            ['Farm Location:', prediction.farm.location],
            ['Area Size:', f"{prediction.farm.area_size} hectares"],
            ['Soil Type:', prediction.farm.soil_type]
        ]
        farm_table = Table(farm_info, colWidths=[120, 350])
        farm_table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('TEXTCOLOR', (0, 0), (0, -1), colors.HexColor('#34495e')),
            ('TEXTCOLOR', (1, 0), (1, -1), colors.HexColor('#2c3e50')),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#ecf0f1')),
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#f8f9fa')),
        ]))
        elements.append(farm_table)
        elements.append(Spacer(1, 20))

        # Prediction Results Section
        elements.append(Paragraph('Prediction Results', styles['SubTitle']))
        prediction_data = [
            ['Soil Moisture Prediction', 'Irrigation Recommendation'],
            [f"{prediction.soil_moisture_result:.2f}%", prediction.irrigation_result],
            ['Algorithm: ' + prediction.algorithm, 'Algorithm: ' + prediction.algorithm_irr]
        ]
        prediction_table = Table(prediction_data, colWidths=[235, 235])
        prediction_table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('FONTSIZE', (0, 1), (-1, 1), 20),
            ('FONTSIZE', (0, 2), (-1, 2), 9),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.HexColor('#2c3e50')),
            ('TEXTCOLOR', (0, 1), (-1, 1), colors.HexColor('#1a5f7a')),
            ('TEXTCOLOR', (0, 2), (-1, 2), colors.HexColor('#7f8c8d')),
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#ecf0f1')),
            ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#bdc3c7')),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
            ('TOPPADDING', (0, 0), (-1, -1), 12),
        ]))
        elements.append(prediction_table)
        elements.append(Spacer(1, 20))

        # Measurement Details
        elements.append(Paragraph('Measurement Details', styles['SubTitle']))
        measurement_data = [
            ['Parameter', 'Value', 'Unit'],
            ['Temperature', f"{prediction.temperature:.1f}", '°C'],
            ['Humidity', f"{prediction.humidity:.1f}", '%'],
            ['Battery Voltage', f"{prediction.battery_voltage:.1f}", 'V'],
            ['Location', prediction.location, ''],
            ['Date & Time', prediction.created_at.strftime('%B %d, %Y %H:%M'), '']
        ]
        measurement_table = Table(measurement_data, colWidths=[160, 160, 150])
        measurement_table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#ecf0f1')),
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#34495e')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('ALIGN', (1, 1), (2, -2), 'CENTER'),
        ]))
        elements.append(measurement_table)

        # Footer
        elements.append(Spacer(1, 40))
        footer_text = f"Generated by Soil Sense on {timezone.now().strftime('%B %d, %Y %H:%M')}"
        footer_style = ParagraphStyle(
            'Footer',
            parent=styles['Normal'],
            fontSize=8,
            textColor=colors.HexColor('#95a5a6'),
            alignment=TA_CENTER
        )
        elements.append(Paragraph(footer_text, footer_style))

        # Build PDF
        doc.build(elements)
        pdf = buffer.getvalue()
        buffer.close()

        response = HttpResponse(content_type='application/pdf')
        response['Content-Disposition'] = f'attachment; filename="soil_sense_prediction_{pk}.pdf"'
        response.write(pdf)
        return response
    except Exception as e:
        messages.error(request, f'Error generating PDF: {str(e)}')
        return redirect('technician:reports')
