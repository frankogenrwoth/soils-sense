from django.shortcuts import render, redirect
from django.http import JsonResponse, HttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from authentication.models import Role, User
from django.core.exceptions import PermissionDenied
from .models import (
    Farm, Crop, SoilMoistureReading, 
    WeatherData, IrrigationEvent, Alert
)
from django.db.models import Q, Avg
from datetime import datetime, timedelta
import json
import csv
import io
import pandas as pd

# Keeping the decorator definition for future use, but not applying it
def farmer_required(view_func):
    """Decorator to check if the user is a farmer"""
    def wrapper(request, *args, **kwargs):
        if not request.user.is_authenticated:
            return redirect('authentication:login')
        if request.user.role != Role.FARMER:
            raise PermissionDenied("You must be a farmer to access this page.")
        return view_func(request, *args, **kwargs)
    return wrapper

@login_required
def dashboard(request):
    """Main farmer dashboard view with real-time data and ML predictions"""
    # Get the user's farms
    farms = Farm.objects.filter(user=request.user)
    
    # Get the selected farm (default to first farm)
    selected_farm_id = request.GET.get('farm_id')
    if selected_farm_id:
        selected_farm = farms.filter(id=selected_farm_id).first()
    else:
        selected_farm = farms.first()

    if not selected_farm:
        context = {'error': 'No farms found. Please add a farm first.'}
        return render(request, 'farmer/dashboard.html', context)

    # Get latest soil moisture reading
    latest_reading = SoilMoistureReading.objects.filter(
        farm=selected_farm
    ).first()  # Already ordered by -timestamp

    # Get weather data
    latest_weather = WeatherData.objects.filter(
        farm=selected_farm,
        is_forecast=False
    ).first()

    # Get latest irrigation event
    latest_irrigation = IrrigationEvent.objects.filter(
        farm=selected_farm
    ).first()

    # Get unread alerts
    recent_alerts = Alert.objects.filter(
        farm=selected_farm,
        is_read=False
    )[:3]

    # Get soil moisture data for the past 7 days
    seven_days_ago = datetime.now() - timedelta(days=7)
    moisture_history = SoilMoistureReading.objects.filter(
        farm=selected_farm,
        timestamp__gte=seven_days_ago
    ).order_by('timestamp')

    # Prepare chart data
    moisture_dates = [reading.timestamp.strftime('%Y-%m-%d') for reading in moisture_history]
    moisture_values = [float(reading.soil_moisture_percent) for reading in moisture_history]

    # Get daily averages for the stats
    yesterday = datetime.now() - timedelta(days=1)
    today_avg = SoilMoistureReading.objects.filter(
        farm=selected_farm,
        timestamp__date=datetime.now().date()
    ).aggregate(Avg('soil_moisture_percent'))['soil_moisture_percent__avg'] or 0

    yesterday_avg = SoilMoistureReading.objects.filter(
        farm=selected_farm,
        timestamp__date=yesterday.date()
    ).aggregate(Avg('soil_moisture_percent'))['soil_moisture_percent__avg'] or 0

    # Calculate moisture change
    moisture_change = today_avg - yesterday_avg if yesterday_avg > 0 else 0

    context = {
        'farms': farms,
        'selected_farm': selected_farm,
        'current_moisture': round(float(latest_reading.soil_moisture_percent), 1) if latest_reading else 0,
        'temperature': round(float(latest_reading.temperature_celsius), 1) if latest_reading else 0,
        'humidity': round(float(latest_reading.humidity_percent), 1) if latest_reading else 0,
        'moisture_change': round(moisture_change, 1),
        'moisture_change_direction': 'up' if moisture_change >= 0 else 'down',
        'moisture_dates': json.dumps(moisture_dates),
        'moisture_values': json.dumps(moisture_values),
        'recent_alerts': recent_alerts,
        'latest_irrigation': latest_irrigation,
        'weather_data': latest_weather,
        # Placeholders for ML model integration
        'prediction_available': False,  # Will be True when ML model is integrated
        'moisture_predictions': [],     # Will contain ML model predictions
        'recommendation': None,         # Will contain ML model recommendations
    }
    
    return render(request, 'farmer/dashboard.html', context)

def profile(request):
    """Profile management view"""
    if request.method == 'POST':
        email = request.POST.get('email')
        
        # Update user object
        user = request.user
        user.email = email
        user.save()
        
        messages.success(request, 'Profile updated successfully!')
        return redirect('farmer:profile')
    
    return render(request, 'farmer/profile.html')

@login_required
def farm_management(request):
    """Farm management view"""
    farms = Farm.objects.filter(user=request.user)
    crops = Crop.objects.filter(farm__user=request.user)
    
    context = {
        'farms': farms,
        'crops': crops
    }
    return render(request, 'farmer/farm_management.html', context)

@csrf_exempt
@require_http_methods(["POST"])
@login_required
def add_farm(request):
    """Add new farm location"""
    try:
        # Create farm
        farm = Farm.objects.create(
            user=request.user,
            farm_name=request.POST.get('farm_name'),
            location=request.POST.get('location'),
            area_size=request.POST.get('area_size'),
            soil_type=request.POST.get('soil_type'),
            description=request.POST.get('description')
        )
        
        return JsonResponse({'success': True})
        
    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)})

@csrf_exempt
@require_http_methods(["POST"])
@login_required
def add_crop(request):
    """Add new crop"""
    try:
        # Get the farm and verify ownership
        farm = Farm.objects.get(id=request.POST.get('farm'), user=request.user)
        
        # Create crop
        crop = Crop.objects.create(
            farm=farm,
            crop_name=request.POST.get('crop_name'),
            variety=request.POST.get('variety'),
            planting_date=request.POST.get('planting_date'),
            expected_harvest_date=request.POST.get('expected_harvest_date'),
            status=request.POST.get('status'),
            area_planted=request.POST.get('area_planted'),
            notes=request.POST.get('notes')
        )
        
        return JsonResponse({'success': True})
        
    except Farm.DoesNotExist:
        return JsonResponse({'success': False, 'error': 'Farm not found or access denied'})
    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)})

def analytics(request):
    """Analytics dashboard view"""
    return render(request, 'farmer/analytics.html')

def recommendations(request):
    """Smart recommendations view"""
    return render(request, 'farmer/recommendations.html')

@login_required
def soil_data_management(request):
    """View for managing soil moisture data"""
    farms = Farm.objects.filter(user=request.user)
    
    # Get filter parameters
    selected_farm_id = request.GET.get('farm')
    date_from = request.GET.get('date_from')
    date_to = request.GET.get('date_to')

    # Base query for readings
    readings = SoilMoistureReading.objects.filter(farm__user=request.user)

    # Apply filters
    if selected_farm_id:
        readings = readings.filter(farm_id=selected_farm_id)
    if date_from:
        readings = readings.filter(timestamp__date__gte=date_from)
    if date_to:
        readings = readings.filter(timestamp__date__lte=date_to)

    # Order by timestamp
    readings = readings.order_by('-timestamp')
    
    context = {
        'farms': farms,
        'readings': readings,
        'selected_farm_id': selected_farm_id,
        'date_from': date_from,
        'date_to': date_to
    }
    return render(request, 'farmer/soil_data_management.html', context)

@login_required
def add_soil_reading(request):
    """Add a manual soil moisture reading"""
    if request.method == 'POST':
        try:
            farm_id = request.POST.get('farm')
            farm = Farm.objects.get(id=farm_id, user=request.user)
            
            # Create the reading with all fields from the dataset structure
            reading = SoilMoistureReading.objects.create(
                farm=farm,
                sensor_id=request.POST.get('sensor_id'),
                soil_moisture_percent=float(request.POST.get('soil_moisture_percent')),
                temperature_celsius=float(request.POST.get('temperature_celsius')),
                humidity_percent=float(request.POST.get('humidity_percent')),
                battery_voltage=float(request.POST.get('battery_voltage')),
                status=request.POST.get('status'),
                irrigation_action=request.POST.get('irrigation_action'),
                timestamp=datetime.now(),
                reading_source='manual_input'
            )
            
            messages.success(request, 'Soil moisture reading added successfully!')
        except Farm.DoesNotExist:
            messages.error(request, 'Invalid farm selected.')
        except ValueError as e:
            messages.error(request, f'Invalid data format: {str(e)}')
        except Exception as e:
            messages.error(request, f'An error occurred: {str(e)}')
    
    return redirect('farmer:soil_data_management')

@login_required
def download_csv_template(request):
    """Provide a downloadable CSV template with the correct headers matching the model dataset."""
    response = HttpResponse(content_type='text/csv')
    response['Content-Disposition'] = 'attachment; filename="soil_moisture_template.csv"'
    
    writer = csv.writer(response)
    # Write headers matching the model dataset structure
    writer.writerow(['record_id', 'sensor_id', 'location', 'soil_moisture_percent', 
                    'temperature_celsius', 'humidity_percent', 'timestamp', 
                    'status', 'battery_voltage', 'irrigation_action'])
    
    # Write example row
    writer.writerow(['1', 'SENSOR_1', 'Farm A', '45.5', '25.3', '65.2', 
                    '2024-03-21 14:30:00', 'Normal', '3.62', 'None'])
    
    return response

@login_required
def upload_soil_data(request):
    """Upload soil moisture data from CSV file"""
    if request.method == 'POST':
        try:
            farm_id = request.POST.get('farm')
            farm = Farm.objects.get(id=farm_id, user=request.user)
            
            if 'file' not in request.FILES:
                messages.error(request, 'No file was uploaded.')
                return redirect('farmer:soil_data_management')
            
            csv_file = request.FILES['file']
            
            if not csv_file.name.endswith('.csv'):
                messages.error(request, 'Please upload a CSV file.')
                return redirect('farmer:soil_data_management')
            
            # Read the CSV file
            try:
                df = pd.read_csv(csv_file)
                required_columns = ['record_id', 'sensor_id', 'location', 'soil_moisture_percent',
                                'temperature_celsius', 'humidity_percent', 'timestamp',
                                'status', 'battery_voltage', 'irrigation_action']
                
                # Convert column names to lowercase and strip whitespace
                df.columns = df.columns.str.lower().str.strip()
                
                # Check if all required columns are present
                missing_columns = [col for col in required_columns if col.lower() not in df.columns]
                if missing_columns:
                    messages.error(request, f'Missing required columns: {", ".join(missing_columns)}. Please use our template.')
                    return redirect('farmer:soil_data_management')
                
                # Process each row
                readings_to_create = []
                errors = []
                
                for index, row in df.iterrows():
                    try:
                        # Clean and validate data
                        timestamp = pd.to_datetime(row['timestamp'])
                        soil_moisture = float(row['soil_moisture_percent'])
                        temperature = float(row['temperature_celsius'])
                        humidity = float(row['humidity_percent'])
                        battery = float(row['battery_voltage'])
                        
                        # Validate ranges
                        if not (0 <= soil_moisture <= 100):
                            raise ValueError('Soil moisture must be between 0 and 100%')
                        if not (0 <= humidity <= 100):
                            raise ValueError('Humidity must be between 0 and 100%')
                        if not (-50 <= temperature <= 100):
                            raise ValueError('Temperature must be between -50°C and 100°C')
                        if not (0 <= battery <= 5):
                            raise ValueError('Battery voltage must be between 0 and 5V')
                        
                        reading = SoilMoistureReading(
                            farm=farm,
                            sensor_id=row['sensor_id'],
                            timestamp=timestamp,
                            soil_moisture_percent=soil_moisture,
                            temperature_celsius=temperature,
                            humidity_percent=humidity,
                            status=row['status'],
                            battery_voltage=battery,
                            irrigation_action=row['irrigation_action'],
                            reading_source='csv_upload'
                        )
                        readings_to_create.append(reading)
                    except (ValueError, TypeError) as e:
                        errors.append(f'Row {index + 2}: {str(e)}')
                
                if errors:
                    error_message = 'Errors found in CSV file:\n' + '\n'.join(errors)
                    messages.error(request, error_message)
                    return redirect('farmer:soil_data_management')
                
                if not readings_to_create:
                    messages.error(request, 'No valid readings found in the CSV file.')
                    return redirect('farmer:soil_data_management')
                
                # Bulk create the readings
                SoilMoistureReading.objects.bulk_create(readings_to_create)
                messages.success(request, f'Successfully uploaded {len(readings_to_create)} soil moisture readings!')
                
            except pd.errors.EmptyDataError:
                messages.error(request, 'The uploaded CSV file is empty.')
            except pd.errors.ParserError:
                messages.error(request, 'Error parsing CSV file. Please make sure you are using our template format.')
            except Exception as e:
                messages.error(request, f'An error occurred while processing the file: {str(e)}')
        
        except Farm.DoesNotExist:
            messages.error(request, 'Invalid farm selected.')
        except Exception as e:
            messages.error(request, f'An error occurred: {str(e)}')
    
    return redirect('farmer:soil_data_management')

@login_required
def filter_soil_data(request):
    """API endpoint for filtering soil moisture data"""
    try:
        # Get filter parameters
        farm_id = request.GET.get('farm')
        date_from = request.GET.get('date_from')
        date_to = request.GET.get('date_to')

        # Base query
        readings = SoilMoistureReading.objects.filter(farm__user=request.user)

        # Apply filters
        if farm_id:
            readings = readings.filter(farm_id=farm_id)
        if date_from:
            readings = readings.filter(timestamp__date__gte=date_from)
        if date_to:
            readings = readings.filter(timestamp__date__lte=date_to)

        # Order by timestamp
        readings = readings.order_by('-timestamp')

        # Prepare data for response
        data = [{
            'farm_name': reading.farm.farm_name,
            'timestamp': reading.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
            'soil_moisture_percent': float(reading.soil_moisture_percent),
            'temperature_celsius': float(reading.temperature_celsius),
            'humidity_percent': float(reading.humidity_percent),
            'reading_source': reading.reading_source,
            'status': reading.status
        } for reading in readings]

        return JsonResponse({'data': data})
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)
