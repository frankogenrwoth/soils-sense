from django.shortcuts import render, redirect, get_object_or_404
from django.http import JsonResponse, HttpResponse
from django.views.decorators.http import require_http_methods, require_POST
from django.views.decorators.csrf import csrf_protect
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from authentication.models import Role
from django.core.exceptions import PermissionDenied
from .models import (
    Farm, Crop, SoilMoistureReading, 
    WeatherData, IrrigationEvent, Alert, PredictionResult
)
from django.db.models import Avg
from datetime import datetime, timedelta
import json
import csv
import pandas as pd
from ml.predictor import SoilMoisturePredictor, IrrigationRecommender
from ml.config import REGRESSION_ALGORITHMS, CLASSIFICATION_ALGORITHMS, DEFAULT_ALGORITHMS
from django.utils import timezone
import os
from django.conf import settings

def farmer_required(view_func):
    def wrapper(request, *args, **kwargs):
        if not request.user.is_authenticated:
            return redirect('authentication:login')
        if request.user.role != Role.FARMER:
            raise PermissionDenied("You must be a farmer to access this page.")
        return view_func(request, *args, **kwargs)
    return wrapper

@login_required
def dashboard(request):
    farms = Farm.objects.filter(user=request.user)
    
    selected_farm_id = request.GET.get('farm_id')
    if selected_farm_id:
        selected_farm = farms.filter(id=selected_farm_id).first()
    else:
        selected_farm = farms.first()

    if not selected_farm:
        context = {'error': 'No farms found. Please add a farm first.'}
        return render(request, 'farmer/dashboard.html', context)

    latest_reading = SoilMoistureReading.objects.filter(
        farm=selected_farm
    ).order_by('-timestamp').first()

    latest_weather = WeatherData.objects.filter(
        farm=selected_farm,
        is_forecast=False
    ).order_by('-timestamp').first()

    latest_irrigation = IrrigationEvent.objects.filter(
        farm=selected_farm
    ).order_by('-start_time').first()

    recent_alerts = Alert.objects.filter(
        farm=selected_farm,
        is_read=False
    )[:3]

    seven_days_ago = datetime.now() - timedelta(days=7)
    moisture_history = SoilMoistureReading.objects.filter(
        farm=selected_farm,
        timestamp__gte=seven_days_ago,
        soil_moisture_percent__isnull=False  # Exclude null values
    ).order_by('timestamp')

    # Filter out any None values and safely convert to float
    moisture_dates = []
    moisture_values = []
    for reading in moisture_history:
        try:
            if reading.soil_moisture_percent is not None:
                moisture_dates.append(reading.timestamp.strftime('%Y-%m-%d'))
                moisture_values.append(float(reading.soil_moisture_percent))
        except (ValueError, TypeError):
            continue

    yesterday = datetime.now() - timedelta(days=1)
    today_avg = SoilMoistureReading.objects.filter(
        farm=selected_farm,
        timestamp__date=datetime.now().date(),
        soil_moisture_percent__isnull=False  # Exclude null values
    ).aggregate(Avg('soil_moisture_percent'))['soil_moisture_percent__avg'] or 0

    yesterday_avg = SoilMoistureReading.objects.filter(
        farm=selected_farm,
        timestamp__date=yesterday.date(),
        soil_moisture_percent__isnull=False  # Exclude null values
    ).aggregate(Avg('soil_moisture_percent'))['soil_moisture_percent__avg'] or 0

    moisture_change = today_avg - yesterday_avg if yesterday_avg > 0 else 0

    # Safely handle None values when getting latest readings
    try:
        current_moisture = round(float(latest_reading.soil_moisture_percent), 1) if latest_reading and latest_reading.soil_moisture_percent is not None else 0
        temperature = round(float(latest_reading.temperature_celsius), 1) if latest_reading and latest_reading.temperature_celsius is not None else 0
        humidity = round(float(latest_reading.humidity_percent), 1) if latest_reading and latest_reading.humidity_percent is not None else 0
    except (ValueError, TypeError, AttributeError):
        current_moisture = 0
        temperature = 0
        humidity = 0

    context = {
        'farms': farms,
        'selected_farm': selected_farm,
        'current_moisture': current_moisture,
        'temperature': temperature,
        'humidity': humidity,
        'moisture_change': round(moisture_change, 1),
        'moisture_change_direction': 'up' if moisture_change >= 0 else 'down',
        'moisture_dates': json.dumps(moisture_dates),
        'moisture_values': json.dumps(moisture_values),
        'recent_alerts': recent_alerts,
        'latest_irrigation': latest_irrigation,
        'weather_data': latest_weather,
        'prediction_available': False,
        'moisture_predictions': [],
        'recommendation': None,
    }
    
    return render(request, 'farmer/dashboard.html', context)

@login_required
def profile(request):
    if request.method == 'POST':
        if 'update_profile' in request.POST:
            # Get form data
            email = request.POST.get('email', '').strip()
            first_name = request.POST.get('first_name', '').strip()
            last_name = request.POST.get('last_name', '').strip()
            
            # Set a default email if none provided
            if not email:
                email = f"{request.user.username}@soilsense.com"
                
            # Update user fields
            request.user.email = email
            request.user.first_name = first_name
            request.user.last_name = last_name
            
            try:
                request.user.save()
                messages.success(request, 'Profile updated successfully!')
            except Exception as e:
                messages.error(request, f'Error updating profile: {str(e)}')
        
        elif 'change_password' in request.POST:
            old_password = request.POST.get('old_password')
            new_password1 = request.POST.get('new_password1')
            new_password2 = request.POST.get('new_password2')
            
            if not request.user.check_password(old_password):
                messages.error(request, 'Current password is incorrect.')
            elif new_password1 != new_password2:
                messages.error(request, 'New passwords do not match.')
            elif len(new_password1) < 8:
                messages.error(request, 'Password must be at least 8 characters long.')
            else:
                request.user.set_password(new_password1)
                request.user.save()
                messages.success(request, 'Password changed successfully. Please log in again.')
                return redirect('authentication:login')
                
        return redirect('farmer:profile')
        
    # Get user's farms count
    farms_count = Farm.objects.filter(user=request.user).count()
    
    context = {
        'farms_count': farms_count,
        'edit_mode': request.GET.get('edit', False)
    }
    return render(request, 'farmer/profile.html', context)

@login_required
def farm_management(request):
    farms = Farm.objects.filter(user=request.user)
    crops = Crop.objects.filter(farm__user=request.user)
    
    context = {
        'farms': farms,
        'crops': crops
    }
    return render(request, 'farmer/farm_management.html', context)

@require_http_methods(["POST"])
@login_required
def add_farm(request):
    try:
        farm = Farm.objects.create(
            user=request.user,
            farm_name=request.POST.get('farm_name'),
            location=request.POST.get('location'),
            area_size=request.POST.get('area_size'),
            soil_type=request.POST.get('soil_type'),
            description=request.POST.get('description')
        )
        
        messages.success(request, f'Farm "{farm.farm_name}" has been added successfully!')
        return redirect('farmer:farm_management')
        
    except Exception as e:
        messages.error(request, str(e))
        return redirect('farmer:farm_management')

@require_http_methods(["POST"])
@login_required
def add_crop(request):
    try:
        farm = Farm.objects.get(id=request.POST.get('farm'), user=request.user)
        
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
        
        messages.success(request, f'Crop "{crop.crop_name}" has been added to {farm.farm_name} successfully!')
        return redirect('farmer:farm_management')
        
    except Farm.DoesNotExist:
        messages.error(request, 'Farm not found or access denied')
        return redirect('farmer:farm_management')
    except Exception as e:
        messages.error(request, str(e))
        return redirect('farmer:farm_management')

@login_required
def analytics(request):
    # Get user's farms
    farms = Farm.objects.filter(user=request.user)
    
    # Get selected farm or default to first farm
    selected_farm_id = request.GET.get('farm_id')
    if selected_farm_id:
        selected_farm = farms.filter(id=selected_farm_id).first()
    else:
        selected_farm = farms.first()

    if not selected_farm:
        messages.error(request, 'No farms found. Please add a farm first.')
        return render(request, 'farmer/analytics.html', {'farms': []})

    # Get date range (default to last 7 days)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=7)

    # Get soil moisture readings for the selected farm
    moisture_readings = SoilMoistureReading.objects.filter(
        farm=selected_farm,
        timestamp__range=(start_date, end_date)
    ).order_by('timestamp')

    # Initialize ML predictor
    predictor = SoilMoisturePredictor()
    
    # Get algorithm from request or use default
    algorithm = request.GET.get('algorithm', DEFAULT_ALGORITHMS["soil_moisture_predictor"])

    # Prepare data for charts with predictions
    dates = []
    moisture_values = []
    temp_values = []
    humidity_values = []
    predicted_moisture_values = []

    for reading in moisture_readings:
        try:
            # Get prediction for each reading
            prediction_result = predictor.predict_moisture(
                sensor_id=reading.sensor_id,
                location=reading.farm.location,
                temperature_celsius=float(reading.temperature_celsius),
                humidity_percent=float(reading.humidity_percent),
                battery_voltage=float(reading.battery_voltage),
                status=reading.status,
                irrigation_action="None",  # Default to None since this is historical data
                timestamp=reading.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                algorithm=algorithm
            )
            
            if prediction_result.get('success'):
                dates.append(reading.timestamp.strftime('%Y-%m-%d'))
                predicted_value = prediction_result['predicted_value']
                predicted_moisture_values.append(round(float(predicted_value), 1))
                temp_values.append(float(reading.temperature_celsius))
                humidity_values.append(float(reading.humidity_percent))
        except Exception as e:
            continue

    # Calculate average predicted moisture
    avg_moisture = (sum(predicted_moisture_values) / len(predicted_moisture_values)) if predicted_moisture_values else 0
    avg_moisture = round(avg_moisture, 1)

    # Get irrigation events
    irrigation_events = IrrigationEvent.objects.filter(
        farm=selected_farm,
        start_time__range=(start_date, end_date)
    ).count()

    # Calculate water savings based on predicted values
    water_saved = sum(
        max(0, (moisture - 70) * 0.1)  # 0.1L saved per 1% above optimal
        for moisture in predicted_moisture_values
        if moisture > 70
    )

    # Get weather impact
    weather_data = WeatherData.objects.filter(
        farm=selected_farm,
        timestamp__range=(start_date, end_date)
    )

    # Calculate weather effects
    rainfall_effect = weather_data.filter(precipitation__gt=0).count()
    high_temp_effect = weather_data.filter(temperature__gt=30).count()
    wind_effect = weather_data.filter(wind_speed__gt=20).count()
    
    context = {
        'farms': farms,
        'selected_farm': selected_farm,
        'dates': json.dumps(dates),
        'moisture_values': json.dumps(predicted_moisture_values),  # Use predicted values
        'temp_values': json.dumps(temp_values),
        'humidity_values': json.dumps(humidity_values),
        'avg_moisture': avg_moisture,
        'irrigation_events': irrigation_events,
        'water_saved': round(water_saved, 1),
        'rainfall_effect': round((rainfall_effect / max(1, len(dates))) * 100, 1),
        'temperature_effect': round((high_temp_effect / max(1, len(dates))) * 100, 1),
        'wind_effect': round((wind_effect / max(1, len(dates))) * 100, 1),
        'selected_algorithm': algorithm,  # Add selected algorithm to context
        'available_algorithms': list(REGRESSION_ALGORITHMS.keys()),  # Add available algorithms
    }
    
    return render(request, 'farmer/analytics.html', context)


@login_required
def recommendations(request):
    # Get user's farms
    farms = Farm.objects.filter(user=request.user)
    
    # Get selected farm or default to first farm
    selected_farm_id = request.GET.get('farm_id')
    if selected_farm_id:
        selected_farm = farms.filter(id=selected_farm_id).first()
    else:
        selected_farm = farms.first()

    if not selected_farm:
        return render(request, 'farmer/recommendations.html', {'farms': farms})

    # Get latest soil moisture reading for input data
    latest_reading = SoilMoistureReading.objects.filter(
        farm=selected_farm
    ).order_by('-timestamp').first()

    # Initialize ML predictor and get algorithm
    predictor = SoilMoisturePredictor()
    algorithm = request.GET.get('algorithm', DEFAULT_ALGORITHMS["soil_moisture_predictor"])

    # Get current moisture through prediction
    current_moisture = 0
    moisture_status = 'Unknown'
    
    if latest_reading:
        try:
            # Get prediction using the ML model
            prediction_result = predictor.predict_moisture(
                sensor_id=latest_reading.sensor_id,
                location=selected_farm.location,
                temperature_celsius=float(latest_reading.temperature_celsius),
                humidity_percent=float(latest_reading.humidity_percent),
                battery_voltage=float(latest_reading.battery_voltage),
                status=latest_reading.status,
                irrigation_action="None",  # Default to None since this is current reading
                timestamp=latest_reading.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                algorithm=algorithm
            )
            
            if prediction_result.get('success'):
                current_moisture = round(float(prediction_result['predicted_value']), 1)
                # Determine moisture status based on predicted value
                if current_moisture < 30:
                    moisture_status = 'Low'
                elif current_moisture > 70:
                    moisture_status = 'High'
                else:
                    moisture_status = 'Optimal'
        except Exception as e:
            print(f"Prediction error: {str(e)}")
            # Fallback to raw reading if prediction fails
            current_moisture = round(float(latest_reading.soil_moisture_percent), 1)

    # Get irrigation recommendation based on predicted moisture
    irrigation_recommendation = "No irrigation needed at this time."
    water_amount = None
    if moisture_status == 'Low':
        irrigation_recommendation = "Immediate irrigation recommended"
        # Calculate water amount based on area and moisture deficit
        area_size = float(selected_farm.area_size)
        moisture_deficit = 50 - current_moisture  # Assuming 50% is optimal
        water_amount = round(area_size * moisture_deficit * 100)  # Simple calculation, adjust as needed
    elif moisture_status == 'High':
        irrigation_recommendation = "Hold irrigation until soil moisture decreases"

    # Get farm's crops and prepare recommendations
    crops = []
    for crop in Crop.objects.filter(farm=selected_farm):
        # Use the predicted moisture value for crop recommendations
        moisture_percentage = current_moisture
        if current_moisture < 30:
            moisture_status = 'Critical'
            status_class = 'text-red-600'
        elif current_moisture < 50:
            moisture_status = 'Warning'
            status_class = 'text-yellow-600'
        else:
            moisture_status = 'Good'
            status_class = 'text-green-600'

        # Generate recommended actions based on predicted conditions
        recommended_actions = []
        if moisture_status == 'Critical':
            recommended_actions.append("Implement immediate irrigation")
            recommended_actions.append("Consider mulching to retain moisture")
        elif moisture_status == 'Warning':
            recommended_actions.append("Schedule irrigation within next 24 hours")
            recommended_actions.append("Monitor soil moisture closely")
        else:
            recommended_actions.append("Maintain current irrigation schedule")
            recommended_actions.append("Monitor weather forecast for changes")

        # Add crop-specific recommendations based on growth stage
        days_since_planting = (datetime.now().date() - crop.planting_date).days
        days_to_harvest = (crop.expected_harvest_date - datetime.now().date()).days

        if days_since_planting < 30:  # Early stage
            recommended_actions.append("Focus on root development")
        elif days_to_harvest < 30:  # Near harvest
            recommended_actions.append("Prepare for harvest planning")
        
        crops.append({
            'crop_name': crop.crop_name,
            'variety': crop.variety,
            'status': crop.status,
            'planting_date': crop.planting_date,
            'expected_harvest_date': crop.expected_harvest_date,
            'moisture_status': moisture_status,
            'moisture_percentage': moisture_percentage,
            'recommended_actions': recommended_actions
        })

    context = {
        'farms': farms,
        'selected_farm': selected_farm,
        'current_moisture': current_moisture,
        'moisture_status': moisture_status,
        'irrigation_recommendation': irrigation_recommendation,
        'water_amount': water_amount,
        'crops': crops,
        'selected_algorithm': algorithm,  # Add selected algorithm to context
        'available_algorithms': list(REGRESSION_ALGORITHMS.keys()),  # Add available algorithms
    }
    
    return render(request, 'farmer/recommendations.html', context)
#Predictions here

@login_required
def predictions(request):
    farms = Farm.objects.filter(user=request.user)
    
    # Get all predictions for the user's farms
    predictions = PredictionResult.objects.filter(
        farm__in=farms
    ).select_related('farm').order_by('-created_at')
    
    if request.method == 'POST':
        try:
            farm_id = request.POST.get('farm')
            farm = get_object_or_404(Farm, id=farm_id, user=request.user)
            
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
            prediction = PredictionResult.objects.create(
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
            return redirect('farmer:predictions')
            
        except Exception as e:
            messages.error(request, f'Error making prediction: {str(e)}')
            return redirect('farmer:predictions')
    
    context = {
        'farms': farms,
        'predictions': predictions,
        'soil_algorithms': REGRESSION_ALGORITHMS,
        'irrigation_algorithms': CLASSIFICATION_ALGORITHMS,
        'default_soil_algorithm': DEFAULT_ALGORITHMS['soil_moisture_predictor'],
        'default_irrigation_algorithm': DEFAULT_ALGORITHMS['irrigation_recommendation']
    }
    return render(request, 'farmer/predictions.html', context)

@login_required
def get_latest_readings(request, farm_id):
    try:
        # Get the farm and verify ownership
        farm = get_object_or_404(Farm, id=farm_id, user=request.user)
        
        # Get the latest reading for this farm
        latest_reading = SoilMoistureReading.objects.filter(
            farm=farm
        ).order_by('-timestamp').first()
        
        if not latest_reading:
            return JsonResponse({
                'error': 'No readings found for this farm'
            }, status=404)
            
        # Return the reading data
        return JsonResponse({
            'location': farm.location,
            'temperature_celsius': latest_reading.temperature_celsius,
            'humidity_percent': latest_reading.humidity_percent,
            'battery_voltage': latest_reading.battery_voltage,
            'soil_moisture_percent': latest_reading.soil_moisture_percent,
            'status': latest_reading.status,
            'timestamp': latest_reading.timestamp.strftime('%Y-%m-%d %H:%M:%S')
        })
        
    except Farm.DoesNotExist:
        return JsonResponse({
            'error': 'Farm not found or access denied'
        }, status=404)
    except Exception as e:
        return JsonResponse({
            'error': str(e)
        }, status=500)

@login_required
def soil_data_management(request):
    farms = Farm.objects.filter(user=request.user)
    
    selected_farm_id = request.GET.get('farm')
    date_from = request.GET.get('date_from')
    date_to = request.GET.get('date_to')

    readings = SoilMoistureReading.objects.filter(farm__user=request.user)

    if selected_farm_id:
        readings = readings.filter(farm_id=selected_farm_id)
    if date_from:
        readings = readings.filter(timestamp__date__gte=date_from)
    if date_to:
        readings = readings.filter(timestamp__date__lte=date_to)

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
@require_POST
def add_soil_reading(request):
    try:
        farm = get_object_or_404(Farm, id=request.POST.get('farm'), user=request.user)
        
        # Calculate soil moisture based on other parameters
        # This is a placeholder calculation - replace with your actual algorithm
        temperature = float(request.POST.get('temperature_celsius'))
        humidity = float(request.POST.get('humidity_percent'))
        
        # Simple example calculation (replace with your actual algorithm)
        soil_moisture = (humidity * 0.7) + (30 - temperature * 0.3)  # This is just an example
        soil_moisture = max(0, min(100, soil_moisture))  # Ensure value is between 0 and 100
        
        reading = SoilMoistureReading.objects.create(
            farm=farm,
            sensor_id=request.POST.get('sensor_id'),
            soil_moisture_percent=soil_moisture,
            temperature_celsius=temperature,
            humidity_percent=humidity,
            battery_voltage=request.POST.get('battery_voltage'),
            status=request.POST.get('status'),
            reading_source='manual_input'
        )
        
        return JsonResponse({'success': True, 'message': 'Reading added successfully'})
    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)})

@login_required
@csrf_protect
@require_POST
def delete_reading(request, reading_id):
    try:
        reading = get_object_or_404(
            SoilMoistureReading.objects.select_related('farm'),
            id=reading_id,
            farm__user=request.user
        )
        
        reading.delete()
        
        return JsonResponse({
            'success': True,
            'message': 'Reading deleted successfully'
        })
        
    except SoilMoistureReading.DoesNotExist:
        return JsonResponse({
            'success': False,
            'error': 'Reading not found'
        }, status=404)
    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': str(e)
        }, status=500)

@login_required
@csrf_protect
@require_POST
def delete_farm(request, farm_id):
    try:
        farm = Farm.objects.filter(id=farm_id, user=request.user).first()
        
        if not farm:
            return JsonResponse({
                'success': False,
                'error': 'Farm not found or you do not have permission to delete it'
            }, status=404)
        
        farm.delete()
        
        return JsonResponse({
            'success': True,
            'message': 'Farm deleted successfully'
        })
    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': str(e)
        }, status=500)

@login_required
@csrf_protect
@require_POST
def delete_crop(request, crop_id):
    try:
        crop = Crop.objects.filter(id=crop_id, farm__user=request.user).first()
        
        if not crop:
            return JsonResponse({
                'success': False,
                'error': 'Crop not found or you do not have permission to delete it'
            }, status=404)
        
        crop.delete()
        
        return JsonResponse({
            'success': True,
            'message': 'Crop deleted successfully'
        })
    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': str(e)
        }, status=500)

@login_required
def download_csv_template(request):
    response = HttpResponse(content_type='text/csv')
    response['Content-Disposition'] = 'attachment; filename="soil_moisture_template.csv"'
    
    writer = csv.writer(response)
    writer.writerow(['record_id', 'sensor_id', 'location', 'soil_moisture_percent', 
                    'temperature_celsius', 'humidity_percent', 'timestamp', 
                    'status', 'battery_voltage', 'irrigation_action'])
    
    writer.writerow(['1', 'SENSOR_1', 'Farm A', '45.5', '25.3', '65.2', 
                    '2024-03-21 14:30:00', 'Normal', '3.62', 'None'])
    
    return response

@login_required
def upload_soil_data(request):
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
            
            try:
                df = pd.read_csv(csv_file)
                required_columns = ['record_id', 'sensor_id', 'location', 'soil_moisture_percent',
                                'temperature_celsius', 'humidity_percent', 'timestamp',
                                'status', 'battery_voltage']
                
                df.columns = df.columns.str.lower().str.strip()
                
                missing_columns = [col for col in required_columns if col.lower() not in df.columns]
                if missing_columns:
                    messages.error(request, f'Missing required columns: {", ".join(missing_columns)}. Please use our template.')
                    return redirect('farmer:soil_data_management')
                
                readings_to_create = []
                errors = []
                
                for index, row in df.iterrows():
                    try:
                        timestamp = pd.to_datetime(row['timestamp'])
                        soil_moisture = float(row['soil_moisture_percent'])
                        temperature = float(row['temperature_celsius'])
                        humidity = float(row['humidity_percent'])
                        battery = float(row['battery_voltage'])
                        
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
    try:
        farm_id = request.GET.get('farm')
        date_from = request.GET.get('date_from')
        date_to = request.GET.get('date_to')

        readings = SoilMoistureReading.objects.filter(farm__user=request.user)

        if farm_id:
            readings = readings.filter(farm_id=farm_id)
        if date_from:
            readings = readings.filter(timestamp__date__gte=date_from)
        if date_to:
            readings = readings.filter(timestamp__date__lte=date_to)

        readings = readings.order_by('-timestamp')

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

@login_required
@csrf_protect
@require_POST
def delete_prediction(request, prediction_id):
    try:
        prediction = get_object_or_404(PredictionResult, id=prediction_id, farm__user=request.user)
        prediction.delete()
        messages.success(request, 'Prediction deleted successfully!')
    except Exception as e:
        messages.error(request, f'Error deleting prediction: {str(e)}')
    return redirect('farmer:predictions')

@login_required
def download_prediction_pdf(request, prediction_id):
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.units import inch, cm
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.enums import TA_CENTER, TA_RIGHT
        from io import BytesIO
        import os
        
        # Get the prediction
        prediction = get_object_or_404(PredictionResult, id=prediction_id, farm__user=request.user)
        farmer = prediction.farm.user
        
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
        
        # Add logo if exists (you'll need to add your logo file)
        logo_path = os.path.join(settings.STATIC_ROOT, 'images', 'logo.png')
        if os.path.exists(logo_path):
            logo = Image(logo_path)
            logo.drawHeight = 1.5*inch
            logo.drawWidth = 1.5*inch
            elements.append(logo)
            elements.append(Spacer(1, 20))
        
        # Title
        elements.append(Paragraph('Soil Sense - Prediction Report', styles['CustomTitle']))
        
        # Farmer Information Section
        elements.append(Paragraph('Farmer Information', styles['SubTitle']))
        
        farmer_info = [
            ['Name:', f"{farmer.first_name} {farmer.last_name}"],
            ['Email:', farmer.email],
            ['Farm Name:', prediction.farm.farm_name],
            ['Farm Location:', prediction.farm.location],
            ['Area Size:', f"{prediction.farm.area_size} hectares"],
            ['Soil Type:', prediction.farm.soil_type]
        ]
        
        farmer_table = Table(farmer_info, colWidths=[120, 350])
        farmer_table.setStyle(TableStyle([
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
        elements.append(farmer_table)
        elements.append(Spacer(1, 20))
        
        # Prediction Results Section
        elements.append(Paragraph('Prediction Results', styles['SubTitle']))
        
        # Create a styled box for the main predictions
        prediction_data = [
            ['Soil Moisture Prediction', 'Irrigation Recommendation'],
            [f"{prediction.soil_moisture_result:.2f}%", prediction.irrigation_result],
            ['Algorithm: ' + prediction.algorithm, 'Algorithm: ' + prediction.algorithm_irr]
        ]
        
        prediction_table = Table(prediction_data, colWidths=[235, 235])
        prediction_table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),  # Headers
            ('FONTSIZE', (0, 1), (-1, 1), 20),  # Main values
            ('FONTSIZE', (0, 2), (-1, 2), 9),   # Algorithm info
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.HexColor('#2c3e50')),  # Headers
            ('TEXTCOLOR', (0, 1), (-1, 1), colors.HexColor('#1a5f7a')),  # Main values
            ('TEXTCOLOR', (0, 2), (-1, 2), colors.HexColor('#7f8c8d')),  # Algorithm info
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#ecf0f1')),  # Header background
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
            ('ALIGN', (1, 1), (2, -2), 'CENTER'),  # Center align values and units
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
        
        # Get the value of the BytesIO buffer
        pdf = buffer.getvalue()
        buffer.close()
        
        # Generate response
        response = HttpResponse(content_type='application/pdf')
        response['Content-Disposition'] = f'attachment; filename="soil_sense_prediction_{prediction_id}.pdf"'
        response.write(pdf)
        
        return response
        
    except Exception as e:
        messages.error(request, f'Error generating PDF: {str(e)}')
        return redirect('farmer:predictions')
