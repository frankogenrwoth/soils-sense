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
    WeatherData, IrrigationEvent, Alert, PredictionResult, Notification
)
from django.db.models import Avg
from datetime import datetime, timedelta
import json
import csv
import pandas as pd
import traceback
from ml.predictor import SoilMoisturePredictor, IrrigationRecommender
from ml.config import REGRESSION_ALGORITHMS, CLASSIFICATION_ALGORITHMS, DEFAULT_ALGORITHMS
from django.utils import timezone
import os
from django.conf import settings
from ml import MLEngine
import traceback
from apps.administrator.models import Model

from .utils import farmer_role_required


ml_engine = MLEngine()

def farmer_required(view_func):
    def wrapper(request, *args, **kwargs):
        if not request.user.is_authenticated:
            return redirect('authentication:login')
        if request.user.role != Role.FARMER:
            raise PermissionDenied("You must be a farmer to access this page.")
        return view_func(request, *args, **kwargs)
    return wrapper

@farmer_role_required
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

    # Get the last 7 days of readings
    seven_days_ago = timezone.now() - timedelta(days=7)
    moisture_history = SoilMoistureReading.objects.filter(
        farm=selected_farm,
        timestamp__gte=seven_days_ago,
    ).order_by('timestamp')

    # Initialize lists for chart data
    dates = []
    predicted_values = []

    # Initialize the soil moisture predictor
    predictor = SoilMoisturePredictor()

    # Process readings and collect data for charts
    for reading in moisture_history:
        try:
            dates.append(reading.timestamp.strftime('%Y-%m-%d %H:%M'))
            
            # Make prediction using gradient boosting
            prediction = predictor.predict_moisture(
                sensor_id=reading.sensor_id,
                location=selected_farm.location,
                temperature_celsius=reading.temperature_celsius,
                humidity_percent=reading.humidity_percent,
                battery_voltage=reading.battery_voltage,
                status=reading.status,
                irrigation_action=reading.irrigation_action,
                timestamp=reading.timestamp,
                algorithm='random_forest'
            )
            
            if prediction['success']:
                predicted_values.append(float(prediction['predicted_value']))
            else:
                predicted_values.append(None)

        except (ValueError, TypeError) as e:
            print(f"Error processing reading: {e}")
            continue

    # Get latest readings for display
    latest_reading = moisture_history.first()
    try:
        current_moisture = round(float(latest_reading.soil_moisture_percent), 1) if latest_reading and latest_reading.soil_moisture_percent is not None else 0
        temperature = round(float(latest_reading.temperature_celsius), 1) if latest_reading and latest_reading.temperature_celsius is not None else 0
        humidity = round(float(latest_reading.humidity_percent), 1) if latest_reading and latest_reading.humidity_percent is not None else 0
    except (ValueError, TypeError, AttributeError):
        current_moisture = 0
        temperature = 0
        humidity = 0

    # Get alerts
    critical_alerts = Alert.objects.filter(farm=selected_farm, is_read=False, severity='critical')
    unread_moisture_alerts = Alert.objects.filter(
        farm=selected_farm,
        is_read=False,
        alert_type__in=['low_moisture', 'high_moisture']
    ).order_by('-timestamp')

    context = {
        'farms': farms,
        'selected_farm': selected_farm,
        'current_moisture': current_moisture,
        'temperature': temperature,
        'humidity': humidity,
        'dates': json.dumps(dates),
        'predicted_values': json.dumps(predicted_values),
        'critical_alerts': critical_alerts,
        'unread_moisture_alerts': unread_moisture_alerts,
    }
    
    return render(request, 'farmer/dashboard.html', context)

@farmer_role_required
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

@farmer_role_required
def farm_management(request):
    farms = Farm.objects.filter(user=request.user)
    crops = Crop.objects.filter(farm__user=request.user)
    
    context = {
        'farms': farms,
        'crops': crops
    }
    return render(request, 'farmer/farm_management.html', context)

@require_http_methods(["POST"])
@farmer_role_required
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
@farmer_role_required
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

@farmer_role_required
def analytics(request):
    farms = Farm.objects.filter(user=request.user)
    
    # Ensure these are always defined
    current_predicted_moisture = None
    current_actual_moisture = None
    latest_reading = None
    
    # Get selected farm and time range
    farm_id = request.GET.get('farm_id')
    time_range = int(request.GET.get('time_range', 7))  # Default to 7 days
    algorithm = request.GET.get('algorithm', DEFAULT_ALGORITHMS["soil_moisture_predictor"])
    
    if farm_id:
        selected_farm = get_object_or_404(Farm, id=farm_id, user=request.user)
    else:
        selected_farm = farms.first()
    
    if not selected_farm:
        messages.error(request, 'No farms found. Please add a farm first.')
        return redirect('farmer:dashboard')

    # Calculate date range
    end_date = timezone.now()
    start_date = end_date - timedelta(days=time_range)
    
    # Get soil moisture readings
    readings = SoilMoistureReading.objects.filter(
        farm=selected_farm,
        timestamp__range=(start_date, end_date)
    ).order_by('timestamp')
    
    # Initialize predictor
    predictor = SoilMoisturePredictor()
    
    # Prepare data for charts
    dates = []
    actual_values = []
    predicted_values = []
    correlation_data = []
    temperature_values = []  # New
    humidity_values = []     # New
    
    # Process readings and generate predictions (historical)
    for reading in readings:
        try:
            # Get actual reading
            if reading.soil_moisture_percent is not None:
                actual_value = float(reading.soil_moisture_percent)
                actual_values.append(actual_value)
            else:
                actual_values.append(None)
                
            # Generate prediction
            prediction_result = predictor.predict_moisture(
                sensor_id=reading.sensor_id,
                location=selected_farm.location,
                temperature_celsius=float(reading.temperature_celsius),
                humidity_percent=float(reading.humidity_percent),
                battery_voltage=float(reading.battery_voltage),
                status=reading.status,
                irrigation_action="None",
                timestamp=reading.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                algorithm=algorithm
            )
            
            if prediction_result.get('success'):
                predicted_value = float(prediction_result['predicted_value'])
                predicted_values.append(predicted_value)
            else:
                predicted_values.append(None)
                
            # Add date
            dates.append(reading.timestamp.strftime('%Y-%m-%d'))
            # Add temp/humidity
            temperature_values.append(float(reading.temperature_celsius))
            humidity_values.append(float(reading.humidity_percent))
            
            # Add correlation data point if both values exist
            if reading.temperature_celsius is not None and prediction_result.get('success'):
                correlation_data.append({
                    'x': float(reading.temperature_celsius),
                    'y': predicted_value
                })
                
        except Exception as e:
            print(f"Error processing reading: {str(e)}")
            continue
    
    # Calculate statistics from predictions
    valid_predictions = [v for v in predicted_values if v is not None]
    if valid_predictions:
        avg_moisture = sum(valid_predictions) / len(valid_predictions)
        max_moisture = max(valid_predictions)
        min_moisture = min(valid_predictions)
        
        # Get dates for max and min moisture
        max_moisture_date = None
        min_moisture_date = None
        for i, value in enumerate(predicted_values):
            if value == max_moisture:
                max_moisture_date = datetime.strptime(dates[i], '%Y-%m-%d')
            if value == min_moisture:
                min_moisture_date = datetime.strptime(dates[i], '%Y-%m-%d')
    else:
        avg_moisture = 0
        max_moisture = 0
        min_moisture = 0
        max_moisture_date = None
        min_moisture_date = None
    
    context = {
        'farms': farms,
        'selected_farm': selected_farm,
        'time_range': time_range,
        'selected_algorithm': algorithm,
        'soil_algorithms': REGRESSION_ALGORITHMS,
        # Statistics using predicted values
        'avg_moisture': round(float(avg_moisture), 2),
        'max_moisture': round(float(max_moisture), 2),
        'min_moisture': round(float(min_moisture), 2),
        'max_moisture_date': max_moisture_date,
        'min_moisture_date': min_moisture_date,
        # Chart data
        'dates': json.dumps(dates),
        'actual_values': json.dumps(actual_values),
        'predicted_values': json.dumps(predicted_values),
        'correlation_data': json.dumps(correlation_data),
        'temperature_values': json.dumps(temperature_values),  # New
        'humidity_values': json.dumps(humidity_values),        # New
        # Hybrid: current prediction
        'current_predicted_moisture': current_predicted_moisture,
        'current_actual_moisture': current_actual_moisture,
        'latest_reading': latest_reading,
    }
    
    return render(request, 'farmer/analytics.html', context)


@farmer_role_required
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
                    moisture_status = 'Low Soil Moisture'
                elif current_moisture > 70:
                    moisture_status = 'High Soil Moisture'
                else:
                    moisture_status = 'Optimal Soil Moisture'
        except Exception as e:
            print(f"Prediction error: {str(e)}")
            # Fallback to raw reading if prediction fails
            current_moisture = round(float(latest_reading.soil_moisture_percent), 1)

    # Get irrigation recommendation based on predicted moisture
    irrigation_recommendation = "No irrigation needed at this time."
    water_amount = None
    if moisture_status == 'Low Soil Moisture':
        irrigation_recommendation = "Immediate irrigation recommended"
        # Calculate water amount based on area and moisture deficit
        area_size = float(selected_farm.area_size)
        moisture_deficit = 50 - current_moisture  # Assuming 50% is optimal
        water_amount = round(area_size * moisture_deficit * 100)  # Simple calculation, adjust as needed
    elif moisture_status == 'High Soil Moisture':
        irrigation_recommendation = "Hold irrigation until soil moisture decreases"

    # Get farm's crops and prepare recommendations
    crops = []
    for crop in Crop.objects.filter(farm=selected_farm):
        # Use the predicted moisture value for crop recommendations
        moisture_percentage = current_moisture
        if current_moisture < 30:
            moisture_status = 'Low Soil Moisture'
            status_class = 'text-red-600'
        elif current_moisture < 50:
            moisture_status = 'Warning Soil Moisture'
            status_class = 'text-yellow-600'
        else:
            moisture_status = 'Good Soil Moisture'
            status_class = 'text-green-600'

        # Generate recommended actions based on predicted conditions
        recommended_actions = []
        if moisture_status == 'Critical Soil Moisture':
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

@farmer_role_required
def predictions(request):
    farms = Farm.objects.filter(user=request.user)
    
    # Get all predictions for the user's farms
    predictions = PredictionResult.objects.filter(
        farm__in=farms
    ).select_related('farm').order_by('-created_at')
    
    if request.method == 'POST':
        try:
            print(request.POST)
            farm_id = request.POST.get('farm')
            farm = get_object_or_404(Farm, id=farm_id, user=request.user)
            
            # Check if this is a CSV upload
            if request.POST.get('upload_type') == 'csv':
                if 'file' not in request.FILES:
                    messages.error(request, 'No file was uploaded.')
                    return redirect('farmer:predictions')
                
                csv_file = request.FILES['file']
                if not csv_file.name.endswith('.csv'):
                    messages.error(request, 'Please upload a CSV file.')
                    return redirect('farmer:predictions')
                
                try:
                    # Read CSV file

                    print(request.POST)

                    df = pd.read_csv(csv_file)
                    required_columns = ['temperature_celsius', 'humidity_percent', 'battery_voltage', 'status']
                    
                    # Check for required columns
                    missing_columns = [col for col in required_columns if col not in df.columns]
                    if missing_columns:
                        messages.error(request, f'Missing required columns: {", ".join(missing_columns)}. Please use our template.')
                        return redirect('farmer:predictions')
                    
                    # Initialize predictors
                    soil_predictor = SoilMoisturePredictor()
                    irrigation_recommender = IrrigationRecommender()
                    
                    # Get algorithms from form
                    algorithm = request.POST.get('algorithm')
                    algorithm_irr = request.POST.get('algorithm_irr')
                    
                    # Process each row
                    predictions_created = 0
                    for _, row in df.iterrows():
                        try:
                            # Make soil moisture prediction
                            soil_moisture_result = soil_predictor.predict_moisture(
                                sensor_id=str(row.get('sensor_id', 1)),
                                location=farm.location,
                                temperature_celsius=float(row['temperature_celsius']),
                                humidity_percent=float(row['humidity_percent']),
                                battery_voltage=float(row['battery_voltage']),
                                status=str(row.get('status', 'Normal')),
                                irrigation_action=str(row.get('irrigation_action', 'None')),
                                timestamp=row.get('timestamp', timezone.now().strftime('%Y-%m-%d %H:%M:%S')),
                                algorithm=algorithm
                            )
                            
                            # Get the predicted soil moisture value
                            soil_moisture_value = soil_moisture_result['predicted_value'] if isinstance(soil_moisture_result, dict) else soil_moisture_result
                            
                            # Get irrigation recommendation
                            irrigation_result = irrigation_recommender.recommend_irrigation(
                                soil_moisture_percent=soil_moisture_value,
                                temperature_celsius=float(row['temperature_celsius']),
                                humidity_percent=float(row['humidity_percent']),
                                battery_voltage=float(row['battery_voltage']),
                                status=str(row.get('status', 'Normal')),
                                timestamp=row.get('timestamp', timezone.now().strftime('%Y-%m-%d %H:%M:%S')),
                                algorithm=algorithm_irr
                            )
                            
                            # Get the recommendation value
                            irrigation_recommendation = irrigation_result['predicted_value'] if isinstance(irrigation_result, dict) else irrigation_result
                            
                            # Save prediction
                            PredictionResult.objects.create(
                                farm=farm,
                                location=farm.location,
                                temperature=float(row['temperature_celsius']),
                                humidity=float(row['humidity_percent']),
                                battery_voltage=float(row['battery_voltage']),
                                soil_moisture_result=soil_moisture_value,
                                irrigation_result=irrigation_recommendation,
                                algorithm=algorithm,
                                algorithm_irr=algorithm_irr,
                            )
                            predictions_created += 1
                            
                        except Exception as e:
                            print(f"Error processing row: {str(e)}")
                            continue
                    
                    if predictions_created > 0:
                        messages.success(request, f'Successfully created {predictions_created} predictions from CSV data!')
                    else:
                        messages.warning(request, 'No valid predictions could be created from the CSV data.')
                    
                except pd.errors.EmptyDataError:
                    messages.error(request, 'The uploaded CSV file is empty.')
                except pd.errors.ParserError:
                    messages.error(request, 'Error parsing CSV file. Please make sure it is properly formatted.')
                except Exception as e:
                    messages.error(request, f'Error processing CSV file: {str(e)}')
                
                return redirect('farmer:predictions')
            
            # Handle manual input
            else:
                # Get form data
                location = request.POST.get('location')
                temperature = float(request.POST.get('temperature'))
                humidity = float(request.POST.get('humidity'))
                battery_voltage = float(request.POST.get('battery_voltage'))
                algorithm = request.POST.get('algorithm')
                algorithm_irr = request.POST.get('algorithm_irr')

                def clean_ml_algo(value):
                    model_name = None
                    algorithm = None
                    version = None

                    raw = value.split("_")

                    if value.find("version") == -1:
                        version = None
                    else:
                        version = raw[-1]
                        raw.pop()
                        raw.pop()

                    algorithm = "_".join(raw[-2:])
                    model_name = "_".join(raw[:-2])

                    return model_name, algorithm, version
                
                model_name, algorithm, version = clean_ml_algo(algorithm)
                model_name_irr, algorithm_irr, version_irr = clean_ml_algo(algorithm_irr)

                
                # Make predictions
                soil_predictor = SoilMoisturePredictor()
                irrigation_recommender = IrrigationRecommender()
                
                # Get current timestamp
                current_time = timezone.now()
                
                # Predict soil moisture


                soil_moisture_result = ml_engine.predict(model_type=model_name, algorithm=algorithm, version=version, data={
                    "temperature_celsius": temperature,
                    "humidity_percent": humidity,
                    "battery_voltage": battery_voltage,
                    "status": "Normal",
                    "irrigation_action": "None",
                    "timestamp": current_time,
                })
                print(soil_moisture_result)
                # Get the predicted value
                soil_moisture_value = soil_moisture_result['predicted_value'] if isinstance(soil_moisture_result, dict) else soil_moisture_result
                
                # Get irrigation recommendation
                irrigation_result = ml_engine.predict(model_type=model_name_irr, algorithm=algorithm_irr, version=version_irr, data={
                    "soil_moisture_percent": soil_moisture_value,
                    "temperature_celsius": temperature,
                    "humidity_percent": humidity,
                    "battery_voltage": battery_voltage,
                    "status": "Normal",
                    "timestamp": current_time,
                })
                print(irrigation_result)
                
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
                    algorithm_irr=algorithm_irr,
                )

                messages.success(request, 'Prediction created successfully!')
            
            return redirect('farmer:predictions')
            
        except Exception as e:
            print(traceback.print_exc())
            messages.error(request, f'Error making prediction: {str(e)}')
            return redirect('farmer:predictions')
    modal_list = ml_engine.get_available_models()
    
    custom_activated_models = Model.objects.filter(is_active=False)
    non_activated =[mod.get_model_version() for mod in custom_activated_models]

    print(non_activated)
    print(modal_list)

    print([(modal_list[0].find(odd) != -1) for odd in non_activated if modal_list[0].find(odd) != -1])
    print([(modal_list[3].find(odd) != -1) for odd in non_activated if modal_list[3].find(odd) != -1])



    

    modal_list = [
        algo 
        for algo in modal_list 
        if len([(algo.find(odd)!= -1) for odd in non_activated if algo.find(odd) != -1]) == 0
    ]
    print(modal_list)
    
    soil_moisture_modal_list = [algo for algo in modal_list if algo.startswith("soil_moisture_predictor")]
    irrigation_modal_list = [algo for algo in modal_list if algo.startswith("irrigation")]

    

    print(modal_list)
    
    context = {
        'farms': farms,
        'predictions': predictions,
        'soil_algorithms': REGRESSION_ALGORITHMS,
        'irrigation_algorithms': CLASSIFICATION_ALGORITHMS,
        'default_soil_algorithm': "soil_moisture_predictor_random_forest",
        'default_irrigation_algorithm': "irrigation_recommendation_random_forest",
        'soil_moisture_modal_list': soil_moisture_modal_list,
        'irrigation_modal_list': irrigation_modal_list,
    }
    return render(request, 'farmer/predictions.html', context)

@farmer_role_required
def download_predictions_csv(request):
    import csv
    from django.http import HttpResponse
    from .models import PredictionResult

    # Get all predictions for the user's farms
    farms = Farm.objects.filter(user=request.user)
    predictions = PredictionResult.objects.filter(farm__in=farms).select_related('farm').order_by('-created_at')

    # Create the HttpResponse object with CSV header
    response = HttpResponse(content_type='text/csv')
    response['Content-Disposition'] = 'attachment; filename="soil_sense_predictions.csv"'

    writer = csv.writer(response)
    # Write header
    writer.writerow([
        'Farm Name', 'Location', 'Date & Time', 'Temperature (°C)', 'Humidity (%)', 'Battery Voltage (V)',
        'Soil Moisture Prediction (%)', 'Irrigation Recommendation', 'Soil Algorithm', 'Irrigation Algorithm'
    ])
    # Write data rows
    for p in predictions:
        writer.writerow([
            p.farm.farm_name,
            p.location,
            p.created_at.strftime('%Y-%m-%d %H:%M'),
            p.temperature,
            p.humidity,
            p.battery_voltage,
            p.soil_moisture_result,
            p.irrigation_result,
            p.algorithm,
            p.algorithm_irr
        ])
    return response

@farmer_role_required
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

@farmer_role_required
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

@farmer_role_required
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

@farmer_role_required
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

@farmer_role_required
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

@farmer_role_required
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

@farmer_role_required
def download_csv_template(request):
    """Download empty CSV template for data upload"""
    response = HttpResponse(content_type='text/csv')
    response['Content-Disposition'] = 'attachment; filename="soil_moisture_template.csv"'
    
    writer = csv.writer(response)
    writer.writerow(['sensor_id', 'location', 'temperature_celsius', 'humidity_percent', 'timestamp', 'status', 'battery_voltage'])
    writer.writerow(['SENSOR_1', 'Farm A', '25.3', '65.2', '2024-03-21 14:30:00', 'Normal', '3.62'])
    
    return response

@farmer_role_required
def download_sensor_data(request):
    """Download sensor readings as CSV"""
    try:
        # Get farm_id from query parameters
        farm_id = request.GET.get('farm_id')
        if not farm_id:
            messages.error(request, 'Please select a farm first')
            return redirect('farmer:soil_data_management')

        # Verify farm ownership
        farm = get_object_or_404(Farm, id=farm_id, user=request.user)
        
        # Get timeframe from query params (default to last 24 hours)
        hours = int(request.GET.get('hours', 24))
        end_time = timezone.now()
        start_time = end_time - timedelta(hours=hours)
        
        # Get readings for the specified farm within the time range
        readings = SoilMoistureReading.objects.filter(
            farm=farm,
            timestamp__gte=start_time,
            timestamp__lte=end_time
        ).order_by('timestamp')

        # Create the HttpResponse object with CSV header
        response = HttpResponse(content_type='text/csv')
        response['Content-Disposition'] = f'attachment; filename="{farm.farm_name}_sensor_data_{timezone.now().strftime("%Y%m%d_%H%M%S")}.csv"'

        # Create CSV writer
        writer = csv.writer(response)
        
        # Write header
        writer.writerow([
            'Timestamp',
            'Location', 
            'Temperature (°C)', 
            'Humidity (%)', 
            'Soil Moisture (%)',
            'Battery Voltage (V)',
            'Status',
            'Current Action'
        ])

        # Write data
        for reading in readings:
            writer.writerow([
                reading.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                farm.location,
                reading.temperature_celsius,
                reading.humidity_percent,
                reading.soil_moisture_percent,
                reading.battery_voltage,
                reading.status,
                reading.irrigation_action
            ])

        return response
        
    except ValueError as e:
        messages.error(request, f'Invalid parameters: {str(e)}')
        return redirect('farmer:soil_data_management')
    except Exception as e:
        messages.error(request, f'Error generating CSV: {str(e)}')
        return redirect('farmer:soil_data_management')

@farmer_role_required
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
                required_columns = ['sensor_id', 'location', 'temperature_celsius', 'humidity_percent', 'timestamp', 'status', 'battery_voltage']
                
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
                        temperature = float(row['temperature_celsius'])
                        humidity = float(row['humidity_percent'])
                        battery = float(row['battery_voltage'])
                        
                        # Calculate soil moisture as in manual entry
                        soil_moisture = (humidity * 0.7) + (30 - temperature * 0.3)
                        soil_moisture = max(0, min(100, soil_moisture))
                        
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

@farmer_role_required
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

@farmer_role_required
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

@farmer_role_required
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
        
        # Get farmer name or fallback to username
        farmer_name = farmer.username
        if farmer.first_name and farmer.last_name:
            farmer_name = f"{farmer.first_name} {farmer.last_name}"
        
        farmer_info = [
            ['Name:', farmer_name],
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

@farmer_role_required
def notifications(request):
    notifications = Notification.objects.filter(user=request.user).order_by('-created_at')
    return render(request, 'farmer/notifications.html', {
        'notifications': notifications
    })

@farmer_role_required
def mark_notification_read(request, notification_id):
    notification = get_object_or_404(Notification, id=notification_id, user=request.user)
    notification.is_read = True
    notification.save()
    return redirect('farmer:notifications')

@farmer_role_required
def get_unread_count(request):
    count = Notification.objects.filter(user=request.user, is_read=False).count()
    return JsonResponse({'count': count})

@farmer_role_required
@require_http_methods(["POST", "GET"])
def delete_notification(request, notification_id):
    try:
        notification = get_object_or_404(Notification, id=notification_id, user=request.user)
        notification.delete()
        messages.success(request, 'Notification deleted successfully')
    except Exception as e:
        messages.error(request, f'Error deleting notification: {str(e)}')
    return redirect('farmer:notifications')

@farmer_role_required
def get_sensor_data(request, farm_id):
    """API endpoint to get sensor readings for a specific farm"""
    try:
        # Verify farm ownership
        farm = get_object_or_404(Farm, id=farm_id, user=request.user)
        
        # Get the timeframe from query params (default to 30 minutes)
        minutes = int(request.GET.get('minutes', 30))
        
        # Calculate the time range
        end_time = timezone.now()
        start_time = end_time - timedelta(minutes=minutes)
        
        # Get readings for the specified farm within the time range
        readings = SoilMoistureReading.objects.filter(
            farm=farm,
            timestamp__gte=start_time,
            timestamp__lte=end_time
        ).order_by('timestamp')
        
        # Format the readings
        data = {
            'success': True,
            'readings': [{
                'timestamp': reading.timestamp.isoformat(),
                'temperature': reading.temperature_celsius,
                'humidity': reading.humidity_percent,
                'soil_moisture': reading.soil_moisture_percent,
                'battery_voltage': reading.battery_voltage,
                'status': reading.status,
                'irrigation_action': reading.irrigation_action
            } for reading in readings]
        }
        
        return JsonResponse(data)
    except Farm.DoesNotExist:
        return JsonResponse({
            'success': False,
            'error': 'Farm not found or access denied'
        }, status=404)
    except ValueError as e:
        return JsonResponse({
            'success': False,
            'error': f'Invalid timeframe parameter: {str(e)}'
        }, status=400)
    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': str(e)
        }, status=500)



from django.views.decorators.http import require_GET

@require_GET
def get_sensor_data_api(request, farm_id):
    try:
        # Verify farm ownership
        farm = get_object_or_404(Farm, id=farm_id, user=request.user)

        # Get the timeframe from query params (default to 30 minutes)
        minutes = int(request.GET.get('minutes', 30))

        # Calculate the time range
        end_time = timezone.now()
        start_time = end_time - timedelta(minutes=minutes)

        # Get readings for the specified farm within the time range
        readings = SoilMoistureReading.objects.filter(
            farm=farm,
            timestamp__gte=start_time,
            timestamp__lte=end_time
        ).order_by('timestamp')

        # Format the readings
        data = {
            'success': True,
            'readings': [
                {
                    'timestamp': reading.timestamp.isoformat(),
                    'temperature': reading.temperature_celsius,
                    'humidity': reading.humidity_percent,
                    'soil_moisture': reading.soil_moisture_percent,
                    'battery_voltage': reading.battery_voltage,
                    'status': reading.status,
                    'irrigation_action': reading.irrigation_action
                }
                for reading in readings
            ]
        }

        print(data)

        return JsonResponse(data, content_type='application/json')
    except Farm.DoesNotExist:
        error = {'success': False, 'error': 'Farm not found or access denied'}
        return JsonResponse(error, status=404)
    except ValueError as e:
        error = {'success': False, 'error': f'Invalid timeframe parameter: {str(e)}'}
        return JsonResponse(error, status=400)
    except Exception as e:
        print(traceback.format_exc())
        error = {'success': False, 'error': str(e)}
        return JsonResponse(error, status=500)

