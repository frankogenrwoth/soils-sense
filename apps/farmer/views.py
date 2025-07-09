from django.shortcuts import render, redirect
from django.http import JsonResponse, FileResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from authentication.models import Role, User
from django.core.exceptions import PermissionDenied
from .models import SoilDataFile, Farm, Crop
from django.db.models import Q
from datetime import datetime
import json

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

def dashboard(request):
    """Main farmer dashboard view"""
    context = {
        'current_moisture': 68,
        'temperature': 24,
        'humidity': 75,
        'crop_health': 'Good'
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

def farm_management(request):
    """Farm management view"""
    # Get test user for development
    test_user, created = User.objects.get_or_create(
        username='testuser',
        defaults={
            'email': 'test@example.com',
            'role': Role.FARMER
        }
    )
    
    farms = Farm.objects.filter(user=test_user)
    crops = Crop.objects.filter(farm__user=test_user)
    
    context = {
        'farms': farms,
        'crops': crops
    }
    return render(request, 'farmer/farm_management.html', context)

@csrf_exempt
@require_http_methods(["POST"])
def add_farm(request):
    """Add new farm location"""
    try:
        # Get test user for development
        test_user, created = User.objects.get_or_create(
            username='testuser',
            defaults={
                'email': 'test@example.com',
                'role': Role.FARMER
            }
        )
        
        # Create farm
        farm = Farm.objects.create(
            user=test_user,
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
def add_crop(request):
    """Add new crop"""
    try:
        # Get the farm
        farm = Farm.objects.get(id=request.POST.get('farm'))
        
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
        return JsonResponse({'success': False, 'error': 'Farm not found'})
    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)})

def manage_files(request):
    """File management view"""
    return render(request, 'farmer/manage_files.html')

def view_history(request):
    """File history view with filtering"""
    # Get filter parameters
    file_type = request.GET.get('file_type', '')
    date_str = request.GET.get('date', '')

    # Get all files for testing purposes
    files = SoilDataFile.objects.all()

    # Apply filters if provided
    if file_type:
        files = files.filter(file_type=file_type)
    
    if date_str:
        try:
            filter_date = datetime.strptime(date_str, '%Y-%m-%d').date()
            files = files.filter(upload_date__date=filter_date)
        except ValueError:
            pass  # Invalid date format, ignore filter

    # Calculate totals
    total_files = files.count()
    csv_files = files.filter(file_type='csv').count()
    json_files = files.filter(file_type='json').count()

    context = {
        'total_files': total_files,
        'csv_files': csv_files,
        'json_files': json_files,
        'uploaded_files': files
    }
    return render(request, 'farmer/view_history.html', context)

def analytics(request):
    """Analytics dashboard view"""
    return render(request, 'farmer/analytics.html')

def recommendations(request):
    """Smart recommendations view"""
    return render(request, 'farmer/recommendations.html')

@csrf_exempt
@require_http_methods(["POST"])
def upload_file(request):
    """Handle file upload with validation"""
    try:
        if 'file' not in request.FILES:
            return JsonResponse({'error': 'No file provided'}, status=400)
        
        file = request.FILES['file']
        file_name = file.name.lower()
        
        # Validate file
        if not (file_name.endswith('.csv') or file_name.endswith('.json')):
            return JsonResponse({'error': 'Only CSV and JSON files are allowed'}, status=400)
        if file.size > 10 * 1024 * 1024:  # 10MB limit
            return JsonResponse({'error': 'File size must be less than 10MB'}, status=400)
        
        # Create file record
        file_type = 'csv' if file_name.endswith('.csv') else 'json'
        soil_data_file = SoilDataFile.objects.create(
            file=file,
            file_name=file_name,
            file_type=file_type,
            file_size=file.size,
            user=request.user
        )
        
        return JsonResponse({
            'success': True,
            'file_name': file_name,
            'file_size': soil_data_file.get_file_size_mb(),
            'file_type': file_type.upper()
        })
        
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)

def download_file(request):
    """Download a file"""
    file_id = request.GET.get('file_id')
    if not file_id:
        return JsonResponse({'error': 'No file ID provided'}, status=400)
    
    try:
        soil_file = SoilDataFile.objects.get(id=file_id)
        response = FileResponse(soil_file.file)
        response['Content-Type'] = 'text/csv' if soil_file.file_type == 'csv' else 'application/json'
        response['Content-Disposition'] = f'attachment; filename="{soil_file.file_name}"'
        return response
    except SoilDataFile.DoesNotExist:
        return JsonResponse({'error': 'File not found'}, status=404)
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)

@csrf_exempt
@require_http_methods(["POST"])
def delete_file(request):
    """Delete a file"""
    file_id = request.POST.get('file_id')
    if not file_id:
        return JsonResponse({'error': 'No file ID provided'}, status=400)
    
    try:
        soil_file = SoilDataFile.objects.get(id=file_id)
        soil_file.file.delete()  # Delete the actual file
        soil_file.delete()  # Delete the database record
        return JsonResponse({'success': True})
    except SoilDataFile.DoesNotExist:
        return JsonResponse({'error': 'File not found'}, status=404)
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)
