from django.shortcuts import render, redirect
from django.http import JsonResponse, FileResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from authentication.models import Role
from django.core.exceptions import PermissionDenied
from .models import SoilDataFile
from django.db.models import Q
from datetime import datetime

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
        return redirect('farmer:profile')
    
    return render(request, 'farmer/profile.html')

def manage_files(request):
    """File management view"""
    return render(request, 'farmer/manage_files.html')

def view_history(request):
    """File history view with filtering"""
    # Get filter parameters
    file_type = request.GET.get('file_type', '')
    date_str = request.GET.get('date', '')

    # Query files for the current user
    files = SoilDataFile.objects.filter(user=request.user)

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

@login_required
def download_file(request):
    """Download a file"""
    file_id = request.GET.get('file_id')
    if not file_id:
        return JsonResponse({'error': 'No file ID provided'}, status=400)
    
    try:
        soil_file = SoilDataFile.objects.get(id=file_id, user=request.user)
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
@login_required
def delete_file(request):
    """Delete a file"""
    file_id = request.POST.get('file_id')
    if not file_id:
        return JsonResponse({'error': 'No file ID provided'}, status=400)
    
    try:
        soil_file = SoilDataFile.objects.get(id=file_id, user=request.user)
        soil_file.file.delete()  # Delete the actual file
        soil_file.delete()  # Delete the database record
        return JsonResponse({'success': True})
    except SoilDataFile.DoesNotExist:
        return JsonResponse({'error': 'File not found'}, status=404)
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)
