from django.shortcuts import render, redirect
from django.http import JsonResponse, FileResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from authentication.models import Role
from django.core.exceptions import PermissionDenied

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
    context = {
        'total_files': 0,
        'csv_files': 0,
        'json_files': 0,
        'uploaded_files': []
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
        
        # TODO: Save file and process it
        
        return JsonResponse({
            'success': True,
            'file_name': file_name,
            'file_size': f"{file.size / 1024 / 1024:.2f}MB",
            'file_type': 'CSV' if file_name.endswith('.csv') else 'JSON'
        })
        
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)

def download_file(request):
    """Download a file"""
    file_id = request.GET.get('file_id')
    if not file_id:
        return JsonResponse({'error': 'No file ID provided'}, status=400)
    
    # TODO: Implement file download logic
    return JsonResponse({'error': 'Not implemented yet'}, status=501)

@csrf_exempt
@require_http_methods(["POST"])
def delete_file(request):
    """Delete a file"""
    file_id = request.POST.get('file_id')
    if not file_id:
        return JsonResponse({'error': 'No file ID provided'}, status=400)
    
    # TODO: Implement file deletion logic
    return JsonResponse({'error': 'Not implemented yet'}, status=501)
