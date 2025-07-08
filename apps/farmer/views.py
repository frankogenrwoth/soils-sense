from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods

def dashboard(request):
    """Main farmer dashboard view"""
    context = {
        'current_moisture': 68,
        'temperature': 24,
        'humidity': 75,
        'crop_health': 'Good'
    }
    return render(request, 'dashboard.html', context)

def view_history(request):
    """File history view with filtering"""
    context = {
        'total_files': 0,
        'csv_files': 0,
        'json_files': 0,
        'uploaded_files': []
    }
    return render(request, 'dashboard.html', context)

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
        
        return JsonResponse({
            'success': True,
            'file_name': file_name,
            'file_size': f"{file.size / 1024 / 1024:.2f}MB",
            'file_type': 'CSV' if file_name.endswith('.csv') else 'JSON'
        })
        
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)


