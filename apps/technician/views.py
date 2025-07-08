from django.shortcuts import render

# Create your views here.

def dashboard(request):
    # Sample data for the dashboard - your colleagues will replace this with real data later
    context = {
        'active_farms': 24,
        'active_sensors': 156,
        'alerts': 7,
        'pending_reports': 3,
        'recent_alerts': [
            {
                'type': 'alert',
                'message': 'Moisture Level Critical',
                'farm_id': 'F123',
                'time': '2m ago'
            },
            {
                'type': 'warning',
                'message': 'pH Level Warning',
                'farm_id': 'F145',
                'time': '15m ago'
            },
            {
                'type': 'normal',
                'message': 'Sensor Maintenance Due',
                'farm_id': 'F156',
                'time': '1h ago'
            }
        ],
        'sensor_thresholds': [
            {
                'parameter': 'Soil Moisture',
                'min': '30%',
                'max': '70%',
                'status': 'normal'
            },
            {
                'parameter': 'pH Level',
                'min': '6.0',
                'max': '7.5',
                'status': 'warning'
            },
            {
                'parameter': 'Temperature',
                'min': '15°C',
                'max': '35°C',
                'status': 'normal'
            }
        ],
        'recent_reports': [
            {
                'title': 'Monthly Soil Analysis Report',
                'farm_id': 'F123',
                'description': 'Comprehensive soil health analysis',
                'generated_by': 'System',
                'time': '3 days ago'
            },
            {
                'title': 'Sensor Maintenance Log',
                'description': 'Routine check and calibration report',
                'generated_by': 'Tech Team',
                'time': '1 week ago'
            }
        ]
    }
    return render(request, 'technician_dashboard.html', context)
