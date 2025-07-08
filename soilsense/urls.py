from django.contrib import admin
from django.urls import path, include
from django.shortcuts import redirect
from django.contrib.auth import views as auth_views

def redirect_to_farmer(request):
    """Redirect root URL to farmer dashboard"""
    return redirect('farmer:dashboard')

urlpatterns = [
    path("admin/", admin.site.urls),
    path("authentication/", include("authentication.urls")),
    path("farmer/", include("apps.farmer.urls")),
	  path("", include("apps.landing_page.urls")),
    path("technician/", include("apps.technician.urls")),
	  path("landing_page/", include("apps.landing_page.urls")),
]
