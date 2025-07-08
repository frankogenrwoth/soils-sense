from django.shortcuts import render
from .forms import ContactForm

# Create your views here.

# Landing page view
def landing_page(request):
    """
    Render the landing page.
    """
    return render(request, 'landing_page.html')
# About page view
def about_page(request):
    """
    Render the about page.
    """
    return render(request, 'about.html')

# Contact page view
def contact_page(request):
    """
    Render the contact page.
    """
    
    return render(request, 'contact.html')
