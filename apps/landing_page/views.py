from django.shortcuts import render
from .forms import ContactForm
from .models import ContactMessage
from django.contrib import messages
from django.urls import reverse_lazy
from django.core.mail import send_mail
from django.views.generic.edit import FormView
from django.views.generic import TemplateView

# ✅ Contact Page
class ContactPage(FormView):
    template_name = 'contact.html'
    form_class = ContactForm

    success_url = reverse_lazy('contact')  # No 'landing_page:' prefix


    def form_valid(self, form):
        # Save to database
        ContactMessage.objects.create(
            name=form.cleaned_data['name'],
            email=form.cleaned_data['email'],
            message=form.cleaned_data['message']
        )

        # Optional: send email
        send_mail(
            subject=f"Contact Form Submission from {form.cleaned_data['name']}",
            message=form.cleaned_data['message'],
            from_email=form.cleaned_data['email'],
            recipient_list=['garangayelj@gmail.com'],
            fail_silently=False,
        )
        self.extra_context = {'show_success': True}
        messages.success(self.request, "Thank you for contacting us. Your message has been received.")
        return super().form_valid(form)

# ✅ About Page
class AboutPage(TemplateView):
    template_name = 'about.html'

# ✅ Landing Page
class LandingPage(TemplateView):
    template_name = 'landing_page.html'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['title'] = "Welcome to SoilSense"
        context['description'] = "Your partner in sustainable agriculture."
        return context
    