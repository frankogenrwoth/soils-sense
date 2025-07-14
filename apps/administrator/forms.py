from django import forms
from django.contrib.auth import get_user_model


User = get_user_model()

class UserForm(forms.ModelForm):
    """User form for creating and updating user objects"""
    class Meta:
        model = User
        fields = "__all__"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        pass


