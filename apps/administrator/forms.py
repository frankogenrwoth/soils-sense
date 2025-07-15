from django import forms
from django.contrib.auth import get_user_model


User = get_user_model()


class UserForm(forms.ModelForm):
    """User form for creating and updating user objects"""

    password = forms.CharField(
        widget=forms.PasswordInput(),
        required=False,
        help_text="Leave blank to keep current password when updating",
    )

    class Meta:
        model = User
        fields = [
            "username",
            "first_name",
            "last_name",
            "email",
            "role",
            "password",
            "image",
            "phone_number",
        ]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for field_name, field in self.fields.items():
            if field_name == "password":
                field.widget.attrs["class"] = (
                    "w-full border border-gray-300 rounded px-3 py-2 focus:outline-none focus:ring-2 focus:ring-green-200"
                )
                field.widget.attrs["placeholder"] = "Enter password"
            elif field_name == "image":
                field.widget.attrs["class"] = (
                    "w-full border border-gray-300 rounded px-3 py-2 focus:outline-none focus:ring-2 focus:ring-green-200"
                )
                field.required = False
            else:
                field.widget.attrs["class"] = (
                    "w-full border border-gray-300 rounded px-3 py-2 focus:outline-none focus:ring-2 focus:ring-green-200"
                )


    def clean_password(self):
        password = self.cleaned_data.get("password")
        
        if self.instance.pk and not password:
            return password
        
        if not password:
            raise forms.ValidationError("Password is required for new users")
        return password
