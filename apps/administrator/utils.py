from functools import wraps
from django.http import HttpResponseForbidden
from django.shortcuts import redirect
from authentication.models import Role
from django.contrib.auth.decorators import login_required


def admin_role_required(view_func):
    @login_required
    @wraps(view_func)
    def _wrapped_view(request, *args, **kwargs):
        user = request.user

        if user.is_authenticated and user.role != Role.ADMINISTRATOR:
            return HttpResponseForbidden(
                "You do not have permission to access this page."
            )

        return view_func(request, *args, **kwargs)

    return _wrapped_view
