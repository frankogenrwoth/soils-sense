from django.shortcuts import render
from django.views import View
import logging

logger = logging.getLogger(__name__)


class UserManagementView(View):
    template_name = "administrator/user_management.html"
