# api/urls.py
from django.urls import path
from .views import VerifyAPIView

urlpatterns = [
    path('verify/', VerifyAPIView.as_view(), name='verify-claim'),
]
