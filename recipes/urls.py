# recipes/urls.py
from django.urls import path
from .views import RecipePredictionView

urlpatterns = [
    path('predict/', RecipePredictionView.as_view(), name='recipe-predict'),
]
