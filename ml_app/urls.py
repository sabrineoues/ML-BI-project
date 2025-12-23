from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('dashboard/', views.dashboard, name='dashboard'),
    path('knn/', views.knn_view, name='knn'),
    path('xgboost/', views.xgb_view, name='xgboost'),
    path('regression/', views.regression_view, name='regression'),
    path('kmeans/', views.kmeans_view, name='kmeans'),
]
