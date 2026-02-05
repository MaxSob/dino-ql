
from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('api/run/<int:run_id>/', views.api_run_data, name='api_run_data'),
    path('api/episode/<int:episode_id>/', views.api_episode_replay, name='api_episode_replay'),
    path('about/', views.about, name='about'),
]
