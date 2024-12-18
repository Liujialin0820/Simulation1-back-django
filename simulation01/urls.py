from django.urls import path
from . import views


urlpatterns = [
    path("", views.firstMethod, name="first"),
    path("<int:pk>/update/", views.update_parameter, name="update_parameter"),
]
