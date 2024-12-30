from django.urls import path
from . import views


urlpatterns = [
    path("", views.firstMethod, name="first"),
    path("update/", views.update_parameter, name="update_parameter"),
]
