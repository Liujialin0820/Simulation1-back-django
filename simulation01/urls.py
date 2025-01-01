from django.urls import path
from . import views


urlpatterns = [
    path("", views.firstMethod, name="first"),
    path("getdata/", views.get_IRR_data, name="first"),
    path("parameters/", views.ParametersListCreateAPIView.as_view()),
]
