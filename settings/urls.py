from django.urls import path, include

urlpatterns = [
    path("api/simulation01/", include("simulation01.urls")),
]
