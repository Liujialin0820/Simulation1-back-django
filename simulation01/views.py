from django.http import JsonResponse
from .models import JModel


# Create your views here.
def firstMethod(request):
    data = JModel.objects.filter(jname="BasicBar").first()
    return JsonResponse(data.jdata)
