from django.shortcuts import render
from django.http import JsonResponse
from .models import MyModel

def get_data(request):
    data = MyModel.objects.all().values()  # Retrieve data from the database
    return JsonResponse(list(data), safe=False)

# Create your views here.
