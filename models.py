from django.db import models

# Create your models here.

class MyModel(models.Model):
    field1 = models.CharField(max_length=450)
    field2 = models.IntegerField()
    # Add more fields as needed

