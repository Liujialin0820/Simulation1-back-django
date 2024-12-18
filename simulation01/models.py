from django.db import models


# Create your models here.
class Parameters(models.Model):
    data = models.JSONField()  # 用于存储 JSON 数据
    name = models.CharField(max_length=256)
    user = models.CharField(max_length=256, blank=True)
