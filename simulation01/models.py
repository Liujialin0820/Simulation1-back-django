from django.db import models


# Create your models here.
class JModel(models.Model):
    jdata = models.JSONField()  # 用于存储 JSON 数据
    jname = models.CharField(max_length=256)
