from django.db import models
from django.contrib.auth.models import User


# Create your models here.
class Logs(models.Model):
    date = models.DateTimeField("识别时间", auto_now_add=True)
    image = models.TextField("图片base64", )
    result = models.TextField("返回值")
    user = models.ForeignKey(User, related_name="Logs", on_delete=models.CASCADE)

    class Meta:
        verbose_name_plural = verbose_name = "识别记录"




