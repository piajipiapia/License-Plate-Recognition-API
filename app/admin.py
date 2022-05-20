from django.contrib import admin

from app.models import *


# Register your models here.

class LogsAdmin(admin.ModelAdmin):
    list_display = [
      "id",  "date", "user", "result"
    ]


admin.site.register(Logs, LogsAdmin)
