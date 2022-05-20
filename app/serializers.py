from rest_framework import serializers

from .models import *
import json


class LogsSerializers(serializers.ModelSerializer):
    username = serializers.SerializerMethodField()
    result = serializers.SerializerMethodField()

    class Meta:
        model = Logs
        fields = ["id", "date", "image", "result", "username"]

    def get_result(self, obj):
        return json.loads(obj.result)

    def get_username(self, obj):
        return obj.user.username


class UserSerializers(serializers.ModelSerializer):
    class Meta:
        model = User
        fields = [
            "id", "username", "is_staff", "is_superuser", "is_active",
            "date_joined", "last_login"
        ]
