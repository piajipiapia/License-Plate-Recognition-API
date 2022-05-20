from django.shortcuts import render
from django.http import JsonResponse, HttpResponse
from rest_framework.views import APIView
from rest_framework.authentication import SessionAuthentication
from rest_framework import permissions
from drf_yasg.utils import swagger_auto_schema
from drf_yasg import openapi
from django.contrib import auth
from app.serializers import *
import time, random, re, base64
from PIL import Image
from io import BytesIO
import numpy
from keras.models import load_model
from rest_framework import generics
from rest_framework.pagination import PageNumberPagination
from rest_framework.response import Response
import multiprocessing
import tensorflow as tf
from nets.cnn_model.colourDetection import judge_plate_color
import threading
from tensorflow.python.keras.backend import set_session
from nets.demo import *
import uuid
# 程序开始时声明
sess = tf.Session()
graph = tf.get_default_graph()

# 在model加载前添加set_session
set_session(sess)

model = reload_model.LPR("nets/model/cascade.xml"  ## opencv 定位
                         , "nets/model/model12.h5",  ## 神经网络切割
                         "nets/model/ocr_plate_all_gru.h5")  ## 神经网络识别


# Create your views here.

class CsrfExemptSessionAuthentication(SessionAuthentication):
    """
    关闭csrf验证
    """

    def enforce_csrf(self, request):
        return  # To not perform the csrf check previously happening





class OCR(APIView):
    authentication_classes = (CsrfExemptSessionAuthentication,)
    permission_classes = (permissions.AllowAny, )

    @swagger_auto_schema(
        operation_summary="识别图片、 (需登录)",
        operation_description='识别图片！ ',
        request_body=openapi.Schema(
            type=openapi.TYPE_OBJECT,
            properties={
                'image': openapi.Schema(
                    type=openapi.TYPE_STRING,

                    description='图片 base64 格式 ： ‘data:image/jpeg;base64,iVBORw0KGgoAA==’'
                ),

            }),
        responses={
            "200": "识别成功！",
            "500": "识别错误！ 可能是图片错误"
        }
    )
    def post(self, request):
        request_data = json.loads(request.body.decode())
        image = request_data.get("image")
        target_size = (150, 150)
        global graph
        global sess
        img = Image.open(BytesIO(base64.b64decode(re.sub(r"data:image\S+?base64,", "", image))))
        grr = np.array(img)[..., ::-1]
        image_name = f"/static/{uuid.uuid4().hex}.jpg"
        with graph.as_default():
            set_session(sess)
            ocr_data = model.SimpleRecognizePlateByE2E(grr)
        confidence = 0
        if ocr_data:
            pstr, confidence, rect = sorted(ocr_data, key=lambda s: s[1])[::-1][0]

        if confidence < 0.7:
            return JsonResponse({"code": 500, "msg": "暂未检测到车牌!请选择更加清晰的照片"})

        carimg = grr[int(rect[1]):int(rect[1] + rect[3]),
                 int(rect[0]):int(rect[0] + rect[2])]  # 裁剪坐标为[y0:y1, x0:x1]

        cv2.imwrite(f"app/{image_name}", carimg)
        color = judge_plate_color(carimg[..., ::-1])[0]

        return JsonResponse({"code": 200, "data": {
            "image": image_name,
            "result": {
                "color": color,
                "carname": pstr,
                "loc": str([int(rect[0]), int(rect[1]), int(rect[0] + rect[2]), int(rect[1] + rect[3])])
            }

        }})


class MyPageNumberPagination(PageNumberPagination):
    page_size = 10
    max_page_size = 500
    page_size_query_param = 'limit'
    page_query_param = 'page'

    def get_paginated_response(self, data):
        return Response(data)

