
import cv2
import numpy as np

# 使用opencv 的haar进行对车牌位置的定位

watch_cascade = cv2.CascadeClassifier('./model/cascade.xml')


def computeSafeRegion(shape,bounding_rect):
    ## 转换选框的坐标点
    top = bounding_rect[1] # y
    bottom  = bounding_rect[1] + bounding_rect[3] # y +  h
    left = bounding_rect[0] # x
    right =   bounding_rect[0] + bounding_rect[2] # x +  w

    min_top = 0
    max_bottom = shape[0]
    min_left = 0
    max_right = shape[1]

 
    if top < min_top:
        top = min_top
    if left < min_left:
        left = min_left


    if bottom > max_bottom:
        bottom = max_bottom

    if right > max_right:
        right = max_right

    return [left,top,right-left,bottom-top]


def cropped_from_image(image,rect):
    ## 将车牌从图片中切出来
    x, y, w, h = computeSafeRegion(image.shape,rect)
    return image[y:y+h,x:x+w]


def detectPlateRough(image_gray,resize_h = 720,en_scale =1.08 ,top_bottom_padding_rate = 0.05):
    # 将图片中所有的车牌匹配出来 并且进行切出来  这个方法会返回一个已经切好了的图片列表

    if top_bottom_padding_rate>0.2:
        print("错误 图片padding填充比例不能高于 > 0.2:",top_bottom_padding_rate)
        exit(1)

    height = image_gray.shape[0]
    padding =    int(height*top_bottom_padding_rate)
    scale = image_gray.shape[1]/float(image_gray.shape[0])

    image = cv2.resize(image_gray, (int(scale*resize_h), resize_h))

    image_color_cropped = image[padding:resize_h-padding,0:image_gray.shape[1]]

    image_gray = cv2.cvtColor(image_color_cropped,cv2.COLOR_RGB2GRAY)

    watches = watch_cascade.detectMultiScale(image_gray, en_scale, 2, minSize=(36, 9),maxSize=(36*40, 9*40))

    cropped_images = []
    for (x, y, w, h) in watches:
        cropped_origin = cropped_from_image(image_color_cropped, (int(x), int(y), int(w), int(h)))
        x -= w * 0.14
        w += w * 0.28
        y -= h * 0.6
        h += h * 1.1;

        cropped = cropped_from_image(image_color_cropped, (int(x), int(y), int(w), int(h)))


        cropped_images.append([cropped,[x, y+padding, w, h],cropped_origin])
    return cropped_images
