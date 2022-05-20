



import time

def SpeedTest(image_path):
    """取20张图片做识别速度测试"""
    grr = np.array(Image.open(image_path))
    model = pr.LPR("model/cascade.xml", "model/model12.h5", "model/ocr_plate_all_gru.h5")
    model.SimpleRecognizePlateByE2E(grr)
    t0 = time.time()
    for x in range(20):
        model.SimpleRecognizePlateByE2E(grr)
    t = (time.time() - t0)/20.0
    print ("Image size :" + str(grr.shape[1])+"x"+str(grr.shape[0]) +  " need " + str(round(t*1000,2))+"ms")

    

from PIL import ImageFont
from PIL import Image
from PIL import ImageDraw
fontC = ImageFont.truetype("./nets/Font/platech.ttf", 14, 0)

def drawRectBox(image,rect,addText):
    """在图片中框出选框 并且贴上文字"""
    cv2.rectangle(image, (int(rect[0]), int(rect[1])), (int(rect[0] + rect[2]), int(rect[1] + rect[3])), (0,0, 255), 2,cv2.LINE_AA)
    cv2.rectangle(image, (int(rect[0]-1), int(rect[1])-16), (int(rect[0] + 115), int(rect[1])), (0, 0, 255), -1,
                  cv2.LINE_AA)
    img = Image.fromarray(image)
    draw = ImageDraw.Draw(img)
    draw.text((int(rect[0]+1), int(rect[1]-16)), addText, (255, 255, 255), font=fontC)
    imagex = np.array(img)
    return imagex





import reload_model 
import cv2
import numpy as np


if  __name__ == "__main__":
    
    grr = np.array(Image.open("images_rec/0025-0_1-317483_392511-391511_317510_318483_392484-0_0_9_33_32_24_7-107-4.jpg"))

    model = reload_model.LPR("model/cascade.xml","model/model12.h5","model/ocr_plate_all_gru.h5")
    for pstr,confidence,rect in model.SimpleRecognizePlateByE2E(grr):
            if confidence>0.5:
                image = drawRectBox(grr, rect, pstr+" "+str(round(confidence,3)))
                print ("plate_str:")
                print (pstr)
                print ("plate_confidence")
                print (confidence)                
    cv2.imshow("image",image[:,:,::-1])
    cv2.waitKey(0)



    SpeedTest("images_rec/0025-0_1-317483_392511-391511_317510_318483_392484-0_0_9_33_32_24_7-107-4.jpg")
