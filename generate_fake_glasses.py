#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  3 10:12:11 2018

@author: chenys
"""

import dlib
from PIL import Image
import glob
import cv2
import os
from imutils import face_utils, translate, rotate, resize
import numpy as np



path = input("输入原始图像路径："    )
max_width = int(input("输入希望得到的数据的宽度："))
pic_type = input("输入的图片格式为(png,jpg等）:")
new_path = input("生成文件夹名：")
if os.path.exists(new_path):
    print('文件夹存在，继续。。。')
    pass
else:
    os.mkdir(new_path)


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
dir_list = os.listdir(path)


for dir_l in dir_list:
    print(dir_l)
    tmp= os.path.join(new_path,dir_l)
    if os.path.exists(tmp):
        pass
    else:
        os.mkdir(tmp)
    if pic_type =='jpg':
        path = os.path.join(path,dir_l,'*.jpg')
    else:
        path = os.path.join(path,dir_l,'*.png') 
    
    
    
    imagelist = glob.glob(path)
    for image_path in imagelist:
        image = cv2.imread(image_path)    
        image = resize(image, width=max_width)
        save_name = image_path.split('/')[-1]
        
        ##获取随机眼镜
        index = np.random.randint(1,19)
        glass = Image.open('glasses/'+str(index)+".png")
    
           
        img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = []
        
        rects = detector(img_gray, 0)
        img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        for rect in rects: 
            
            face = {}
            shades_width = rect.right() - rect.left()
        
            # predictor used to detect orientation in place where current face is
            shape = predictor(img_gray, rect)
            shape = face_utils.shape_to_np(shape)
        
            # grab the outlines of each eye from the input image
            leftEye = shape[36:42]
            rightEye = shape[42:48]
        
            # compute the center of mass for each eye
            leftEyeCenter = leftEye.mean(axis=0).astype("int")
            rightEyeCenter = rightEye.mean(axis=0).astype("int")
        
        	    # compute the angle between the eye centroids
            dY = leftEyeCenter[1] - rightEyeCenter[1] 
            dX = leftEyeCenter[0] - rightEyeCenter[0]
            angle = np.rad2deg(np.arctan2(dY, dX)) 
        
            current_glass = glass.resize((shades_width, int(shades_width * glass.size[1] / glass.size[0])),
                                   resample=Image.LANCZOS)
            current_glass = current_glass.rotate(angle, expand=True)
            current_glass = current_glass.transpose(Image.FLIP_TOP_BOTTOM)
        
            face['glasses_image'] = current_glass
            left_eye_x = leftEye[0,0] - shades_width // 6
            left_eye_y = leftEye[0,1] - shades_width // 8
            face['final_pos'] = (left_eye_x, left_eye_y)
        
            
            
            img.paste(current_glass, (left_eye_x, left_eye_y), current_glass)
    
        
            
            fake_image = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
            save_name = os.path.join(tmp,save_name)
            cv2.imwrite(save_name, fake_image)

