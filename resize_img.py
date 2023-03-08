# -*- coding:utf-8 -*-
import cv2
import numpy
import os


path_dir= "./RGB_16000(cap_complete)/"
file_list=os.listdir(path_dir)


for i in file_list:
 p= os.path.join(path_dir, i)
 img = cv2.imread(p)
 # 행 : Height, 열:width
 print(i)
 #height, width = img.shape[:2]
# 이미지 축소
 print(img)
 shrink = cv2.resize(img, None, fx=0.2, fy=0.2, interpolation=cv2.INTER_AREA)
 cv2.imshow('Shrink', shrink)
 cv2.imwrite(os.path.join('./RGB_16000_0.2(cap_complete)/', str(i)), shrink)
 #cv2.waitKey(0)
 #cv2.destroyAllWindows()