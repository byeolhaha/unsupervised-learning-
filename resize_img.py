# -*- coding:utf-8 -*-
import os


# 디렉토리 만들기
def make_dir(dir_path,dir_name):
     return os.mkdir(dir_path + "/" + dir_name + "/")

path_dir= "./cat_dog/"
file_list=os.listdir(path_dir)
make_dir("./","resized_cat_dog")

for i in file_list:
 p= os.path.join(path_dir, i)
 img = cv2.imread(p)
 # 행 : Height, 열:width
 print(i)
 #height, width = img.shape[:2]
# 이미지 축소
 print(img)
 shrink = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
 cv2.imshow('Shrink', shrink)
 cv2.imwrite(os.path.join('./resized_cat_dog/', str(i)), shrink)
 #cv2.waitKey(0)
 #cv2.destroyAllWindows()
