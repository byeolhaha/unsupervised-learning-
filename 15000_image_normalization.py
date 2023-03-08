import cv2 as cv
import os

# 파일에 있는 이미지 읽어오기
def get_imlist(path):
    return [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.jpg')]

imlist= get_imlist('./dif_16000/')
file_list=os.listdir('./dif_16000/')



for images,list in zip(imlist,file_list):
    sum_colors = [0, 0, 0]
    img=cv.imread(images)
    for i in range (0,10):
      for j in range(0,10):
          a = img[i, j]
          sum_colors = sum_colors + a

    avg_colors = sum_colors / 100
    norm_colors= [222.22,206.22,190.22] -avg_colors
    for x in range(0, 473):
       for y in range(0,1179):
        normalize = img[x, y] + norm_colors
        a,b,c= normalize
        normalize=[0] *3
        normalize[0]=a
        normalize[1]=b
        normalize[2]=c
        for num in range(0,3):
            pixel= normalize[num]
            if pixel < 0 :
                normalize[num] = 0
            elif pixel > 255:
                normalize[num] = 255
        img[x,y]=normalize
    cv.imwrite(os.path.join("./dif_new_16000/", list), img)





