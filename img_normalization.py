import os

# 파일에 있는 이미지 읽어오기
def get_imlist(path):
    return [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.jpg')]

imlist= get_imlist('./resized_15000_0.2/')
sum_colors=0
for imgs in imlist :
    img=cv.imread(imgs)
    for i in range (0,10):
      for j in range(0,10):
          a = img[i, j]
          print(a)
          sum_colors = sum_colors + a

    avg_colors = sum_colors / 100
    norm_colors= [224.46,206.06,190.86] -avg_colors
    if norm_colors.any > 0:
      for i in range(0,236):
        for j in range(0,95):
            normalize = img[i, j] + norm_colors
            for i in normalize :
                if i < 0 :
                    i = 0
                elif i > 255:
                    i= 255
            img[i,j] = normalize






