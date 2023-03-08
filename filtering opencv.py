import numpy as np
import matplotlib.pyplot as plt

import cv2
# 폴더 만들기
import os



# 파일에 있는 이미지 읽어오기
def get_imlist(path):
    return [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.jpg')]
# norm
def norm_get_imlist(path):
    """all return a list of filenames for all jpg images in a directory """
    return [os.path.join(path, f) for f in file_list if f.endswith('.jpg')]
# 디렉토리 만들기
def make_dir(dir_path,dir_name):
     return os.mkdir(dir_path + "/" + dir_name + "/")

#경로 합치기
def path(dir_path,dir_name):
    return os.path.join(dir_path,dir_name)



#이미지 가져오기
# get directory all files
'''
norm_file=open("norm_circle,white.txt",mode='a')
for root, dirs, files in os.walk('./circle,white/'):
    tsne_filelist=[0]*len(files)
    for file in files:
         file=file.replace('.jpg',"")
         norm_file.write("%s\n"%file)
         print(file)
# get the list of images in the directory
cap_file=open("norm_circle,white.txt",mode="rt")
lines=cap_file.readlines()
file_list=[0]*len(lines)
for line,i in zip(lines,range(0,len(lines))):
    file_list[i]=line
file_list.sort()
for i in range(len(file_list)):
  file_list[i]=file_list[i]+str('.jpg')
  file_list[i]=file_list[i].replace("\n","")
print(file_list)
'''
imlist = get_imlist('./norm_circle,white/')

#파일 이름 리스트로 받기
path_dir='./norm_circle,white/'
file_list=os.listdir(path_dir)
dir_path='./'

# ADAPTIVE THRESHOLDING
make_dir(dir_path,'norm_circle,white_opencv')
a=path(dir_path,'norm_circle,white_opencv')
make_dir(a,'gray_image')
make_dir(a,'thresh_global')
make_dir(a,'thresh_mean')
make_dir(a,'thresh_gaussian')
path_gray_image=path(a,'gray_image')
path_thresh_global=path(a,'thresh_global')
path_thresh_mean=path(a,'thresh_mean')
path_thresh_gaussian=path(a,'thresh_gaussian')

for im,list in zip(imlist,file_list):
 gray_image = cv2.imread(im,0)

 ret, thresh_global = cv2.threshold(gray_image, 190, 255, cv2.THRESH_BINARY)
 thresh_mean = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
 thresh_gaussian = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
 names = ['Original Image', 'Global Thresholding', 'Adaptive Mean Threshold', 'Adaptive Gaussian Thresholding']
 images = [gray_image, thresh_global, thresh_mean, thresh_gaussian]
 cv2.imwrite(os.path.join(path_gray_image,list), gray_image)
 cv2.imwrite(os.path.join(path_thresh_global,list), thresh_global)
 cv2.imwrite(os.path.join(path_thresh_mean,list), thresh_mean)
 cv2.imwrite(os.path.join(path_thresh_gaussian,list), thresh_gaussian)


for i in range(4):
 plt.subplot(2, 2, i + 1), plt.imshow(images[i], 'gray')
 plt.title(names[i])
 plt.xticks([]), plt.yticks([])

plt.show()





'''

# images_visual_features
make_dir(dir_path,'images_visual_features')
b=path(dir_path,'images_visual_features')
make_dir(b,'canny')
make_dir(b,'sobel')
make_dir(b,'laplacian')
path_canny=path(b,'canny')
path_sobel=path(b,'sobel')
path_laplacian=path(b,'laplacian')

for im,list in zip(imlist,file_list):
#imgae_visual_features
    src = cv2.imread(im, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_8U, ksize=3)
    canny = cv2.Canny(src, 55, 400)
    sobel = cv2.Sobel(gray, cv2.CV_8U, 0, 1, 3)
# 이미지 저장하기
    cv2.imwrite(os.path.join(path_canny,list), canny)
    cv2.imwrite(os.path.join(path_sobel,list ),sobel )
    cv2.imwrite(os.path.join(path_laplacian,list), laplacian )
# cv2.Sobel(그레이스케일 이미지, 정밀도, x방향 미분, y방향 미분, 커널, 배율, 델타, 픽셀 외삽법)
'''
'''ddepth : 결과 이미지 데이터 삽입
cv2.CV_8U(이미지 픽셀값을 unit8로 설정)
cv2.CV_16U(이미지 픽셀값을 unit16로 설정)
cv2.CV_32F(이미지 픽셀값을 float32로 설정)
cv2.CV_64F(이미지 픽셀값을 float64로 설정)
'''
'''
# 정밀도는 이미지의 정밀도
# x방향미분은 x방향으로 미분할 값
# y방향미분은 y방향으로 미분할 값
# 커널은 소벨 커널의 크기설정, 1,3,5,7 주로사용
# 배율은 계산된 미분값에 대한 배율
# 델타는 계산전미분값에 대한 추가값
# 픽셀 외삽법은 이미지를 가장자리를 처리할 경우 영역밖의 픽셀은 추정해서 값을 할당하는데 테두리모드라고 생각하면 됨.
# x방향 미분값과 y 방향 미분값의 합이 1이상이어야 하고 각각의 값은 0보다 커야함.

# cv2.Laplacian(그레이스케일 이미지, 정밀도, 커널, 배율, 델타, 픽셀 외삽법)
# 정밀도는 결과이미지의 정밀도
# 배율은 계산된 미분값에 대한 배율값
# 델타는 계산전 미분값에 대한 추가값
# 픽셀 외삽법은 이미지를 가장자리를 처리할 경우 영역밖의 픽셀은 추정해서 값을 할당하는데 테두리모드라고 생각하면 됨.
# 커널 값이 1일 경우 3x3 aperture size를 사용함(중심값 -4)
'''
'''
cv2.imshow("canny", canny)
cv2.imshow("sobel", sobel)
cv2.imshow("laplacian", laplacian)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''
'''






#thresholding_images
# here 0 means that the image is loaded in gray scale format
make_dir(dir_path,'threshold_images')
c=path(dir_path,'threshold_images')
make_dir(c,'gray_image')
make_dir(c,'thresh_binary')
make_dir(c,'thresh_binary_inv')
make_dir(c,'thresh_trunc')
make_dir(c,'thresh_tozero')
make_dir(c,'thresh_tozero_inv')
path_gray_image=path(c,'gray_image')
path_thresh_binary=path(c,'thresh_binary')
path_thresh_binary_inv=path(c,'thresh_binary_inv')
path_thresh_trunc=path(c,'thresh_trunc')
path_thresh_tozero=path(c,'thresh_tozero')
path_thresh_tozero_inv=path(c,'thresh_tozero_inv')



for im,list in zip(imlist,file_list):
 gray_image = cv2.imread(im,0)

 ret, thresh_binary = cv2.threshold(gray_image, 190, 255, cv2.THRESH_BINARY)
 ret, thresh_binary_inv = cv2.threshold(gray_image, 190, 255, cv2.THRESH_BINARY_INV)
 ret, thresh_trunc = cv2.threshold(gray_image, 190, 255, cv2.THRESH_TRUNC)
 ret, thresh_tozero = cv2.threshold(gray_image, 127, 255, cv2.THRESH_TOZERO)
 ret, thresh_tozero_inv = cv2.threshold(gray_image, 190, 255, cv2.THRESH_TOZERO_INV)
# DISPLAYING THE DIFFERENT THRESHOLDING STYLES
 names = ['Oiriginal Image', 'BINARY', 'THRESH_BINARY_INV', 'THRESH_TRUNC', 'THRESH_TOZERO', 'THRESH_TOZERO_INV']
 images = [gray_image, thresh_binary, thresh_binary_inv, thresh_trunc, thresh_tozero, thresh_tozero_inv]
 cv2.imwrite(os.path.join(path_gray_image,list),gray_image )
 cv2.imwrite(os.path.join(path_thresh_binary ,list),thresh_binary )
 cv2.imwrite(os.path.join(path_thresh_binary_inv ,list),thresh_binary_inv )
 cv2.imwrite(os.path.join(path_thresh_trunc,list),thresh_trunc )
 cv2.imwrite(os.path.join(path_thresh_tozero,list),thresh_tozero )
 cv2.imwrite(os.path.join(path_thresh_tozero_inv,list), thresh_tozero_inv)

for i in range(6):
 plt.subplot(2, 3, i + 1), plt.imshow(images[i], 'gray')
 plt.title(names[i])
 plt.xticks([]), plt.yticks([])

plt.show()


'''