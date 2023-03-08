import os
import cv2 as cv
import sys
# pca 실행
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import axis, imshow, gray, show, figure, subplot, plot
from scipy import linalg
from PIL import Image, ImageDraw
from scipy.cluster.vq import *
from sklearn.decomposition import PCA
import matplotlib.transforms as mtransforms
from sklearn.cluster import KMeans
from sklearn import metrics
from scipy.spatial.distance import cdist
import numpy as np
import os

#from pylab import *
# zarr better than hdf5
import zarr
zarr.__version__


# 디렉토리에 있는 파일들을 리스트로 만들고 그리고 그 리스트레 있는 이름들을 이미지로 가져오기
def get_imlist(path):
    """all return a list of filenames for all jpg images in a directory """
    return [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.jpg')]

# 디렉토리 path와 내가 지정한 리스트의 이름을 비교해서 리스트에 있는 이미지 이름만을 path에 저장된 곳에서 가져오기
def get_self_imlist(path):
    """all return a list of filenames for all jpg images in a directory """
    return [os.path.join(path, f) for f in file_cluster if f.endswith('.jpg')]
'''
# tsne

def tsne_get_imlist(path):
    """all return a list of filenames for all jpg images in a directory """
    return [os.path.join(path, f) for f in file_list if f.endswith('.jpg')]
'''
def pca(X):
    """principal Component Analysis
        input: X, matrix with training data stored as flattend arrays in rows
        return : projection matrix (with important dimention first),wariance and mean"""
    # get dimention
    num_data, dim = X.shape
    # axis=0 의 의미를 행끼리
    mean_X = X.mean(axis=0)
    X = X - mean_X
    #X = np.array(X, dtype=np.float16)
    if dim > num_data:
        M = np.dot(X, X.T)  # covariance 메트릭스
        e, EV = linalg.eigh(M)  # 에이젠 벡터과 에이젠 value
        tmp = np.dot(X.T, EV).T
        V = tmp[::-1]
        #e=abs(e)
        S = np.sqrt(e)[::-1]
        print('V=',V)
        print('S=',S)
        for i in range(V.shape[1]):
            # 열끼리
            V[:,i] /= S

    else:
        # PCA -SVD
        U, S, V = linalg.svd(X)
        V = V[:num_data]
    return V, S, mean_X

#디렉토리 만들기
def make_dir(dir_path, dir_name):
    return os.mkdir(dir_path + "/" + dir_name + "/")


def remove_values_from_list(the_list, val):
   return [value for value in the_list if value != val]

axisname= 'RGB_16000_0.2(cap_complete)'

# get directory all files
'''
cap_file=open("norm_16000_delete_each.txt",mode='a')
for root, dirs, files in os.walk('./norm_16000_9/'):
    tsne_filelist=[0]*len(files)
    for file in files:
         file=file.replace('.jpg',"")
         cap_file.write("%s\n"%file)
         print(file)
# get the list of images in the directory
cap_file=open("norm_16000_delete_each.txt",mode="rt")
lines=cap_file.readlines()
file_list=[0]*len(lines)
for line,i in zip(lines,range(0,len(lines))):
    file_list[i]=line
file_list.sort()
for i in range(len(file_list)):
  file_list[i]=file_list[i]+str('.jpg')
  file_list[i]=file_list[i].replace("\n","")
print(file_list)

imlist = tsne_get_imlist('./resized_16000_0.2/')
'''
imlist= get_imlist('./RGB_16000_0.2(cap_complete)/')

im = np.array(Image.open(imlist[0]))  # 크기를 알기 위해 예시 하나 뽑기
#m, n = im.shape[0:2]
m, n ,c= im.shape[0:3]  # 행렬의 길이 알기 및 채널 알기
imnbr = len(imlist)  # 전체 그림의 수를 알기
print(imnbr)

#tmp = np.zeros((imnbr,22420))
tmp = np.zeros((imnbr,67260))
#tmp = np.zeros((imnbr,557667))
#tmp = np.zeros((imnbr,1673001))
i = 0

for im in imlist:
    #i += 1
    # 원래 (행,렬, 모드)로 저장되는 array를 1차원으로 만들어 array로 저장하기
    #tmp[i, :] = np.array(Image.open(im)).flatten().reshape(1, 22420)
    tmp[i, :] = np.array(Image.open(im)).flatten().reshape(1, 67260)
    #tmp[i, :] = np.array(Image.open(im)).flatten().reshape(1,557667 )
    # tmp[i,:] = np.array(Image.open(im)).flatten().reshape(1, 1673001)
    i += 1
    print(i)
    #if  i== 1 :
    #    tmp = np.delete(tmp, (0), axis=0)

print('tmp.shape=',tmp.shape)
print('tmp=',tmp)


V, S, mean_X = pca(tmp)
print("V.shape=",V.shape)
# (100, 1673001)
print("mean_X.shape=",mean_X.shape)

#zarr 적용하여 최적의  chunk 찾기
VV = zarr.array(V, chunks=True)
mean_XX= zarr.array(mean_X,chunks=True)

# zarr 저장하기
zarr.save_array('./VV.zarr', VV)
zarr.save_array('./mean_XX.zarr', mean_XX)

figure()
gray()
subplot(3, 4, 1)  # 그림들이 한 줄아 4개씩 해서 총 8개의 그림이 등장
#imshow(mean_X.reshape(m, n))
imshow(mean_X.reshape(m, n, c))

for i in range(7):  # i는 0부터 6
    subplot(3, 4, i + 2)
    #imshow(V[i].reshape(m, n))
    imshow(V[i].reshape(m, n,c))

show()


# load model file 가져오기

V=zarr.load('./VV.zarr')
mean_X=zarr.load('./mean_XX.zarr')

#예시 하나 보여주기
imshow(Image.open(imlist[0]))
gray()
axis('off')
show()



# 모든 이미지로 1차원의 이미지로 만들기
immatrix = np.array([np.array(Image.open(im)).flatten()
                  for im in imlist], 'f')

# project on the 40 first PCs
mean_X=np.array(mean_X)
mean_X=mean_X.flatten()
#elbow method

distortions = []
inertias = []
mapping1 = {}
mapping2 = {}
K = range(1, 10)

for k in K:
    # Building and fitting the model
    kmeanModel = KMeans(n_clusters=k).fit(tmp)
    kmeanModel.fit(tmp)

    distortions.append(sum(np.min(cdist(tmp, kmeanModel.cluster_centers_,
                                        'euclidean'), axis=1)) / tmp.shape[0])
    inertias.append(kmeanModel.inertia_)

    mapping1[k] = sum(np.min(cdist(tmp, kmeanModel.cluster_centers_,
                                   'euclidean'), axis=1)) / tmp.shape[0]
    mapping2[k] = kmeanModel.inertia_

for key,val in mapping1.items():
    print(str(key)+' : '+str(val))

plt.plot(K, distortions, 'bx-')
plt.xlabel('Values of K')
plt.ylabel('Distortion')
plt.title('The Elbow Method using Distortion')
plt.show()

#적절한 차원의 수 알아보기
pcaa=PCA()
pcaa.fit(tmp)
cumsum=np.cumsum(pcaa.explained_variance_ratio_)
d=np.argmax(cumsum >= 0.95) + 1
print('적절한 차원의 수=',d)

# pca 차원이 3으로 나옴
pc = 3
cluster = [4,5,6,7,8,9]


projected = np.array([np.dot(V[:pc],immatrix[i]-mean_X) for i in range(imnbr)])

#projected 파일을 저장하기
'''
RESULT_FILE = 'resized_15000_0.2_188.out'

f = open(RESULT_FILE, 'tw')
n,d = projected.shape
for i in range(n):
    for j in range(d):
        print("%f" % projected[i, [j]], end = '', file = f)
        if j < d - 1: print(',', end = '', file = f)
    print(file = f)

f.close()
'''
for c in cluster:
#k-means
 projected= whiten(projected)
 centroids,distortion=kmeans(projected,c)

 code,distance=vq(projected,centroids)




 dir_path='/home/j35635aa/PycharmProjects/untitled'
 print(imlist)
 # make cluster folder
 for k in range(c):
    ind = np.where(code == k)[0]
    ind=list(ind)
    print(ind)
    file_cluster = [0] * len(ind)
    for num,ind_num in zip(range(0,len(ind)),ind):
        file_cluster[num]=imlist[ind_num]
    print(file_cluster)
    for r in range(len(file_cluster)):
        file_cluster[r]=file_cluster[r].replace('./RGB_16000_0.2(cap_complete)/',"")
    print(file_cluster)
    self_imlist = get_self_imlist('./RGB_16000_0.2(cap_complete)/')
    dir_name= ("%s_%i_%i"%(axisname,c,k))
    make_dir(dir_path, dir_name)
    for i,j in zip(self_imlist,range(0,len(file_cluster))):
          img = cv.imread(i)
          cv.imwrite(os.path.join("./%s"%dir_name, file_cluster[j]), img)
          cv.waitKey(0)


    figure()
    gray()
    for i in range(np.minimum(len(ind), 40)):
        subplot(c,10,i+1)
        #imshow(immatrix[ind[i]].reshape((473,1179,3)))
        #imshow(immatrix[ind[i]].reshape((473, 393,3)))
        #imshow(immatrix[ind[i]].reshape((236, 95)))
        imshow(immatrix[ind[i]].reshape((236, 95,3)))

        axis('off')
 show()


#height and width
 h,w = 3000,3000

 img = Image.new('RGB',(w,h),(255,255,255))
 draw = ImageDraw.Draw(img)

 draw.line((0,h/2,w,h/2),fill=(255,0,0))
 draw.line((w/2,0,w/2,h),fill=(255,0,0))

 scale = abs(projected).max(0)
 scale_xy=scale[0:2]
 scale_xz=[scale[0], scale[2]]
 scale_yz=[scale[1],scale[2]]
 projected_xy=projected[:,0:2]

 projected_xz=np.zeros((imnbr,2))
 projected_yz=np.zeros((imnbr,2))


 for i in range(0,imnbr):
    projected_xz[i,0]=np.array(projected[i,0])
    projected_xz[i,1]=np.array(projected[i,2])
    projected_yz[i,0] =np.array(projected[i, 1])
    projected_yz[i,1] =np.array(projected[i, 2])

 scaled_xy = np.floor(np.array([(p / scale_xy) * (w/2-20,h/2-20) + (w/2,h/2) for p in projected_xy]))
 scaled_xz= np.floor(np.array([(p / scale_xz) * (w/2-20,h/2-20) + (w/2,h/2) for p in projected_xz]))
 scaled_yz = np.floor(np.array([(p / scale_yz) * (w/2-20,h/2-20) + (w/2,h/2) for p in projected_yz]))
 #print(scaled)
 for i in range(imnbr):
    nodeim = Image.open(imlist[i])
    nodeim.thumbnail((109,109)) #돌아가는 숫자들 : 알약버전 357, 25, 7, 35, 49, 499, 109, 219, 249, 259, 559, 659 글씨버전 : 499 499
    '''
    if code[i] == 0:
        flag = Image.new("RGB", (451, 3), (255, 0, 0))  # 빨강
    elif code[i] == 1:
        flag = Image.new("RGB", (451, 3), (225, 228, 0))  # 노란색
    elif code[i] == 2:
        flag = Image.new("RGB", (451, 3), (0, 255, 0))  # 연두색
    elif code[i] == 3:
        flag = Image.new("RGB", (451, 3), (153, 153, 102))  # 황토

    nodeim.paste(flag, (0, 0, 451, 3))
    '''
    ns = nodeim.size
    #print(ns[0]//2, ns[1]//2)


    img.paste(nodeim,(int(scaled_xy[i][0]-ns[0]//2),int(scaled_xy[i][1]-ns[1]//2),
                      int(scaled_xy[i][0]+ns[0]//2)+1,int(scaled_xy[i][1]+ns[1]//2)+1))
    #print(int(scaled[i][0]-ns[0]//2),int(scaled[i][1]-ns[1]//2))

 img.save('%s_%i_xy.jpg'%(axisname,c))



 figure(figsize=(9,9))

 plot(centroids,'bx')

 imshow(np.array(img))
# plt.axis('off')
 show()


 img_1 = Image.new('RGB',(w,h),(255,255,255))
 draw_1 = ImageDraw.Draw(img_1)

 draw_1.line((0,h/2,w,h/2),fill=(255,0,0))
 draw_1.line((w/2,0,w/2,h),fill=(255,0,0))


#print(scaled)
 for i in range(imnbr):
    nodeim = Image.open(imlist[i])
    nodeim.thumbnail((109,109)) #돌아가는 숫자들 : 알약버전 357, 25, 7, 35, 49, 499, 109, 219, 249, 259, 559, 659 글씨버전 : 499 499
    '''
    if code[i] == 0:
        flag = Image.new("RGB", (451, 3), (255, 0, 0))  # 빨강
    elif code[i] == 1:
        flag = Image.new("RGB", (451, 3), (225, 228, 0))  # 노란색
    elif code[i] == 2:
        flag = Image.new("RGB", (451, 3), (0, 255, 0))  # 연두색
    elif code[i] == 3:
        flag = Image.new("RGB", (451, 3), (153, 153, 102))  # 황토
    nodeim.paste(flag, (0, 0, 451, 3))
    '''
    ns = nodeim.size
    #print(ns[0]//2, ns[1]//2)
    img_1.paste(nodeim,(int(scaled_xz[i][0]-ns[0]//2),int(scaled_xz[i][1]-ns[1]//2),
                      int(scaled_xz[i][0]+ns[0]//2)+1,int(scaled_xz[i][1]+ns[1]//2)+1))
    #print(int(scaled[i][0]-ns[0]//2),int(scaled[i][1]-ns[1]//2))

 img_1.save('%s_%i_xz.jpg'%(axisname,c))



 figure(figsize=(9,9))

 plot(centroids,'bx')

 imshow(np.array(img_1))
 # plt.axis('off')
 show()


 img_2 = Image.new('RGB',(w,h),(255,255,255))
 draw_2 = ImageDraw.Draw(img_2)

 draw_2.line((0,h/2,w,h/2),fill=(255,0,0))
 draw_2.line((w/2,0,w/2,h),fill=(255,0,0))


 #print(scaled)
 for i in range(imnbr):
    nodeim = Image.open(imlist[i])
    nodeim.thumbnail((109,109)) #돌아가는 숫자들 : 알약버전 357, 25, 7, 35, 49, 499, 109, 219, 249, 259, 559, 659 글씨버전 : 499 499
    '''
    if code[i] == 0:
        flag = Image.new("RGB", (451, 3), (255, 0, 0))  # 빨강
    elif code[i] == 1:
        flag = Image.new("RGB", (451, 3), (225, 228, 0))  # 노란색
    elif code[i] == 2:
        flag = Image.new("RGB", (451, 3), (0, 255, 0))  # 연두색
    elif code[i] == 3:
        flag = Image.new("RGB", (451, 3), (153, 153, 102))  # 황토
    nodeim.paste(flag, (0, 0, 451, 3))
    '''
    ns = nodeim.size
    #print(ns[0]//2, ns[1]//2)
    img_2.paste(nodeim,(int(scaled_yz[i][0]-ns[0]//2),int(scaled_yz[i][1]-ns[1]//2),
                      int(scaled_yz[i][0]+ns[0]//2)+1,int(scaled_yz[i][1]+ns[1]//2)+1))
    #print(int(scaled[i][0]-ns[0]//2),int(scaled[i][1]-ns[1]//2))

 img_2.save('%s_%i_yz.jpg'%(axisname,c))



 figure(figsize=(9,9))

 plot(centroids,'bx')

 imshow(np.array(img_2))
 # plt.axis('off')
 show()