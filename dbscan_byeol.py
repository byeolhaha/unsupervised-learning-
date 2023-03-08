import time
import numpy as np
from PIL import Image
import pandas as pd
from sklearn.decomposition import PCA, IncrementalPCA
import matplotlib.pyplot as plt
from sklearn .manifold import TSNE
from sklearn.cluster import DBSCAN
import os
import sns
import matplotlib
def get_imlist(path):
    """all return a list of filenames for all jpg images in a directory """
    return [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.jpg')]

imlist= get_imlist('./RGB_16000_0.2(cap_complete)/')

im = np.array(Image.open(imlist[0]))  # 크기를 알기 위해 예시 하나 뽑기

m, n ,c= im.shape[0:3]  # 행렬의 길이 알기 및 채널 알기
imnbr = len(imlist)  # 전체 그림의 수를 알기
print(imnbr)


tmp = np.zeros((imnbr,67260))

i = 0

for im in imlist:

    # 원래 (행,렬, 모드)로 저장되는 array를 1차원으로 만들어 array로 저장하기
    tmp[i, :] = np.array(Image.open(im)).flatten().reshape(1, 67260)
    i += 1
    print(i)


print("tmp.shape=",tmp.shape)
print('tmp=',tmp)

x_value=tmp



n = x_value.shape[0] # how many rows we have in the dataset
chunk_size = 1000

pca = PCA()
pca.fit(tmp)
cumsum = np.cumsum(pca.explained_variance_ratio_) #분산의 설명량을 누적합
num_d = np.argmax(cumsum >= 0.95) + 1 #분산의 설명량이 95%이상 되는 차원의 수
print('Right number of IncrementalPCA is : ', num_d) #result : 160

pca_100 = IncrementalPCA(n_components=160)


#
for i in range(0, n//chunk_size):
    pca_result_160 = pca_100.partial_fit(tmp[i*chunk_size : (i+1)*chunk_size])

pca_result_160 = pca_result_160.fit_transform(x_value)

# cumsum = np.cumsum(pca_result_116.explained_variance_ratio_)
# num_d = np.argmax(cumsum >= 0.95) + 1

time_start = time.time()
X=np.array(pca_result_160)
tsne=TSNE(n_components=2,learning_rate=150,perplexity=30,angle=0.2,verbose=2).fit_transform(X)
print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))
tx,ty=tsne[:,0],tsne[:,1]
tx=(tx-np.min(tx))/(np.max(tx)-np.min(tx))
ty=(ty-np.min(ty))/(np.max(ty)-np.min(ty))





print('starting dbscan')

model = DBSCAN(eps = 4, min_samples=54)
predict = model.fit(tsne)
pd.Series(predict.labels_).value_counts()

y_pred = predict.labels_
print('prediction : ', y_pred)

with open('dbscanlabel_eps4_min54.txt', 'w') as f:
    for item in y_pred:
        f.write("%f\n" % item)

# Assign result to df
dataset = pd.DataFrame({'Column1': tsne[:,0],'Column2': tsne[:,1]})
dataset['cluster_num'] = pd.Series(predict.labels_)

width=4000
height=3000
max_dim=100

full_image=Image.new('RGBA',(width,height))
for img,x,y,z in zip(imlist,tx,ty,range(len(imlist))):
    tile=Image.open(img)
    rs=max(1,tile.width/max_dim,tile.height/max_dim)
    tile=tile.resize((int(tile.width/rs),int(tile.height/rs)),Image.ANTIALIAS)
    '''
    for i in range(max(dataset['cluster_num'])):
        if dataset.iloc[z]['cluster_num'] == i:
            flag = Image.new("RGB", (100, 2), (i,i+20,i+30))
            tile.paste(flag, (0, 0,100, 2))
        elif dataset.iloc[z]['cluster_num'] == -1:
            flag = Image.new("RGB", (100, 2), (0, 0, 0))
            tile.paste(flag, (0, 0, 100, 2))
    '''
    full_image.paste(tile,(int((width-max_dim)*x),int((height-max_dim)*y)),mask=tile.convert('RGBA'))


plt.figure(figsize=(16,12))
plt.imshow(full_image)

full_image.save("./tsne_image_final.png")