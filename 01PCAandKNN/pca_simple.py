#!/home/shannon/miniconda2/envs/cvbot/bin/python
#-*- coding: utf-8 -*-　　　←表示使用 utf-8 編碼
import cv2
import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA

n_samples=239
fig=plt.figure()
fig.suptitle('mean face & eigenfaces', fontsize=16)

# creating the dataset of (240, 2576) images,pixels
template=cv2.imread('1_1.png')
imgSet=cv2.imread('1_1.png',cv2.IMREAD_GRAYSCALE).reshape([1,2576])
for i,person in enumerate(range(40)):
    for j,image in enumerate(range(6)):
        img_name=str(i+1)+'_'+str(j+1)+'.png'
        img=cv2.imread(img_name,cv2.IMREAD_GRAYSCALE).reshape([1,2576])
        imgSet=np.append(imgSet,img,axis=0)
imgSet=np.delete(imgSet,0,0)
print type(imgSet),imgSet.shape

#calculating the meanface
'''meanPixels=[]
for j,pixel in enumerate(range(2576)):
    sum=0
    for i,image in enumerate(range(240)):
        sum+=imgSet[i,j]
    avg=sum/n_samples
    meanPixels.append(avg)
meanImg=np.array(meanPixels).reshape([56,46])
print type(meanImg),meanImg.shape
plt.imshow(meanImg,cmap = 'gray')
plt.show()'''
meanImg_row=np.mean(imgSet, axis=0)
meanImg2=meanImg_row.reshape([56,46])
print type(meanImg2),meanImg2.shape
subfig = plt.subplot("151")
subfig.set_title("meanface")
subfig.imshow(meanImg2,cmap = 'gray')

#converting 1 channel meanface to 3 channel
'''meanImg_3ch = np.zeros_like(template)
meanImg_3ch[:,:,0] = meanImg
meanImg_3ch[:,:,1] = 0
meanImg_3ch[:,:,2] = 0
cv2.namedWindow('cv2 printed Image', cv2.WINDOW_NORMAL)
cv2.imshow('cv2 printed Image',meanImg_3ch)
cv2.waitKey(0)
cv2.destroyAllWindows()'''

pca = PCA(n_components=n_samples,copy=True,whiten=True)
imgSet_centered = imgSet - meanImg_row #every row - mean row
imgSet_trans = pca.fit(imgSet_centered)
#print pca.explained_variance_ratio_
#print pca.singular_values_

# We center the data and compute the sample covariance matrix.
cov_matrix = np.dot(imgSet_centered.T, imgSet_centered) / n_samples
eigenvalues = pca.explained_variance_
for i,(eigenvalue, eigenvector) in enumerate(zip(eigenvalues, pca.components_)):
    if i <=3:
        #print(np.dot(eigenvector.T, np.dot(cov_matrix, eigenvector)))
        print type(eigenvector),eigenvector.shape
        print(eigenvalue)
        print(eigenvector)
        eigenface=eigenvector.reshape([56,46])
        subfig = plt.subplot("15"+str(i+2))
        subfig.set_title("eigenface"+str(i))
        subfig.imshow(eigenface,cmap = 'gray')
plt.show()
