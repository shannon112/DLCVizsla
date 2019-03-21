#!/home/shannon/miniconda2/envs/cvbot/bin/python
#-*- coding: utf-8 -*-　　　←表示使用 utf-8 編碼
import cv2
import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA

n_samples=239
fig=plt.figure()
fig.suptitle('reconstructed images', fontsize=16)

# creating the dataset of (240, 2576) images,pixels
template=cv2.imread('1_1.png')
imgSet=cv2.imread('1_1.png',cv2.IMREAD_GRAYSCALE).reshape([1,2576])
for i,person in enumerate(range(40)):
    for j,image in enumerate(range(6)):
        img_name=str(i+1)+'_'+str(j+1)+'.png'
        img=cv2.imread(img_name,cv2.IMREAD_GRAYSCALE).reshape([1,2576])
        imgSet=np.append(imgSet,img,axis=0)
imgSet=np.delete(imgSet,0,0)
print "dataset ",type(imgSet),imgSet.shape

#calculating the meanface
meanImg_row=np.mean(imgSet, axis=0)
meanImg=meanImg_row.reshape([56,46])
print "meanface ",type(meanImg_row),meanImg_row.shape

#calculating PCA
pca = PCA(n_samples)
#pca = PCA(n_components=n_samples,copy=True,whiten=False)
imgSet_centered = imgSet- meanImg_row #every row - mean row
print "imgSet_centered ",type(imgSet_centered),imgSet_centered.shape ;print imgSet_centered
imgSet_fit = pca.fit(imgSet_centered) #training
eigenspace=imgSet_fit.transform(imgSet_centered) #240,239
print "eigenspace ",type(eigenspace),eigenspace.shape ;print eigenspace
cov_matrix = np.dot(imgSet_centered.T, imgSet_centered) / n_samples# We center the data and compute the sample covariance matrix.
print "cov_matrix ",type(cov_matrix),cov_matrix.shape
eigenvalues = imgSet_fit.explained_variance_ #239,
print "eigenvalues ",type(eigenvalues),eigenvalues.shape
eigenvectors = imgSet_fit.components_ #239,2576
print "eigenvectors",type(eigenvectors),eigenvectors.shape; print eigenvectors
explained=np.sum(imgSet_fit.explained_variance_ratio_)
print "explained",explained

#method 1
'''reconface=np.zeros([1,2576])
for i in range(239):
    if i <=239:
        eigenface=eigenvectors[i,:]*eigenspace[0,i]
        reconface+=eigenface
        #reconface_img=reconface.reshape([56,46])
        #plt.imshow(reconface_img,cmap = 'gray')
        #plt.show()
    else:
        break
reconface+=meanImg_row
reconface_img=reconface.reshape([56,46])
#plt.imshow(reconface_img,cmap = 'gray')
#plt.show()
'''

#method 2 bulit-in function perfect transform for 240 images
reconface_prefect=imgSet_fit.inverse_transform(eigenspace)
reconface_prefect_img=reconface_prefect[0,:].reshape([56,46])
#plt.imshow(reconface_prefect_img,cmap = 'gray')
#plt.show()

#method 3 for 1_1 image reconstruction
target=imgSet[0,:].reshape([1,2576])
target_img=target.reshape([56,46])
subfig = plt.subplot("161")
subfig.set_title("original face 1_1.png")
subfig.imshow(target_img,cmap = 'gray')
print "target ",type(target),target.shape
print target

reconVectors=[3,45,140,229,239]
for i,num in enumerate(reconVectors):
    #reconstructing image
    reconface=np.dot(eigenspace[0,:num],imgSet_fit.components_[:num,:])
    reconface+=meanImg_row

    #calculating MSE
    reconface_img_255=cv2.imread("reconface_"+str(num)+".png",cv2.IMREAD_GRAYSCALE).reshape([1,2576])
    print "reconface_img_255 ",type(reconface_img_255),reconface_img_255.shape
    print reconface_img_255
    error=np.subtract(reconface.astype(np.int16) , target.astype(np.int16))
    print error
    error_square=error**2
    print error_square
    mse=np.sum(error_square)/2576
    print mse
    reconface_img=reconface.reshape([56,46])

    #plot images
    subfig = plt.subplot("16"+str(i+2))
    subfig.set_title("reconstructed by"+str(num)+" eigenvectors\nMSE: "+str(mse))
    subfig.imshow(reconface_img,cmap = 'gray')
    plt.imsave("reconface_"+str(num)+".png",reconface_img,cmap = 'gray')
plt.show()
