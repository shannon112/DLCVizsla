#!/home/shannon/miniconda2/envs/cvbot/bin/python
#-*- coding: utf-8 -*-　　　←表示使用 utf-8 編碼
import cv2
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

fig=plt.figure()
fig.suptitle('reconstructed images', fontsize=16)

# creating the dataset of (240, 2576) images,pixels
# for training
template=cv2.imread('1_1.png')
ans_list=[]
imgSet=cv2.imread('1_1.png',cv2.IMREAD_GRAYSCALE).reshape([1,2576])
for i,person in enumerate(range(40)):
    for j,image in enumerate(range(6)):
        img_name=str(i+1)+'_'+str(j+1)+'.png'
        ans_list.append(i+1)
        img=cv2.imread(img_name,cv2.IMREAD_GRAYSCALE).reshape([1,2576])
        imgSet=np.append(imgSet,img,axis=0)
imgSet=np.delete(imgSet,0,0)
print "ans_list",ans_list
print "imgSet ",type(imgSet),imgSet.shape
# for testing
ans_list_test=[]
imgSet_test=cv2.imread('1_7.png',cv2.IMREAD_GRAYSCALE).reshape([1,2576])
for i,person in enumerate(range(40)):
    for j,image in enumerate(range(4)):
        img_name=str(i+1)+'_'+str(j+7)+'.png'
        ans_list_test.append(i+1)
        img=cv2.imread(img_name,cv2.IMREAD_GRAYSCALE).reshape([1,2576])
        imgSet_test=np.append(imgSet_test,img,axis=0)
imgSet_test=np.delete(imgSet_test,0,0)
print "ans_list_test",ans_list_test
print "imgSet_test ",type(imgSet_test),imgSet_test.shape

#calculating the meanface
meanImg_row=np.mean(imgSet, axis=0)
meanImg=meanImg_row.reshape([56,46])
print "meanface ",type(meanImg_row),meanImg_row.shape

#calculating PCA
n_samples=239 # 240 images - 1 mean image
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
print "explained",explained

imgSet_test_centered = imgSet_test- meanImg_row #every row - mean row
eigenspace_test= imgSet_fit.transform(imgSet_test_centered)
#eigenspace_test= np.dot(imgSet_test,eigenvectors.T)
print "eigenspace_test ",type(eigenspace_test),eigenspace_test.shape ;print eigenspace_test

#reconstructing data by n components
'''reconImgSet_list=[]
for i,num in enumerate(reconVectors):
    reconface=np.dot(eigenspace[:,:num],imgSet_fit.components_[:num,:])
    reconface+=meanImg_row
    print "reconface ",type(reconface),reconface.shape
    reconImgSet_list.append(reconface)
reconImgSet=np.array(reconImgSet_list)
reconImgSet=reconImgSet.reshape(3,240,56,46)
print "reconImgSet ",type(reconImgSet),reconImgSet.shape'''
#reconImgSet  <type 'numpy.ndarray'> (3, 240, 56, 46)


#performing knn with 3CV on diff n & k (using n dim 240images)
nReconVectors=[3,45,140]
params = {'n_neighbors':[1,3,5]}
knn = KNeighborsClassifier()
clf = GridSearchCV(knn, params,cv=3)

res = dict()
for i,n in enumerate(nReconVectors):
    clf.fit(eigenspace[:,:n],ans_list)
    res['n='+str(n)] = np.array(clf.cv_results_['mean_test_score'])
res = pd.DataFrame.from_dict(res,orient='index')
res.columns = ['k=1','k=3','k=5']
print(res)

knn_t = KNeighborsClassifier ( n_neighbors = 1 )
knn_t.fit(eigenspace[:,:45],ans_list)
predict_ans_list=knn_t.predict(eigenspace_test[:,:45])
print accuracy_score(ans_list_test,predict_ans_list)
print predict_ans_list
    #plot images
    #subfig = plt.subplot("16"+str(i+2))
    #subfig.set_title("reconstructed by"+str(num)+" eigenvectors\nMSE: "+str(mse))
    #subfig.imshow(reconface_img,cmap = 'gray')
#plt.show()
