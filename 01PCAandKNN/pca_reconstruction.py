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

# creating the dataset of (240, 2576) images,pixels
# for training
images_list=[]
ans_list=[]
for i,person in enumerate(range(40)):
    for j,image in enumerate(range(6)):
        img_name="data/"+str(i+1)+'_'+str(j+1)+".png"
        img=cv2.imread(img_name,cv2.IMREAD_GRAYSCALE)
        ans_list.append(i+1)
        images_list.append(img)
images=np.array(images_list)
print "images ",images.shape
imgSet=images.reshape(240,2576)
print "imgSet ",imgSet.shape
print "ans_list",ans_list
# for testing
ans_list_test=[]
images_list_test=[]
for i,person in enumerate(range(40)):
    for j,image in enumerate(range(4)):
        img_name="data/"+str(i+1)+'_'+str(j+7)+".png"
        img=cv2.imread(img_name,cv2.IMREAD_GRAYSCALE)
        ans_list_test.append(i+1)
        images_list_test.append(img)
images_test=np.array(images_list_test)
print "images_test ",images_test.shape
imgSet_test=images_test.reshape(160,2576)
print "imgSet_test ",imgSet_test.shape
print "ans_list_test",ans_list_test
print ""

# calculating the meanface
meanImg_row=np.mean(imgSet, axis=0)
meanImg=meanImg_row.reshape([56,46])
print "meanface ",meanImg_row.shape
fig=plt.figure()
fig.suptitle('meanface', fontsize=16)
plt.imshow(meanImg,cmap = 'gray')
# calculating PCA
n_samples=239
pca = PCA(n_samples)
#pca = PCA(n_components=n_samples,copy=True,whiten=False)
imgSet_centered = imgSet- meanImg_row #every row - mean row
print "imgSet_centered ",imgSet_centered.shape
imgSet_fit = pca.fit(imgSet_centered) #training
eigenspace=imgSet_fit.transform(imgSet_centered) #240,239
print "eigenspace ",eigenspace.shape
cov_matrix = np.dot(imgSet_centered.T, imgSet_centered) / n_samples# We center the data and compute the sample covariance matrix.
print "cov_matrix ",cov_matrix.shape
eigenvalues = imgSet_fit.explained_variance_ #239,
print "eigenvalues ",eigenvalues.shape
eigenvectors = imgSet_fit.components_ #239,2576
print "eigenvectors",eigenvectors.shape
explained=np.sum(imgSet_fit.explained_variance_ratio_)
print "explained",explained
imgSet_test_centered = imgSet_test- meanImg_row #every row - mean row
eigenspace_test= imgSet_fit.transform(imgSet_test_centered)
print "eigenspace_test ",eigenspace_test.shape
print ""

# ploting first four eigenfaces
fig2=plt.figure()
fig2.suptitle('ploting first four eigenfaces', fontsize=16)
for i,(eigenvalue, eigenvector) in enumerate(zip(eigenvalues, eigenvectors)):
    if i <=3:
        eigenface=eigenvector.reshape([56,46])
        subfig = plt.subplot("14"+str(i+1))
        subfig.set_title("eigenface"+str(i))
        subfig.imshow(eigenface,cmap = 'gray')

# bulit-in function perform inverse_transform of 240 images from 239d to 2576d
reconface_prefect=imgSet_fit.inverse_transform(eigenspace)
reconface_prefect_img=reconface_prefect[0,:].reshape([56,46])
fig3=plt.figure()
fig3.suptitle('Using 239 eigenvectors\nreconstructed\nby inverse_transform', fontsize=16)
plt.imshow(reconface_prefect_img,cmap = 'gray')

# 1_1 image reconstruction using a few eigenfaces
fig4=plt.figure()
fig4.suptitle('reconstructed images', fontsize=16)
target=imgSet[0,:].reshape([1,2576])
target_img=target.reshape([56,46])
subfig = plt.subplot("161")
subfig.set_title("original face 1_1.png")
subfig.imshow(target_img,cmap = 'gray')
print "target ",target.shape
reconVectors=[3,45,140,229,239]
for i,num in enumerate(reconVectors):
    #reconstructing image
    reconface=np.dot(eigenspace[0,:num],eigenvectors[:num,:])
    reconface+=meanImg_row
    #calculating MSE
    error=np.subtract(reconface.astype(np.int16) , target.astype(np.int16))
    error_square=error**2
    mse=np.sum(error_square)/2576
    #plot images
    reconface_img=reconface.reshape([56,46])
    subfig = plt.subplot("16"+str(i+2))
    subfig.set_title("Using "+str(num)+" eigenvectors\nreconstructed\nMSE: "+str(mse))
    subfig.imshow(reconface_img,cmap = 'gray')
    #plt.imsave("reconface_"+str(num)+".png",reconface_img,cmap = 'gray')

# evaluating knn with 3CV on diff n & k (using n dim 240images)
nReconVectors=[3,45,140] #n
params = {'n_neighbors':[1,3,5]} #k
knn = KNeighborsClassifier()
clf = GridSearchCV(knn, params,cv=3)
print "      k=1 ,        k=3 ,        k=5"
for i,n in enumerate(nReconVectors):
    clf.fit(eigenspace[:,:n],ans_list)
    results= np.array(clf.cv_results_['mean_test_score'])
    print "n =" , n , results
print ""
'''res = dict()
for i,n in enumerate(nReconVectors):
    clf.fit(eigenspace[:,:n],ans_list)
    res['n='+str(n)] = np.array(clf.cv_results_['mean_test_score'])
res = pd.DataFrame.from_dict(res,orient='index')
res.columns = ['k=1','k=3','k=5']'''

# Using k-nn trainig by k=1, n=45, and then test
knn_t = KNeighborsClassifier ( n_neighbors = 1 )
knn_t.fit(eigenspace[:,:45],ans_list)
predict_ans_list=knn_t.predict(eigenspace_test[:,:45])
print "accuracy_score", accuracy_score(ans_list_test,predict_ans_list)
print "predict_ans_list", predict_ans_list

plt.show()
