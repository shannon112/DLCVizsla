#!/home/shannon/miniconda2/envs/cvbot/bin/python
#-*- coding: utf-8 -*-　　　←表示使用 utf-8 編碼
import cv2
import numpy as np
import random
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn import cluster

# creating the images dataset of (2000, 64, 64, 3) as image,pixels,pixels,channel
categories=["banana","fountain","reef","tractor"]
ans_list=[]
images=[]
for i,category in enumerate(categories):
    for j,image in enumerate(range(500)):
        j+=1
        num=("%03d" % j)
        img_name="{}/{}_{}.JPEG".format(category,category,num)
        img=cv2.imread(img_name)
        images.append(img)
        ans_list.append(i)
imgSet=np.array(images)
ansSet=np.array(ans_list)
print "imgSet ",imgSet.shape
print "ansSet", ansSet.shape
print ""

# creating the patches dataset of  (2000, 16, 16, 16, 3) as image, patch_num, pixels, pixels, channel
grids=[0,16,32,48]
imagePatches=[]
for i in range(2000):
    image=imgSet[i,:,:,:]
    patches=[]
    for col,col_grid in enumerate(grids):
        for row,row_grid in enumerate(grids):
            patch=image[col_grid:col_grid+16,row_grid:row_grid+16,:]
            patches.append(patch)
    imagePatches.append(patches)
imgPthSet=np.array(imagePatches)
print "imgPthSet ",imgPthSet.shape
imgPthSet768=imgPthSet.reshape([2000,16,768])
print "imgPthSet768 ",imgPthSet768.shape
print ""

# separating training set and testing set (1500, 16, 16, 16, 3) (500, 16, 16, 16, 3)
imgPthSet_testing_list=[]
imgPthSet_training_list=[]
ansSet_training_list=[]
ansSet_testing_list=[]
for i in range(2000):
    if (i <375) or (i>=500 and i<875) or (i>=1000 and i<1375) or (i>=1500 and i<1875):
        imgPthSet_training_list.append(imgPthSet768[i,:,:])
        ansSet_training_list.append(ansSet[i])
    else:
        imgPthSet_testing_list.append(imgPthSet768[i,:,:])
        ansSet_testing_list.append(ansSet[i])
imgPthSet_test=np.array(imgPthSet_testing_list).reshape([8000,768])
imgPthSet_train=np.array(imgPthSet_training_list).reshape([24000,768])
ansSet_testing=np.array(ansSet_testing_list)
ansSet_training=np.array(ansSet_training_list)
print "imgPthSet_test ",imgPthSet_test.shape
print "imgPthSet_train ",imgPthSet_train.shape
print "ansSet_training", ansSet_training.shape
print "ansSet_testing", ansSet_testing.shape
print ""

# quickly view of the patches in one image
fig=plt.figure().suptitle('quickly view of patches in a image', fontsize=16)
for i,patch in enumerate(range(16)):
    subfig = plt.subplot(4,4,i+1)
    subfig.set_title(i)
    randPatch=np.array(imgPthSet[0,i,:,:,:])
    randPatch_rgb=randPatch[:,:,::-1]
    subfig.imshow(randPatch_rgb)

# quickly view of random picked images from 4 categories, each image show 3 patches
fig2=plt.figure().suptitle('quickly view of a few patches in each categories', fontsize=16)
for i,category in enumerate(categories):
    image_num=random.randint(500*i,500*(i+1)-1)
    subfig = plt.subplot(4,4,1+i*4)
    subfig.set_title("original from "+category+str(image_num-500*i+1))
    origin_image=imgSet[image_num,:,:,:]
    origin_image_rgb=origin_image[:,:,::-1]
    subfig.imshow(origin_image_rgb)

    for j,patch_num in enumerate([1,6,11]):
        #patch_num=random.randint(0,15)
        subfig = plt.subplot(4,4,1+i*4+j+1)
        subfig.set_title("patch_num "+str(patch_num))
        randPatch=np.array(imgPthSet[image_num,patch_num,:,:,:])
        randPatch_rgb=randPatch[:,:,::-1]
        subfig.imshow(randPatch_rgb)

# k-means clustering
'''kmeans_fit = cluster.KMeans(n_clusters = 15, max_iter=5000).fit(imgPthSet_train)
clustered_labels=kmeans_fit.labels_
clustered_centers=kmeans_fit.cluster_centers_
print "clustered_labels ",clustered_labels.shape
print "clustered_centers ",clustered_centers.shape
f_l = open("clustered_labels24000", "w")
f_c = open("clustered_centers15", "w")
np.savetxt(f_l, np.array(clustered_labels))
np.savetxt(f_c, np.array(clustered_centers))'''
clustered_labels = np.array(np.loadtxt("clustered_labels24000")).astype(np.int8)
clustered_centers = np.array(np.loadtxt("clustered_centers15"))
print "clustered_labels ",clustered_labels.shape
print "clustered_centers ",clustered_centers.shape
print ""

# PCA construct 3d training features
#calculating the mean
meanPth=np.mean(imgPthSet_train, axis=0)
print "meanPth ", meanPth.shape #get one mean patch
#calculating PCA
n_samples=3
pca = PCA(n_samples)
imgPthSet_train_centered = imgPthSet_train- meanPth #every row - mean row
pca_fit = pca.fit(imgPthSet_train_centered) #training
eigenspace=pca_fit.transform(imgPthSet_train_centered)
print "eigenspace ",eigenspace.shape
cov_matrix = np.dot(imgPthSet_train_centered.T, imgPthSet_train_centered) / n_samples# We center the data and compute the sample covariance matrix.
print "cov_matrix ",cov_matrix.shape
eigenvalues = pca_fit.explained_variance_
print "eigenvalues ",eigenvalues.shape
eigenvectors = pca_fit.components_
print "eigenvectors",eigenvectors.shape
explained=np.sum(pca_fit.explained_variance_ratio_)
print "explained",explained
clustered_centers_C=clustered_centers - meanPth
clustered_centers_D=pca_fit.transform(clustered_centers_C)
print "clustered_centers_D",clustered_centers_D.shape
imgPthSet_test_C = imgPthSet_test - meanPth
imgPthSet_test_D=pca_fit.transform(imgPthSet_test_C)
print "imgPthSet_test_D",imgPthSet_test_D.shape
print ""

# 3d plot of k-means clusting result after PCA to 3d
'''
fig3=plt.figure()
ax=fig3.add_subplot(111,projection='3d')
ax.set_title('3d plot of clusting imgPthSet_train to 6 clusters among 15')
colors = ["gray","blue","purple","red","lime","orange"]
for i,clustered_label in enumerate(clustered_labels):
    if clustered_label==3:
        x=eigenspace[i,:][0]; y=eigenspace[i,:][1]; z=eigenspace[i,:][2]
        ax.scatter(x,y,z,facecolor=(0,0,0,0),s=0.1,edgecolor=colors[0],marker='o')
    elif clustered_label==4:
        x=eigenspace[i,:][0]; y=eigenspace[i,:][1]; z=eigenspace[i,:][2]
        ax.scatter(x,y,z,facecolor=(0,0,0,0),s=0.1,edgecolor=colors[1],marker='o')
    elif clustered_label==5:
        x=eigenspace[i,:][0]; y=eigenspace[i,:][1]; z=eigenspace[i,:][2]
        ax.scatter(x,y,z,facecolor=(0,0,0,0),s=0.1,edgecolor=colors[2],marker='o')
    elif clustered_label==6:
        ax.scatter(x,y,z,facecolor=(0,0,0,0),s=0.1,edgecolor=colors[3],marker='o')
        x=eigenspace[i,:][0]; y=eigenspace[i,:][1]; z=eigenspace[i,:][2]
    elif clustered_label==7:
        x=eigenspace[i,:][0]; y=eigenspace[i,:][1]; z=eigenspace[i,:][2]
        ax.scatter(x,y,z,facecolor=(0,0,0,0),s=0.1,edgecolor=colors[4],marker='o')
    elif clustered_label==8:
        x=eigenspace[i,:][0]; y=eigenspace[i,:][1]; z=eigenspace[i,:][2]
        ax.scatter(x,y,z,facecolor=(0,0,0,0),s=0.1,edgecolor=colors[5],marker='o')
for i, num_center in enumerate(range(3,9)):
        x=clustered_centers_D[num_center,:][0]
        y=clustered_centers_D[num_center,:][1]
        z=clustered_centers_D[num_center,:][2]
        ax.scatter(x,y,z,s=100,c=colors[i])
'''
# transfer images from one image 16patches*3d to 15BoW
# calculating norm of one patch to 15centers then do the ^-1 so closest score is largest
# normalize 15centers score in one patch
pths_24000_15d_list=[]
for iPth in range(24000):
    pth_15d=[]
    for jCen in range(15):
        diif=1/np.linalg.norm(eigenspace[iPth,:]-clustered_centers_D[jCen,:])
        pth_15d.append(diif)
    sum=np.sum(pth_15d)
    pth_15d=pth_15d/sum
    pths_24000_15d_list.append(pth_15d)
pths_24000_15d=np.array(pths_24000_15d_list)
print "pths_24000_15d",pths_24000_15d.shape

pths_8000_15d_list=[]
for iPth in range(8000):
    pth_15d=[]
    for jCen in range(15):
        diif=1/np.linalg.norm(imgPthSet_test_D[iPth,:]-clustered_centers_D[jCen,:])
        pth_15d.append(diif)
    sum=np.sum(pth_15d)
    pth_15d=pth_15d/sum
    pths_8000_15d_list.append(pth_15d)
pths_8000_15d=np.array(pths_8000_15d_list)
print "pths_8000_15d",pths_8000_15d.shape
print ""

# max pooling from 16patches*15centers to 15centers max score in each patches
firstPth=0
lastPth=16 #would +16 at each loop
images_1500_15d_list=[]
for iImg in range(1500):
    image_15d=[]
    for iCen in range(15):
        maxVal=max(pths_24000_15d.T[iCen,firstPth:lastPth])
        image_15d.append(maxVal)
    images_1500_15d_list.append(image_15d)
    firstPth+=16
    lastPth+=16
images_1500_15d=np.array(images_1500_15d_list)
print "images_1500_15d",images_1500_15d.shape

firstPth=0
lastPth=16 #would +16 at each loop
images_500_15d_list=[]
for iImg in range(500):
    image_15d=[]
    for iCen in range(15):
        maxVal=max(pths_8000_15d.T[iCen,firstPth:lastPth])
        image_15d.append(maxVal)
    images_500_15d_list.append(image_15d)
    firstPth+=16
    lastPth+=16
images_500_15d=np.array(images_500_15d_list)
print "images_500_15d",images_500_15d.shape
print ""

# plot histogram of 4 images from different category
fig4=plt.figure().suptitle('quickly view of 4 images in each categories in form of BoW', fontsize=16)
for i,category in enumerate(categories):
    y_performance = images_1500_15d[i*375,:]
    x_pos = np.arange(1,len(y_performance)+1)
    x_name = x_pos
    subfig = plt.subplot(1,4,i+1)
    subfig.set_title(category+"001")
    subfig.bar(x_pos, y_performance, align='center', alpha=0.5)
    plt.xticks(x_pos, x_name)
    plt.ylabel('similarity')

# performing knn on BoW
knn = KNeighborsClassifier ( n_neighbors = 5 )
knn.fit(images_1500_15d,ansSet_training)
predict_ans_list=knn.predict(images_500_15d)
print accuracy_score(ansSet_testing,predict_ans_list)
print predict_ans_list
plt.show()
