import glob
import numpy as np
import os

classes_dict = {}
classes_list = ['plane', 'ship', 'storage-tank', 'baseball-diamond',
 'tennis-court', 'basketball-court', 'ground-track-field',
 'harbor', 'bridge', 'small-vehicle', 'large-vehicle', 'helicopter',
 'roundabout', 'soccer-ball-field', 'swimming-pool', 'container-crane']
for index,class_ in zip(range(16),classes_list):
    classes_dict[class_]=index

classes_counter=[0]*16
root='../hw2_train_val/train15000/'
label_filenames = sorted(glob.glob(os.path.join(root,'labelTxt_hbb', '*.txt')))
for filename in label_filenames:
    labels=np.genfromtxt(filename, dtype='str')
    if np.size(labels) == 10:
        label=labels
        class_ = classes_dict[label[8]]
        classes_counter[class_]+=1
    else:
        for label in labels:
            class_ = classes_dict[label[8]]
            classes_counter[class_]+=1
print("trainset",classes_counter)

classes_counter=[0]*16
root='../hw2_train_val/val1500/'
label_filenames = sorted(glob.glob(os.path.join(root,'labelTxt_hbb', '*.txt')))
for filename in label_filenames:
    labels=np.genfromtxt(filename, dtype='str')
    if np.size(labels) == 10:
        label=labels
        class_ = classes_dict[label[8]]
        classes_counter[class_]+=1
    else:
        for label in labels:
            class_ = classes_dict[label[8]]
            classes_counter[class_]+=1
print("testset",classes_counter)
