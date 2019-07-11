import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision
import glob
import os
import numpy as np
from PIL import Image
#[1]
# Creating a custom dataset
class hw2DataSet(Dataset):
    def __init__(self, root, transform=None):
        """ Intialize the hw2_train_val dataset """
        self.filenames = []
        self.root = root
        self.transform = transform
        self.classes = {} #dict

        # read filenames
        self.img_filenames = sorted(glob.glob(os.path.join(root,'images', '*.jpg')))
        self.label_filenames = sorted(glob.glob(os.path.join(root,'labelTxt_hbb', '*.txt')))
        self.len = len(self.img_filenames)

        # define classes
        classes_dict = {}
        classes_list = ['plane', 'ship', 'storage-tank', 'baseball-diamond',
         'tennis-court', 'basketball-court', 'ground-track-field',
         'harbor', 'bridge', 'small-vehicle', 'large-vehicle', 'helicopter',
         'roundabout', 'soccer-ball-field', 'swimming-pool', 'container-crane']
        for index,class_ in zip(range(16),classes_list):
            classes_dict[class_]=index
        self.classes = classes_dict
        self.classesL = classes_list

    def __getitem__(self, index):
        """ Get a sample from the dataset """
        (image_fn, label_fn) = self.img_filenames[index], self.label_filenames[index]
        """ INPUT: image part """
        image = Image.open(image_fn).convert('RGB')
        image = image.resize((448,448))
        if self.transform is not None:
            image = self.transform(image)
        #image.show()
        #print (image.format,image.size,image.mode)

        """ OUTPUT: label part """
        labels=np.genfromtxt(label_fn, dtype='str')
        label_torch = torch.zeros((7,7,26))
        if np.size(labels) == 10:
            label=labels; label_torch = self.encoder(label_torch, label)
        else:
            for label in labels: label_torch = self.encoder(label_torch, label)
        return image, label_torch

    def __len__(self):
        """ Total number of samples in the dataset """
        return self.len

    def encoder(self,label_torch,label):
        # read label
        v1_ = torch.Tensor([float(label[0]),float(label[1])]) / 512.
        #v2_ = torch.Tensor([float(label[2]),float(label[3])]) / 512.
        v3_ = torch.Tensor([float(label[4]),float(label[5])]) / 512.
        #v4_ = torch.Tensor([float(label[6]),float(label[7])]) / 512.
        class_ = self.classes[label[8]]

        # transform label to useful data
        patch_size = 1. / 7
        center = (v1_ + v3_) / 2
        width_height = v3_ - v1_
        #width = max(v1_[0],v2_[0],v3_[0],v4_[0])-min(v1_[0],v2_[0],v3_[0],v4_[0])
        #height = max(v1_[1],v2_[1],v3_[1],v4_[1])-min(v1_[1],v2_[1],v3_[1],v4_[1])
        class_array = torch.zeros(16); class_array[class_]=1

        patch_position = ( center / patch_size ).floor()
        patch_i, patch_j = int(patch_position[1]), int(patch_position[0])
        patch_origin = patch_position * patch_size
        center = (center - patch_origin) / patch_size

        # filling the 7x7x26 matrix
        # box#1 offer for the first one comes, box#2 offer for the last one comes, class belong to the first one comes
        #if float(label_torch[patch_i][patch_j][2]) == 0.: # check if width == 0
        label_torch[patch_i][patch_j][0:2] = center
        label_torch[patch_i][patch_j][2:4] = width_height
        label_torch[patch_i][patch_j][4] = 1
        #label_torch[patch_i][patch_j][10+class_] = 1
        label_torch[patch_i][patch_j][10:26] = class_array
        #else:
        label_torch[patch_i][patch_j][5:7] = center
        label_torch[patch_i][patch_j][7:9] = width_height
        label_torch[patch_i][patch_j][9] = 1

        return label_torch
