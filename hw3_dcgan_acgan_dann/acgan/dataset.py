import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision
import glob
import os
import numpy as np
from PIL import Image
import constants as consts

class acganDataSet(Dataset):
    def __init__(self, root, transform=None):
        """ Intialize the hw2_train_val dataset """
        self.filenames = []
        self.root = root
        self.transform = transform

        # read filenames
        self.img_filenames = sorted(glob.glob(os.path.join(root,'train','*.png')))
        self.label_filename = glob.glob(os.path.join(root,'train.csv'))
        labels = []
        file = open(self.label_filename[0], 'r')
        for i,line in enumerate(file.readlines()):
            if i == 0: continue
            labels.append(int(float(line.strip().split(',')[10])))
        file.close()
        self.labels = labels

        self.len = len(self.img_filenames)

    def __getitem__(self, index):
        """ Get a sample from the dataset """
        image_fn, label_val = self.img_filenames[index], self.labels[index]

        """ INPUT: image part """
        image = Image.open(image_fn).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        #image.show()
        #print (image.format,image.size,image.mode)

        """ INPUT: label part """
        return image, label_val

    def __len__(self):
        """ Total number of samples in the dataset """
        return self.len

'''
trainset = acganDataSet(root='../../hw3_data/face/',
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                    ]))
print('# images in trainset:', len(trainset))
dataloader = torch.utils.data.DataLoader(trainset,
                            batch_size=consts.batch_size,
                            shuffle=True, num_workers=consts.workers)
dataiter = iter(dataloader)
images, labels = dataiter.next()
print('Image tensor in each batch:', images.shape, images.dtype)
print('Label tensor in each batch:', labels.shape, labels.dtype)
'''
