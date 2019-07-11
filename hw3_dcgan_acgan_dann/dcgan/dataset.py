import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision
import glob
import os
import numpy as np
from PIL import Image

class ganDataSet(Dataset):
    def __init__(self, root, transform=None):
        """ Intialize the hw2_train_val dataset """
        self.filenames = []
        self.root = root
        self.transform = transform

        # read filenames
        self.img_filenames = sorted(glob.glob(os.path.join(root, '*.png')))
        self.len = len(self.img_filenames)

    def __getitem__(self, index):
        """ Get a sample from the dataset """
        image_fn = self.img_filenames[index]
        """ INPUT: image part """
        image = Image.open(image_fn).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        #image.show()
        #print (image.format,image.size,image.mode)
        """ INPUT: label part """
        no_label = 0
        return image, no_label

    def __len__(self):
        """ Total number of samples in the dataset """
        return self.len
