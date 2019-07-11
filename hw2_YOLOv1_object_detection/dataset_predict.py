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
        self.img_filenames = sorted(glob.glob(os.path.join(root, '*.jpg')))
        digit_len = len(str(len(self.img_filenames)))
        label_filenames = []

        for img_fn in self.img_filenames:
            label_filenames.append(img_fn.split('/')[-1][:-4]+'.txt')
        #[label_filenames.append(("%0"+str(digit_len)+"d.txt") % i) for i in range(len(self.img_filenames))]
        self.label_filenames = label_filenames
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

        return image, label_fn

    def __len__(self):
        """ Total number of samples in the dataset """
        return self.len

    def decoder(self,pred,filter_lowScore):
        bboxes = []
        image_size = 512
        patch_num = 7
        patch_size = image_size/patch_num
        pred = pred.to('cpu')

        for m_ in range(patch_num): #patch(m_,n_)
            for n_ in range(patch_num):
                pred_26 = pred[m_,n_,:].numpy() #extract
                preclass_index = np.argmax(pred_26[10:])
                for n in range(2): #two box in one patch
                    bbox_conf_score = pred_26[n*5+4]
                    max_class_score = pred_26[10+preclass_index]
                    class_spec_conf_score = bbox_conf_score*max_class_score
                    if (class_spec_conf_score >= filter_lowScore):
                        width, height = pred_26[n*5+2], pred_26[n*5+3]
                        dx, dy = pred_26[n*5], pred_26[n*5+1]
                        x1 = max(0, int(patch_size*(n_+dx)-image_size*width/2))
                        x2 = min(image_size, int(patch_size*(n_+dx)+image_size*width/2))
                        y1 = max(0, int(patch_size*(m_+dy)-image_size*height/2))
                        y2 = min(image_size, int(patch_size*(m_+dy)+image_size*height/2))
                        bboxes.append([x1,y1,x2,y2,bbox_conf_score,preclass_index,class_spec_conf_score])
        return np.array(bboxes)
