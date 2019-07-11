from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import os
import random
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms

import trainer
from dataset import GetLoader
import constants as consts

os.makedirs("models", exist_ok=True)

# define target and source
source_dataset_name = 'usps'
target_dataset_name = 'mnistm'
source_image_root = os.path.join('../../hw3_data/digits/', source_dataset_name)
target_image_root = os.path.join('../../hw3_data/digits/', target_dataset_name)

# Setting random seed
manualSeed = random.randint(1, 10000)
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)
torch.cuda.manual_seed_all(manualSeed)

# GPU/CPU flags
cudnn.benchmark = True

# Creating data loaders
mean = np.array([0.44, 0.44, 0.44])
std = np.array([0.19, 0.19, 0.19])

transform = transforms.Compose([transforms.Resize(32),
                                transforms.ToTensor(),
                                transforms.Normalize(mean,std)])

transform2 = transforms.Compose([transforms.Resize(32),
                                transforms.Grayscale(),
                                transforms.ToTensor(),
                                transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
                                transforms.Normalize(mean,std)])

#########################
# if training mnistm to svhn please use transform2 on target_val & target_train
#########################

source_train = GetLoader(
    img_root=os.path.join(source_image_root,"train"),
    label_path=os.path.join(source_image_root, 'train.csv'),
    transform=transform)

target_val = GetLoader(
    img_root=os.path.join(target_image_root,"test"),
    label_path=os.path.join(target_image_root, 'test.csv'),
    transform=transform)

target_train = GetLoader(
    img_root=os.path.join(target_image_root,"train"),
    label_path=os.path.join(target_image_root, 'train.csv'),
    transform=transform)

source_trainloader = torch.utils.data.DataLoader(source_train, batch_size=consts.batch_size, shuffle=True, num_workers=consts.workers, drop_last=True)
target_valloader = torch.utils.data.DataLoader(target_val, batch_size=consts.batch_size, shuffle=False, num_workers=consts.workers, drop_last=False)
targetloader = torch.utils.data.DataLoader(target_train, batch_size=consts.batch_size, shuffle=True, num_workers=consts.workers, drop_last=True)

# Training
GTA_trainer = trainer.GTA(mean, std, source_trainloader, target_valloader, targetloader)
GTA_trainer.train()
