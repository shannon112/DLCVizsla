# usage:
# python predict.py ./ weight_99200_new_64.pth
import os
import sys
import numpy as np
import random
import matplotlib.pyplot as plt

import torch
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable

# import self-made function
from model import Generator
import constants as consts
from dataset import acganDataSet

# read target dir
output_dir, model_fn = sys.argv[1], sys.argv[2]


######################################################################
# using cuda
######################################################################
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cuda = True if torch.cuda.is_available() else False
print('Device used:', device)


######################################################################
# Random
######################################################################
# Set random seem for reproducibility
manualSeed = 9265 #550 4824
#manualSeed = random.randint(1, 10000) # use if you want new results
np.random.seed(manualSeed)
torch.manual_seed(manualSeed)
print("Random Seed: ", manualSeed)


######################################################################
# load_checkpoint
######################################################################
def load_checkpoint(checkpoint_path, model,device):
    state = torch.load(checkpoint_path, map_location="cuda") # for cuda
    #state = torch.load(checkpoint_path, map_location=device) #for cpu
    model.load_state_dict(state['state_dictG'])
    print('model loaded from %s' % checkpoint_path)


######################################################################
# loading model
######################################################################
netG = Generator().to(device)
load_checkpoint(model_fn,netG,device)


######################################################################
# predict
######################################################################
FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

fake_imgs = None
with torch.no_grad():
    """Saves a grid of generated faces from normal(0) to smile(1)"""
    n_row = 10
    # Sample noise
    z = Variable(FloatTensor(np.random.normal(0, 1, (2*n_row, consts.nz))))
    labels = np.array([0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1])
    labels = Variable(LongTensor(labels))
    gen_imgs = netG(z, labels)
    vutils.save_image(gen_imgs.data, os.path.join(output_dir,"fig2_2.jpg"), nrow=n_row, normalize=True)
    fake_imgs = gen_imgs

'''
######################################################################
# Dataset and DataLoader
######################################################################
trainset = acganDataSet(root='../../hw3_data/face/',
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                    ]))
print('# images in trainset:', len(trainset))
dataloader = torch.utils.data.DataLoader(trainset,
                            batch_size=consts.batch_size,
                            shuffle=True, num_workers=consts.workers)

# Grab a batch of real images from the dataloader
dataiter = iter(dataloader)
images, labels = dataiter.next()
print('Image tensor in each batch:', images.shape, images.dtype)
print('Label tensor in each batch:', labels.shape, labels.dtype)


######################################################################
# plotting comparison
######################################################################
smile_img = []
normal_img = []
for image,label in zip(images,labels):
    if label == 0: normal_img.append(image)
    elif label == 1: smile_img.append(image)
# Plot the real images
plt.figure(figsize=(15,15))
plt.subplot(4,1,1).axis("off")
plt.title("Real Images non-smile")
plt.imshow(np.transpose(vutils.make_grid(normal_img[:10],nrow=10,normalize=True).cpu(),(1,2,0)))
# Plot the real images
plt.subplot(4,1,2).axis("off")
plt.title("Real Images smile")
plt.imshow(np.transpose(vutils.make_grid(smile_img[:10],nrow=10,normalize=True).cpu(),(1,2,0)))
# Plot the fake images
plt.subplot(4,1,3).axis("off")
plt.title("Fake Images non-smile")
plt.imshow(np.transpose(vutils.make_grid(fake_imgs.data[:10],nrow=10,normalize=True).cpu(),(1,2,0)))
# Plot the fake images
plt.subplot(4,1,4).axis("off")
plt.title("Fake Images smile")
plt.imshow(np.transpose(vutils.make_grid(fake_imgs.data[10:],nrow=10,normalize=True).cpu(),(1,2,0)))
plt.savefig('result/RealVsFake.png')
'''
