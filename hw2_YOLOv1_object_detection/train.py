# import library
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch.optim as optim
import torchvision
import glob
import os
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import datetime

# import self-made function
from dataset import hw2DataSet
from models import Yolov1_vgg16bn
from yoloLoss import yoloLoss

#############################
#1. Creating a custom dataset
#############################
#[2]
# load the trainset and testset
trainset = hw2DataSet(root='../hw2_train_val/train15000/',
    transform=transforms.ToTensor())
testset = hw2DataSet(root='../hw2_train_val/val1500/',
    transform=transforms.ToTensor())
print('# images in trainset:', len(trainset))
print('# images in testset:', len(testset))


#[3]
# Use the torch dataloader to iterate through the dataset
trainset_loader = DataLoader(trainset, batch_size=16, shuffle=True, num_workers=4)
testset_loader = DataLoader(testset, batch_size=16, shuffle=False, num_workers=4)
# get some random training images
dataiter = iter(trainset_loader)
images, labels = dataiter.next()
print('Image tensor in each batch:', images.shape, images.dtype)
print('Label tensor in each batch:', labels.shape, labels.dtype)


#[4]
# We can visualize what contains in each batch:
# functions to show an image
def imshow(img):
    npimg = img.numpy() # transfer torch to numpy
    plt.imshow(np.transpose(npimg, (1, 2, 0))) #if using numpy image tranfered from torch
# functions to show an image's label in 7x7x26
def labelshow(img_number):
    for patchi in range(7):
        for patchj in range(7):
            print (patchi,patchj,labels[img_number][patchi][patchj])

# show images and label
'''imshow(torchvision.utils.make_grid(images))
labelshow(0)
plt.show()'''


###########################################
#2. Creating a Convolutional Neural Network
###########################################
#[5]
# Use GPU if available, otherwise stick with cpu
use_cuda = torch.cuda.is_available()
torch.manual_seed(123)
device = torch.device("cuda" if use_cuda else "cpu")
print('Device used:', device)


#[6]
# import model from model.py
model = Yolov1_vgg16bn(pretrained=True).to(device) # Remember to move the model to "device"
print(model)
logfile = open('log.txt', 'w')

def load_checkpoint(checkpoint_path, model,optimizer):
    state = torch.load(checkpoint_path) # for cuda
    #state = torch.load(checkpoint_path, map_location=device) #for cpu
    model.load_state_dict(state['state_dict'])
    optimizer.load_state_dict(state['optimizer'])
    print('model loaded from %s' % checkpoint_path)

#####################
#3. Train the network
#####################
#[7]
# define the training loop
def train(model, epoch, log_interval=10):
    learning_rate = 0.002
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
    criterion = yoloLoss(5,0.05)
    #criterion = nn.MSELoss()
    load_checkpoint("map0799/best.pth",model,optimizer)
    best_test_loss = np.inf
    iteration = 0 # one iteration would go through a ep
    for ep in range(epoch):
        model.train()  # Important: set training mode
        if ep == 0:
            learning_rate = 0.001
            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rate
        if ep == 20:
            learning_rate = 0.0005
            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rate
        #total_loss = 0.
        for batch_idx,(images,target) in enumerate(trainset_loader):
            images, target = images.to(device), target.to(device)
            optimizer.zero_grad() #to zero
            pred = model(images)
            loss = criterion(pred,target)
            #total_loss += loss.images[0]
            loss.backward()  #backpro
            optimizer.step() #update
            #optimizer.zero_grad() #to zero
            if iteration % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    ep, batch_idx * len(images), len(trainset_loader.dataset),
                    100. * batch_idx / len(trainset_loader), loss.item()))
            iteration += 1

        # Evaluate at the end of each epoch
        best_test_loss = validation(model,optimizer,best_test_loss,ep)


#[8]
# evaluate at the end of each epoch.
def validation(model,optimizer,best_test_loss,ep):
    model.eval()
    criterion = yoloLoss(5,0.05)
    #criterion = nn.MSELoss()
    validation_loss = 0.0
    with torch.no_grad(): # This will free the GPU memory used for back-prop
        for batch_idx,(images,target) in enumerate(testset_loader):
            images, target = images.to(device), target.to(device)
            pred = model(images)
            loss = criterion(pred,target)
            validation_loss += loss.item()
    validation_loss /= len(testset_loader)
    print ("validation avg loss:" + str(validation_loss) + '\n')

    # save best loss as best.pth
    if best_test_loss > validation_loss:
        best_test_loss = validation_loss
        print('get best test loss %.5f' % best_test_loss)
        save_checkpoint('best.pth',model,optimizer)
    if ep%10==0:
        save_checkpoint('best_'+str(ep)+'.pth',model,optimizer)

    # write to logfile
    logfile.writelines("ep: "+str(ep)+" validation avg loss:" + str(validation_loss) + "\n")
    logfile.flush()

    return best_test_loss

def save_checkpoint(checkpoint_path, model, optimizer):
    state = {'state_dict': model.state_dict(),
             'optimizer' : optimizer.state_dict()}
    torch.save(state, checkpoint_path)
    print('model saved to %s' % checkpoint_path)

#[9]
# It's time to train the model!
epochs_num = 50
now = datetime.datetime.now()
logfile.writelines("start training at:"+str(now)+"\n")
logfile.flush()
train(model, epochs_num)
now = datetime.datetime.now()
logfile.writelines("end training at:"+str(now)+"\n")
logfile.flush()
