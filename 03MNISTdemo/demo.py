#[1]
# wget -O mnist_png.tar.gz https://github.com/myleott/mnist_png/blob/master/mnist_png.tar.gz?raw=true
# tar zxf mnist_png.tar.gz


#[2]
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import glob
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


#############################
#1. Creating a custom dataset
#############################
#[3]
#PyTorch has many built-in datasets such as MNIST and CIFAR.
#In this tutorial, we demonstrate how to write your own dataset by implementing a custom MNIST dataset class.
class MNIST(Dataset):
    def __init__(self, root, transform=None):
        """ Intialize the MNIST dataset """
        self.images = None
        self.labels = None
        self.filenames = []
        self.root = root
        self.transform = transform

        # read filenames
        for i in range(10):
            filenames = glob.glob(os.path.join(root, str(i), '*.png'))
            for fn in filenames:
                self.filenames.append((fn, i)) # (filename, label) pair

        self.len = len(self.filenames)

    def __getitem__(self, index):
        """ Get a sample from the dataset """
        image_fn, label = self.filenames[index]
        image = Image.open(image_fn)

        if self.transform is not None:
            image = self.transform(image)

        return image, label

    def __len__(self):
        """ Total number of samples in the dataset """
        return self.len


#[4]
#Let's load the images into custom-created Dataset.
# Create the MNIST dataset.
# transforms.ToTensor() automatically converts PIL images to
# torch tensors with range [0, 1]
trainset = MNIST(root='mnist_png/training',
    transform=transforms.ToTensor())

# load the testset
testset = MNIST(root='mnist_png/testing',
    transform=transforms.ToTensor())

print('# images in trainset:', len(trainset)) # Should print 60000
print('# images in testset:', len(testset)) # Should print 10000


#[5]
#In Pytorch, the "DataLoader" class provides a simple way to collect data into batches.
# Use the torch dataloader to iterate through the dataset
trainset_loader = DataLoader(trainset, batch_size=64, shuffle=True, num_workers=1)
testset_loader = DataLoader(testset, batch_size=1000, shuffle=False, num_workers=1)

# get some random training images
dataiter = iter(trainset_loader)
images, labels = dataiter.next()

print('Image tensor in each batch:', images.shape, images.dtype)
print('Label tensor in each batch:', labels.shape, labels.dtype)


#[6]
#We can visualize what contains in each batch:
#import matplotlib.pyplot as plt
#import numpy as np
# functions to show an image
def imshow(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))

# show images
imshow(torchvision.utils.make_grid(images))
# print labels
print('Labels:')
print(' '.join('%5s' % labels[j] for j in range(16)))


###########################################
#2. Creating a Convolutional Neural Network
###########################################
#[7]
# Let's check if GPU is available. If not, use CPU instead.
# Use GPU if available, otherwise stick with cpu
use_cuda = torch.cuda.is_available()
torch.manual_seed(123)
device = torch.device("cuda" if use_cuda else "cpu")
print('Device used:', device)


#[8]
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 10, kernel_size=5),
            nn.MaxPool2d(2),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(10, 20, kernel_size=5),
            nn.Dropout2d(0.5),
            nn.MaxPool2d(2),
            nn.ReLU()
        )
        self.fc1 = nn.Sequential(
            nn.Linear(320, 50),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(-1, 320)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

model = Net().to(device) # Remember to move the model to "device"
print(model)


#####################
#3. Train the network
#####################
#With the data loaded and network created, it's time to train the model!
#First, we define the training loop.
#[09]
def train(model, epoch, log_interval=100):
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    model.train()  # Important: set training mode

    iteration = 0
    for ep in range(epoch):
        for batch_idx, (data, target) in enumerate(trainset_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            if iteration % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    ep, batch_idx * len(data), len(trainset_loader.dataset),
                    100. * batch_idx / len(trainset_loader), loss.item()))
            iteration += 1

        test(model) # Evaluate at the end of each epoch


#[10]
#Remember to evaluate at the end of each epoch.
def test(model):
    criterion = nn.CrossEntropyLoss()
    model.eval()  # Important: set evaluation mode
    test_loss = 0
    correct = 0
    with torch.no_grad(): # This will free the GPU memory used for back-prop
        for data, target in testset_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item() # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(testset_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(testset_loader.dataset),
        100. * correct / len(testset_loader.dataset)))


#[11]
#It's time to train the model!
'''train(model, 5)  # train 5 epochs should get you to about 97% accuracy'''


##################
#4. Save the model
##################
# Now we have a model! Obviously we do not want to retrain the model everytime we want to use it.
# Plus if you are training a super big model, you probably want to save checkpoint periodically so that you can always fall back to the last checkpoint in case something bad happened or you simply want to test models at different training iterations.
#[12]
#Model checkpointing is fairly simple in PyTorch. First, we define a helper function that can save a model to the disk.
def save_checkpoint(checkpoint_path, model, optimizer):
    state = {'state_dict': model.state_dict(),
             'optimizer' : optimizer.state_dict()}
    torch.save(state, checkpoint_path)
    print('model saved to %s' % checkpoint_path)

def load_checkpoint(checkpoint_path, model, optimizer):
    state = torch.load(checkpoint_path)
    model.load_state_dict(state['state_dict'])
    optimizer.load_state_dict(state['optimizer'])
    print('model loaded from %s' % checkpoint_path)


#[13]
#Define a training loop with model checkpointing:
def train_save(model, epoch, save_interval, log_interval=100):
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    model.train()  # set training mode

    iteration = 0
    for ep in range(epoch):
        for batch_idx, (data, target) in enumerate(trainset_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            if iteration % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    ep, batch_idx * len(data), len(trainset_loader.dataset),
                    100. * batch_idx / len(trainset_loader), loss.item()))
            if iteration % save_interval == 0 and iteration > 0:
                save_checkpoint('mnist-%i.pth' % iteration, model, optimizer)
            iteration += 1
        test(model)

    # save the final model
    save_checkpoint('mnist-%i.pth' % iteration, model, optimizer)


#[14]
#Now, we can save the model in each iteration of training.
# create a brand new model
'''model = Net().to(device)
test(model)
train_save(model, 5, 500, 100)'''


#[15]
#Assume that we have stopped our training program.
#To load the saved model, we need to create the model and optimizer once again.
#Then we load the model weight and optimizer.
# create a new model
model = Net().to(device)
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# load from the final checkpoint
load_checkpoint('mnist-4690.pth', model, optimizer)

# should give you the final model accuracy
#test(test)


###################################
#5. Fine-tuning a pre-trained model
###################################
#Sometimes you want to fine-tune a pretrained model instead of training a model from scratch.
#For example, if you want to train a model on a new dataset that contains natural images.
#To achieve the best performance, you can start with a model that's fully trained on ImageNet and fine-tune the model.

#[16]
#Finetuning a model in PyTorch is super easy! First, let's find out what we saved in a checkpoint:
# What's in a state dict?
print(model.state_dict().keys())


#[17]
#Now say we want to load the conv layers from the checkpoint and train the fc layers.
#We can simply load a subset of the state dict with the selected names
checkpoint = torch.load('mnist-4690.pth')
states_to_load = {}
for name, param in checkpoint['state_dict'].items():
    if name.startswith('conv'):
        states_to_load[name] = param

# Construct a new state dict in which the layers we want
# to import from the checkpoint is update with the parameters
# from the checkpoint
model_state = model.state_dict()
model_state.update(states_to_load)

model = Net().to(device)
model.load_state_dict(model_state)


#[18]
#Let's see how is the fine-tuning result.
train(model, 1)  # training 1 epoch will get you to 93%!


#[19]
#We can even use the pretrained conv layers in a different model.
class SmallNet(nn.Module):
    def __init__(self):
        super(SmallNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 10, kernel_size=5),
            nn.MaxPool2d(2),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(10, 20, kernel_size=5),
            nn.Dropout2d(0.5),
            nn.MaxPool2d(2),
            nn.ReLU()
        )
        self.fc1 = nn.Linear(320, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(-1, 320)
        x = self.fc1(x)
        return x

model = SmallNet().to(device)
print(model)


#[20]
checkpoint = torch.load('mnist-4690.pth')
states_to_load = {}
for name, param in checkpoint['state_dict'].items():
    if name.startswith('conv'):
        states_to_load[name] = param

# Construct a new state dict in which the layers we want
# to import from the checkpoint is update with the parameters
# from the checkpoint
model_state = model.state_dict()
model_state.update(states_to_load)

model.load_state_dict(model_state)


#[21]
train(model, 1)  # training 1 epoch will get you to 93%!


###########################
#6. How to debug the model?
###########################
#Debugging in Pytorch is quite simple.
#"Check your tensor shape and data type constantly."
#For example, if we want to visualize the tensor blob when forwarding the model:
#[22]
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 10, kernel_size=5),
            nn.MaxPool2d(2),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(10, 20, kernel_size=5),
            nn.Dropout2d(0.5),
            nn.MaxPool2d(2),
            nn.ReLU()
        )
        self.fc1 = nn.Sequential(
            nn.Linear(320, 50),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = self.conv1(x)
        print('Tensor size and type after conv1:', x.shape, x.dtype)
        x = self.conv2(x)
        print('Tensor size and type after conv2:', x.shape, x.dtype)
        x = x.view(-1, 320)
        print('Tensor size and type after view():', x.shape, x.dtype)
        x = self.fc1(x)
        print('Tensor size and type after fc1:', x.shape, x.dtype)
        x = self.fc2(x)
        print('Tensor size and type after fc2:', x.shape, x.dtype)
        return x

model = Net().to(device) # Remember to move the model to "device"


#[23]
# Fowarding a dummy tensor
x = torch.Tensor(64,1,28,28).to(device) # shape of N*C*H*W
x = model(x)

plt.show()
