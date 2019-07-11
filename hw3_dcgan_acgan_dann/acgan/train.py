import os
import numpy as np
import random
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable

import constants as consts
from dataset import acganDataSet
from model import Generator
from model import Discriminator


# Create image folder if needed
os.makedirs("images", exist_ok=True)
os.makedirs("result", exist_ok=True)

# Decide which device we want to run on
cuda = True if torch.cuda.is_available() else False

######################################################################
# Random
######################################################################
# Set random seem for reproducibility
manualSeed = 1077
#manualSeed = random.randint(1, 10000) # use if you want new results
np.random.seed(manualSeed)
torch.manual_seed(manualSeed)
print("Random Seed: ", manualSeed)


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
dataiter = iter(dataloader)
images, labels = dataiter.next()
print('Image tensor in each batch:', images.shape, images.dtype)
print('Label tensor in each batch:', labels.shape, labels.dtype)

# Plot some training images
'''
real_batch = next(iter(dataloader))
plt.figure(figsize=(8,8))
plt.axis("off")
plt.title("Training Images")
plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))
'''


######################################################################
# Weight Initialization
######################################################################
def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


######################################################################
# Models
######################################################################
# creat generator and discriminator
netG = Generator()
netD = Discriminator()
# Initialize weights
netG.apply(weights_init_normal)
netD.apply(weights_init_normal)
print(netG)
print(netD)


######################################################################
# Loss Functions and Optimizers
######################################################################
# Loss functions
adversarial_loss = torch.nn.BCELoss()
auxiliary_loss = torch.nn.CrossEntropyLoss()

# Create batch of latent vectors that we will use to visualize
#  the progression of the generator
#fixed_noise = torch.randn(64, consts.nz, 1, 1, device=device)

# Establish convention for real and fake labels during training
real_label = 1
fake_label = 0

# Optimizers
optimizerG = torch.optim.Adam(netG.parameters(), lr=consts.lr, betas=(consts.beta1, consts.beta2))
optimizerD = torch.optim.Adam(netD.parameters(), lr=consts.lr, betas=(consts.beta1, consts.beta2))


######################################################################
# saving sampling and loading
######################################################################
def save_checkpoint(checkpoint_path,netG,netD,optimizerG,optimizerD):
    state = {'state_dictG': netG.state_dict(),
             'state_dictD': netD.state_dict(),
             'optimizerG' : optimizerG.state_dict(),
             'optimizerD' : optimizerD.state_dict()}
    torch.save(state, checkpoint_path)
    print('model saved to %s' % checkpoint_path)

def sample_image(n_row, batches_done):
    """Saves a grid of generated digits ranging from 0 to n_classes"""
    # Sample noise
    z = Variable(FloatTensor(np.random.normal(0, 1, (consts.n_classes*n_row, consts.nz))))
    # Get labels ranging from 0 to n_classes for n rows
    #labels = np.array([num for _ in range(n_row) for num in range(n_row)])
    labels = np.array([0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1])
    labels = Variable(LongTensor(labels))
    gen_imgs = netG(z, labels)
    vutils.save_image(gen_imgs.data, "images/%d.png" % batches_done, nrow=n_row, normalize=True)

'''
state = torch.load("weight_91200.pth", map_location="cuda") # for cuda
netG.load_state_dict(state['state_dictG'])
netD.load_state_dict(state['state_dictD'])
optimizerG.load_state_dict(state['optimizerG'])
optimizerD.load_state_dict(state['optimizerD'])
'''

######################################################################
# training loop
######################################################################
FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor
if cuda:
    netG.cuda()
    netD.cuda()
    adversarial_loss.cuda()
    auxiliary_loss.cuda()

G_losses = []
D_losses = []
iters = 0
print("Starting Training Loop...")
for epoch in range(consts.num_epochs):
    for i, (imgs, labels) in enumerate(dataloader):

        batch_size = imgs.shape[0]

        # Adversarial ground truths
        valid = Variable(FloatTensor(batch_size, 1).fill_(1.0), requires_grad=False)
        fake = Variable(FloatTensor(batch_size, 1).fill_(0.0), requires_grad=False)

        # Configure input
        real_imgs = Variable(imgs.type(FloatTensor))
        labels = Variable(labels.type(LongTensor))

        # -----------------
        #  Train Generator
        # -----------------

        optimizerG.zero_grad()

        # Sample noise and labels as generator input
        z = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, consts.nz))))
        gen_labels = Variable(LongTensor(np.random.randint(0, consts.n_classes, batch_size)))

        # Generate a batch of images
        gen_imgs = netG(z, gen_labels)

        # Loss measures generator's ability to fool the discriminator
        validity, pred_label = netD(gen_imgs)
        g_loss = 0.5 * (adversarial_loss(validity, valid) + auxiliary_loss(pred_label, gen_labels))

        g_loss.backward()
        optimizerG.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizerD.zero_grad()

        # Loss for real images
        real_pred, real_aux = netD(real_imgs)
        d_real_loss = (adversarial_loss(real_pred, valid) + auxiliary_loss(real_aux, labels)) / 2

        # Loss for fake images
        fake_pred, fake_aux = netD(gen_imgs.detach())
        d_fake_loss = (adversarial_loss(fake_pred, fake) + auxiliary_loss(fake_aux, gen_labels)) / 2

        # Total discriminator loss
        d_loss = (d_real_loss + d_fake_loss) / 2

        # Calculate discriminator accuracy
        pred = np.concatenate([real_aux.data.cpu().numpy(), fake_aux.data.cpu().numpy()], axis=0)
        gt = np.concatenate([labels.data.cpu().numpy(), gen_labels.data.cpu().numpy()], axis=0)
        d_acc = np.mean(np.argmax(pred, axis=1) == gt)

        d_loss.backward()
        optimizerD.step()

        # Save Losses for plotting later
        G_losses.append(g_loss.item())
        D_losses.append(d_loss.item())

        if (iters % consts.sample_interval ==0) or ((epoch == consts.num_epochs-1) and (i == len(dataloader)-1)):
            sample_image(n_row=10, batches_done=iters)
            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f, acc: %d%%] [G loss: %f]"
                % (epoch, consts.num_epochs, i, len(dataloader), d_loss.item(), 100 * d_acc, g_loss.item())
            )
        if (iters >= 0) and ((iters % consts.sample_interval == 0) or ((epoch == consts.num_epochs-1) and (i == len(dataloader)-1))):
            save_checkpoint('weight_'+str(iters)+'.pth',netG,netD,optimizerG,optimizerD)
        iters += 1

######################################################################
# Results
######################################################################
plt.figure(figsize=(10,5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(G_losses,label="G")
plt.plot(D_losses,label="D")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.savefig('result/LossVisualizer.png')
