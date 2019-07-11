# Number of workers for dataloader
workers = 2

# Batch size during training
batch_size = 128 #100

# Number of channels in the training images. For color images this is 3
nc = 3

# Size of z latent vector (i.e. size of generator input)
nz = 512

# Size of feature maps in generator
ngf = 64

# Size of feature maps in discriminator
ndf = 64

# Number of training epochs
num_epochs = 100

# Learning rate for optimizers
lr = 0.0005#0.0005

# learning rate decay, default=0.0002
lrd = 0.0001

# Beta1 hyperparam for Adam optimizers
beta1 = 0.8#0.8

# weight for adv loss
adv_weight = 0.1

# multiplicative factor for target adv. loss
alpha = 0.3#0.3

# 0~9 digits
nclasses = 10
