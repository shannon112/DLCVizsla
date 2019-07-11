# Number of workers for dataloader
workers = 2

# Batch size during training
batch_size = 64

# images Size
img_size = 64

# Number of channels in the training images. For color images this is 3
nc = 3

# Size of z latent vector (i.e. size of generator input) #latent_dim "dimensionality of the latent space"
nz = 100

# Size of feature maps in generator
ngf = 64

# Size of feature maps in discriminator
ndf = 64

# Number of training epochs
num_epochs = 200

# Learning rate for optimizers
lr = 0.0002

# Beta1 hyperparam for Adam optimizers
beta1 = 0.5
beta2 = 0.999 #same as default

# number of classes for dataset
n_classes = 2 #smile or not

# sample_interval
sample_interval = 200
