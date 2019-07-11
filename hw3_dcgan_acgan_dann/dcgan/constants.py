# Number of workers for dataloader
workers = 2

# Batch size during training
batch_size = 128

# Number of channels in the training images. For color images this is 3
nc = 3

# Size of z latent vector (i.e. size of generator input)
nz = 100

# Size of feature maps in generator, relates to the depth of feature maps carried through the generator
ngf = 64

# Size of feature maps in discriminator, sets the depth of feature maps propagated through the discriminator
ndf = 64

# Number of training epochs
num_epochs = 60

# Learning rate for optimizers
lr = 0.0002

# Beta1 hyperparam for Adam optimizers
beta1 = 0.5

# Number of GPUs available. Use 0 for CPU mode.
ngpu = 1
