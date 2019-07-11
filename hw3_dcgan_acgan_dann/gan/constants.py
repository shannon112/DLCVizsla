batch_size = 128 #size of the batches
workers = 2

img_size = 64 #size of each image dimension
channels = 3 #number of image channels
img_shape = (channels, img_size, img_size)

lr = 0.0002 #adam: learning rate
b1 = 0.5 # adam: decay of first order momentum of gradient"
b2 = 0.999 # adam: decay of first order momentum of gradient"

n_epochs = 200 #number of epochs of training
latent_dim = 100 #dimensionality of the latent space
sample_interval = 400 #interval betwen image samples
