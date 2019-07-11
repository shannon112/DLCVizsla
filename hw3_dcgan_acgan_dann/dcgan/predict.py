# usage:
# python predict.py ./ weight_17000.pth
import os
import sys
import numpy as np
import random

import torch
import torchvision.transforms as transforms
import torchvision.utils as vutils

# import self-made function
from model import Generator
import constants as consts


# read target dir
output_dir, model_fn = sys.argv[1], sys.argv[2]


######################################################################
# using cuda
######################################################################
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Device used:', device)


######################################################################
# Random
######################################################################
# Set random seem for reproducibility
manualSeed = 9834
#manualSeed = random.randint(1, 10000) # use if you want new results
random.seed(manualSeed)
torch.manual_seed(manualSeed)
fixed_noise = torch.randn(64, consts.nz, 1, 1, device=device)
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
with torch.no_grad():
    fake = netG(fixed_noise).detach().cpu()
    vutils.save_image(fake.data[:32], os.path.join(output_dir,"fig1_2.jpg"), nrow=8, normalize=True)
