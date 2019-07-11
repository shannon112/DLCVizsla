import random
import numpy as np
import os

import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torchvision import datasets
from torchvision import transforms

from test import test
from model import CNNModel
from dataset import GetLoader
import constants as consts

source_dataset_name = 'mnistm'
target_dataset_name = 'svhn'
source_image_root = os.path.join('../../hw3_data/digits/', source_dataset_name)
target_image_root = os.path.join('../../hw3_data/digits/', target_dataset_name)
cudnn.benchmark = True

# Create models folder if needed
os.makedirs("models", exist_ok=True)

# Decide which device we want to run on
cuda = True if torch.cuda.is_available() else False

######################################################################
# Random
######################################################################
'''
manual_seed = random.randint(1, 10000)
# manual_seed = 999
random.seed(manual_seed)
torch.manual_seed(manual_seed)
'''


######################################################################
# Dataset and DataLoader
######################################################################
dataset_source = GetLoader(
    img_root=os.path.join(source_image_root,"train"),
    label_path=os.path.join(source_image_root, 'train.csv'),
    transform=transforms.Compose([
        transforms.ToTensor(),  # to 0~1
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)) # to -0.5~0.5
    ])
)
print('# images in dataset_source:', len(dataset_source))
dataloader_source = torch.utils.data.DataLoader(
    dataset=dataset_source,
    batch_size=consts.batch_size,
    shuffle=True,
    num_workers=consts.workers)
sample_batch = next(iter(dataloader_source))
print('Image tensor in each batch:', sample_batch[0].shape, sample_batch[0].dtype)
print('Label tensor in each batch:', sample_batch[1].shape, sample_batch[1].dtype)

dataset_target = GetLoader(
    img_root=os.path.join(target_image_root,"train"),
    label_path=os.path.join(target_image_root, 'train.csv'),
    transform=transforms.Compose([
        transforms.ToTensor(),  # to 0~1
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)) # to -0.5~0.5
    ])
)
print('# images in dataset_target:', len(dataset_target))
dataloader_target = torch.utils.data.DataLoader(
    dataset=dataset_target,
    batch_size=consts.batch_size,
    shuffle=True,
    num_workers=consts.workers)
sample_batch = next(iter(dataloader_target))
print('Image tensor in each batch:', sample_batch[0].shape, sample_batch[0].dtype)
print('Label tensor in each batch:', sample_batch[1].shape, sample_batch[1].dtype)


######################################################################
# Models
######################################################################
my_net = CNNModel()
print(my_net)


######################################################################
# Loss Functions and Optimizers
######################################################################
optimizer = optim.Adam(my_net.parameters(), lr=consts.lr)
loss_class = torch.nn.NLLLoss()
loss_domain = torch.nn.NLLLoss()


######################################################################
# training loop
######################################################################
FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor
if cuda:
    my_net.cuda()
    loss_class.cuda()
    loss_domain.cuda()
for p in my_net.parameters():
    p.requires_grad = True

for epoch in range(consts.num_epochs):
    len_dataloader = min(len(dataloader_source), len(dataloader_target))
    data_source_iter = iter(dataloader_source)
    data_target_iter = iter(dataloader_target)

    for i in range(len_dataloader):
        p = float(i + epoch * len_dataloader) / consts.num_epochs / len_dataloader
        alpha = 2. / (1. + np.exp(-10 * p)) - 1

        # -----------------
        #  Training model using source data
        # -----------------
        data_source = data_source_iter.next()
        s_img, s_label = data_source
        batch_size = len(s_label)
        input_img = FloatTensor(batch_size, 3, consts.image_size, consts.image_size)
        class_label = LongTensor(batch_size)
        domain_label = LongTensor(np.zeros(batch_size))

        if cuda:
            s_img = s_img.cuda()
            s_label = s_label.cuda()
            input_img = input_img.cuda()
            class_label = class_label.cuda()
            domain_label = domain_label.cuda()

        input_img.resize_as_(s_img).copy_(s_img)
        class_label.resize_as_(s_label).copy_(s_label)

        my_net.zero_grad()

        class_output, domain_output = my_net(input_data=input_img, alpha=alpha)
        err_s_label = loss_class(class_output, class_label)
        err_s_domain = loss_domain(domain_output, domain_label)

        # -----------------
        # Training model using target data
        # -----------------

        data_target = data_target_iter.next()
        t_img, _ = data_target  # we would not see the label
        batch_size = len(t_img)
        input_img = FloatTensor(batch_size, 3, consts.image_size, consts.image_size)
        domain_label = LongTensor(np.ones(batch_size))

        if cuda:
            t_img = t_img.cuda()
            input_img = input_img.cuda()
            domain_label = domain_label.cuda()

        input_img.resize_as_(t_img).copy_(t_img)

        _, domain_output = my_net(input_data=input_img, alpha=alpha)
        err_t_domain = loss_domain(domain_output, domain_label)


        # transfer learning
        err = err_t_domain + err_s_domain + err_s_label
        # training on domain
        #err =  err_s_label
        err.backward()
        optimizer.step()

        #print ('epoch: %d, [iter: %d / all %d], err_s_label: %f, err_s_domain: %f, err_t_domain: %f' \
        #      % (epoch, i, len_dataloader, err_s_label.data.cpu().numpy(),
        #         err_s_domain.data.cpu().numpy(), err_t_domain.data.cpu().item()))

    torch.save(my_net, 'models/epoch_{}.pth'.format(epoch))
    test(target_dataset_name, epoch)

print('done')
