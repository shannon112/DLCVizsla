# models_UtoM Accuracy: 5622/10000 (56.220000000000006%)
#python plot.py ../models_UtoM ../../../hw3_data/digits/usps/test ../../../hw3_data/digits/usps/test.csv  ../../../hw3_data/digits/mnistm/test ../../../hw3_data/digits/mnistm/test.csv

# models_MtoS_old Accuracy: 11604/26032 (44.575906576521206%)
#python plot.py ../models_MtoS ../../../hw3_data/digits/mnistm/test ../../../hw3_data/digits/mnistm/test.csv ../../../hw3_data/digits/svhn/test ../../../hw3_data/digits/svhn/test.csv

# models_StoU Accuracy: 1448/2007 (72.14748380667663%)
#python plot.py ../models_StoU ../../../hw3_data/digits/svhn/test ../../../hw3_data/digits/svhn/test.csv ../../../hw3_data/digits/usps/test ../../../hw3_data/digits/usps/test.csv


import numpy as np
import sys
import os

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torchvision.datasets as dset
import torchvision.transforms as transforms

import models
import utils
from dataset import GetLoader
import constants as consts

from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
from torch.autograd import Variable

if __name__ == '__main__':
    os.makedirs("result", exist_ok=True)

    model_path, image_dir_s, label_path_s ,image_dir_t, label_path_t = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5]
    cudnn.benchmark = True
    cuda = True if torch.cuda.is_available() else False

    ######################################################################
    # load data
    ######################################################################
    mean = np.array([0.44, 0.44, 0.44])
    std = np.array([0.19, 0.19, 0.19])

    transform = transforms.Compose([transforms.Resize(32), transforms.ToTensor(), transforms.Normalize(mean,std)])

    dataset_s = GetLoader(
        img_root=image_dir_s,
        label_path=label_path_s,
        transform=transform
    )
    print('# images in dataset:', len(dataset_s))
    dataloader_s = torch.utils.data.DataLoader(
        dataset=dataset_s,
        batch_size=consts.batch_size,
        shuffle=True,
        num_workers=consts.workers
    )
    sample_batch_s = next(iter(dataloader_s))
    print('Image tensor in each batch:', sample_batch_s[0].shape, sample_batch_s[0].dtype)
    print('Label tensor in each batch:', sample_batch_s[1].shape, sample_batch_s[0].dtype)

    dataset_t = GetLoader(
        img_root=image_dir_t,
        label_path=label_path_t,
        transform=transform
    )
    print('# images in dataset:', len(dataset_t))
    dataloader_t = torch.utils.data.DataLoader(
        dataset=dataset_t,
        batch_size=consts.batch_size,
        shuffle=True,
        num_workers=consts.workers
    )
    sample_batch_t = next(iter(dataloader_t))
    print('Image tensor in each batch:', sample_batch_t[0].shape, sample_batch_t[0].dtype)
    print('Label tensor in each batch:', sample_batch_t[1].shape, sample_batch_t[0].dtype)

    ######################################################################
    # load model
    ######################################################################
    netF = models._netF()
    netC = models._netC()
    netF_path = os.path.join(model_path, 'model_best_netF.pth')
    netC_path = os.path.join(model_path, 'model_best_netC.pth')
    netF.load_state_dict(torch.load(netF_path))
    netC.load_state_dict(torch.load(netC_path))

    ######################################################################
    # predict
    ######################################################################
    if cuda:
        netF.cuda()
        netC.cuda()
    # Testing
    netF.eval()
    netC.eval()

    with torch.no_grad():
        s_img, s_label = sample_batch_s
        t_img, t_label = sample_batch_t

        batch_size = len(t_img)

        if cuda:
            t_img = t_img.cuda()
            s_img = s_img.cuda()

        t_img = Variable(t_img)
        s_img = Variable(s_img)

        _, t_features = netC(netF(t_img))
        _, s_features = netC(netF(s_img))

        #############################################################
        # tSNE to visualize digits
        # https://scipy-lectures.org/packages/scikit-learn/auto_examples/plot_tsne.html
        #############################################################
        #Fit and transform with a TSNE
        x_t = (t_features.cpu()).numpy()
        x_s = (s_features.cpu()).numpy()
        y_t = (t_label.cpu()).numpy()
        y_s = (s_label.cpu()).numpy()
        print(x_t.shape,x_t)
        print(x_s.shape,x_s)
        print(y_t.shape,y_t)
        print(y_s.shape,y_s)
        tsne = TSNE(n_components=2, random_state=0)

        #Project the data in 2D
        X_2d_t = tsne.fit_transform(x_t)
        X_2d_s = tsne.fit_transform(x_s)

        #Visualize the data
        target_names = ['0','1','2','3','4','5','6','7','8','9']
        target_ids = range(len(target_names)) # 0~9 digits

        fig1 = plt.figure(figsize=(12, 10)).suptitle('different digits')
        colors = 'r', 'g', 'b', 'c', 'm', 'y', 'k', 'peru', 'orange', 'purple'
        for i, c, label in zip(target_ids, colors, target_names):
            plt.scatter(X_2d_t[y_t== i, 0], X_2d_t[y_t == i, 1], c=c, label=label)
            plt.scatter(X_2d_s[y_s == i, 0], X_2d_s[y_s == i, 1], c=c, label=label)
        plt.legend()
        plt.savefig('result/diff_digit_'+model_path[-11:]+'.png')

        fig2 = plt.figure(figsize=(12, 10)).suptitle('different domain')
        plt.scatter(X_2d_t[y_t == y_t, 0], X_2d_t[y_t == y_t, 1], c='r', label="target")
        plt.scatter(X_2d_s[y_s == y_s, 0], X_2d_s[y_s == y_s, 1], c='g', label="source")
        plt.legend()
        plt.savefig('result/diff_domain_'+model_path[-11:]+'.png')
