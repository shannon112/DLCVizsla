# models_UtoM Accuracy: 5622/10000 (56.220000000000006%)
# python predict.py ../../hw3_data/digits/mnistm/test models_UtoM ./models_UtoM/test_pred.csv
# python ../hw3_eval.py models_UtoM/test_pred.csv ../../hw3_data/digits/mnistm/test.csv

# models_MtoS Accuracy: 11604/26032 (44.575906576521206%)
# python predict.py ../../hw3_data/digits/svhn/test models_MtoS ./models_MtoS/test_pred.csv
# python ../hw3_eval.py models_MtoS/test_pred.csv ../../hw3_data/digits/svhn/test.csv

# models_StoU Accuracy: 1448/2007 (72.14748380667663%)
# python predict.py ../../hw3_data/digits/usps/test models_StoU ./models_StoU/test_pred.csv
# python ../hw3_eval.py models_StoU/test_pred.csv ../../hw3_data/digits/usps/test.csv


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
from pre_dataset import GetLoader
import constants as consts

if __name__ == '__main__':
    image_dir, model_path, pre_label_path = sys.argv[1], sys.argv[2], sys.argv[3]
    cudnn.benchmark = True
    cuda = True if torch.cuda.is_available() else False

    ######################################################################
    # load data
    ######################################################################
    mean = np.array([0.44, 0.44, 0.44])
    std = np.array([0.19, 0.19, 0.19])

    transform = transforms.Compose([transforms.Resize(32),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean,std)])

    transform2 = transforms.Compose([transforms.Resize(32),
                                    transforms.Grayscale(),
                                    transforms.ToTensor(),
                                    transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
                                    transforms.Normalize(mean,std)])

    #########################
    # if training mnistm to svhn please use transform2 on target_val & target_train
    #########################
    target_test = GetLoader(
        img_root=image_dir,
        transform=transform2)
    print('# images in dataset:', len(target_test))
    targetloader = torch.utils.data.DataLoader(target_test, batch_size=consts.batch_size, shuffle=False, num_workers=consts.workers)
    sample_batch = next(iter(targetloader))
    print('Image tensor in each batch:', sample_batch[0].shape, sample_batch[0].dtype)

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

    output_lines=[]
    for i, (input_imgs, img_fns) in enumerate(targetloader):
        with torch.no_grad():
            if cuda:
                input_imgs = input_imgs.cuda()
            input_imgs = Variable(input_imgs)

            outC = netC(netF(input_imgs))
            _, predicted = torch.max(outC.data, 1)

            for j,img_fn in enumerate(img_fns):
                output_lines.append([img_fn, predicted[j].item()])

    ######################################################################
    # output as csv file
    ######################################################################
    with open(os.path.join(pre_label_path),'w') as f:
        for i,line in enumerate(output_lines):
            f.write(line[0]+','+str(line[1]))
            if (i==len(output_lines)-1): break
            f.write('\n')
    print("save predicted file at",os.path.join(pre_label_path))
