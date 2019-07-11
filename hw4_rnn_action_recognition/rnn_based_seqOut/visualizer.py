#python tSNEplot.py valid_features.pt ../../hw4_data/TrimmedVideos/label/gt_valid.csv result/best.pth
import sys
import os
import numpy as np
import glob
import torch.backends.cudnn as cudnn
import torch.utils.data
from torchvision import transforms
from torchvision import datasets
from sklearn.manifold import TSNE
import matplotlib
import matplotlib.pyplot as plt


test_predict_path = sys.argv[1]
test_label_path = sys.argv[2] #TrimmedVideos/label/gt_valid.csv

predict_filenames = sorted(glob.glob(os.path.join(test_predict_path, '*.txt')))
label_filenames = sorted(glob.glob(os.path.join(test_label_path, '*.txt')))
videodir_names = [label_filename.split('/')[-1] for label_filename in label_filenames]

mean_accuracy = 0
for i,(predict_filename,label_filename) in enumerate(zip(predict_filenames,label_filenames)):
    # read predict files
    f = open(os.path.join(predict_filename),'r')
    predict_vals = f.read().splitlines()
    predict_vals = np.array(predict_vals).astype(int)

    # read label files
    f = open(os.path.join(label_filename),'r')
    label_vals = f.read().splitlines()
    label_vals = np.array(label_vals).astype(int)

    # new a plot
    plt.figure(figsize=(16,4))
    colors = plt.cm.get_cmap('tab20',11).colors

    # plotting predict result
    ax1 = plt.subplot(211)
    ax1.set_ylabel('Prediction')
    cmap = matplotlib.colors.ListedColormap([colors[idx] for idx in predict_vals])
    bounds = [i for i in range(len(predict_vals))]
    norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)
    cb1 = matplotlib.colorbar.ColorbarBase(ax1, cmap=cmap,
                                           norm=norm,
                                           boundaries=bounds,
                                           spacing='proportional',
                                           orientation='horizontal')
    # plotting label result
    ax2 = plt.subplot(212)
    ax2.set_ylabel('Orginal Label')
    cmap = matplotlib.colors.ListedColormap([colors[idx] for idx in label_vals])
    bounds = [i for i in range(len(label_vals))]
    norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)
    cb2 = matplotlib.colorbar.ColorbarBase(ax2, cmap=cmap,
                                           norm=norm,
                                           boundaries=bounds,
                                           spacing='proportional',
                                           orientation='horizontal')
    # save as png file
    plt.savefig(test_predict_path+videodir_names[i]+'.png')
