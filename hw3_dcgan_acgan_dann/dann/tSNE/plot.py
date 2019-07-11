#UtoM on M Accuracy: 4416/10000 (44.16%)
#python predict.py ../models_UtoM/best.pth ../../../hw3_data/digits/usps/test ../../../hw3_data/digits/usps/test.csv  ../../../hw3_data/digits/mnistm/test ../../../hw3_data/digits/mnistm/test.csv

#MtoS on S Accuracy: 12704/26032 (48.80147510755993%)
#python predict.py ../models_MtoS/best.pth ../../../hw3_data/digits/mnistm/test ../../../hw3_data/digits/mnistm/test.csv ../../../hw3_data/digits/svhn/test ../../../hw3_data/digits/svhn/test.csv

#StoU on U Accuracy: 1115/2007 (55.55555555555556%)
#python predict.py ../models_StoU/best.pth ../../../hw3_data/digits/svhn/test ../../../hw3_data/digits/svhn/test.csv ../../../hw3_data/digits/usps/test ../../../hw3_data/digits/usps/test.csv


import sys
import os
import torch.backends.cudnn as cudnn
import torch.utils.data
from torchvision import transforms
from torchvision import datasets

from dataset import GetLoader
import constants as consts

from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
from torch.autograd import Variable

os.makedirs("result", exist_ok=True)

model_path, image_dir_s, label_path_s ,image_dir_t, label_path_t = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5]
cuda = True if torch.cuda.is_available() else False
cudnn.benchmark = True
alpha = 0

######################################################################
# load data
######################################################################
dataset_s = GetLoader(
    img_root=image_dir_s,
    label_path=label_path_s,
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
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
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
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
my_net = torch.load(model_path)
my_net = my_net.eval()

######################################################################
# predict
######################################################################
if cuda:
    my_net = my_net.cuda()
n_total = 0
n_correct = 0
FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

output_lines=[]
with torch.no_grad():
    s_img, s_label = sample_batch_s
    t_img, t_label = sample_batch_t

    batch_size = len(t_img)
    input_img_t = FloatTensor(batch_size, 3, consts.image_size, consts.image_size)
    input_img_s = FloatTensor(batch_size, 3, consts.image_size, consts.image_size)

    if cuda:
        t_img = t_img.cuda()
        input_img_t = input_img_t.cuda()
        s_img = s_img.cuda()
        input_img_s = input_img_s.cuda()

    input_img_t.resize_as_(t_img).copy_(t_img)
    input_img_s.resize_as_(s_img).copy_(s_img)

    __,_ , t_features = my_net(input_data=input_img_t, alpha=alpha)
    __,_ , s_features = my_net(input_data=input_img_s, alpha=alpha)

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
    plt.savefig('result/diff_digit_'+model_path[-20:-9]+'.png')

    fig2 = plt.figure(figsize=(12, 10)).suptitle('different domain')
    plt.scatter(X_2d_t[y_t == y_t, 0], X_2d_t[y_t == y_t, 1], c='r', label="target")
    plt.scatter(X_2d_s[y_s == y_s, 0], X_2d_s[y_s == y_s, 1], c='g', label="source")
    plt.legend()
    plt.savefig('result/diff_domain_'+model_path[-20:-9]+'.png')
