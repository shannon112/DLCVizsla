#UtoM on M Accuracy: 4416/10000 (44.16%)
#python predict.py ../../hw3_data/digits/mnistm/test models_UtoM/best.pth ./models_UtoM/test_pred.csv
#python ../hw3_eval.py models_UtoM/test_pred.csv ../../hw3_data/digits/mnistm/test.csv

#MtoS on S Accuracy: 12704/26032 (48.80147510755993%)
#python predict.py ../../hw3_data/digits/svhn/test models_MtoS/best.pth ./models_MtoS/test_pred.csv
#python ../hw3_eval.py models_MtoS/test_pred.csv ../../hw3_data/digits/svhn/test.csv

#StoU on U Accuracy: 1115/2007 (55.55555555555556%)
#python predict.py ../../hw3_data/digits/usps/test models_StoU/best.pth ./models_StoU/test_pred.csv
#python ../hw3_eval.py models_StoU/test_pred.csv ../../hw3_data/digits/usps/test.csv

import sys
import os
import torch.backends.cudnn as cudnn
import torch.utils.data
from torchvision import transforms
from torchvision import datasets

from pre_dataset import GetLoader
import constants as consts

image_dir, model_path, pre_label_path = sys.argv[1], sys.argv[2], sys.argv[3]
cuda = True if torch.cuda.is_available() else False
cudnn.benchmark = True
alpha = 0

######################################################################
# load data
######################################################################
dataset = GetLoader(
    img_root=image_dir,
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
)
print('# images in dataset:', len(dataset))
dataloader = torch.utils.data.DataLoader(
    dataset=dataset,
    batch_size=consts.batch_size,
    shuffle=False,
    num_workers=consts.workers
)
sample_batch = next(iter(dataloader))
print('Image tensor in each batch:', sample_batch[0].shape, sample_batch[0].dtype)

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
for i, (t_img, t_img_fns) in enumerate(dataloader):
    batch_size = len(t_img)
    input_img = FloatTensor(batch_size, 3, consts.image_size, consts.image_size)
    if cuda:
        t_img = t_img.cuda()
        input_img = input_img.cuda()
    input_img.resize_as_(t_img).copy_(t_img)

    class_output, _ = my_net(input_data=input_img, alpha=alpha)
    pred = class_output.data.max(1, keepdim=True)[1]

    for j,img_fn in enumerate(t_img_fns):
        output_lines.append([img_fn, pred[j][0].item()])

######################################################################
# output as csv file
######################################################################
with open(os.path.join(pre_label_path),'w') as f:
    for i,line in enumerate(output_lines):
        f.write(line[0]+','+str(line[1]))
        if (i==len(output_lines)-1): break
        f.write('\n')
print("save predicted file at",os.path.join(pre_label_path))
