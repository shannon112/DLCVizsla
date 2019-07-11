# python predict.py valid_features.pt ../../hw4_data/TrimmedVideos/label/gt_valid.csv best_46.pth ./
import sys
import os
import numpy as np
import torch
from torchvision import transforms
from reader import getVideoList

test_features_path = sys.argv[1]
test_label_path = sys.argv[2] #TrimmedVideos/label/gt_valid.csv
model_path = sys.argv[3]
pre_label_path = sys.argv[4]


# loading features extracted by pretrain model
print("\nloading videos feature...")
test_features = torch.load(test_features_path).view(-1,2048)
print("test_features",test_features.shape)


# load model
my_net = torch.load(model_path)
my_net = my_net.eval()
my_net = my_net.cuda()
predict_labels,_ = my_net(test_features.cuda())
predict_vals = torch.argmax(predict_labels,1).cpu().data
print("prediction:",predict_vals)


# get test label csv content
#dict = getVideoList(os.path.join(test_label_path))


# output as csv file
with open(os.path.join(pre_label_path,"p1_valid.txt"),'w') as f:
    #f.write("Video_index,Video_name,Video_category,Start_times,End_times,Action_labels,Nouns\n")
    for i,predict_val in enumerate(predict_vals):
        #f.write(dict['Video_index'][i]+','+dict['Video_name'][i]+','+dict['Video_category'][i]+','+dict['Start_times'][i]+','+dict['End_times'][i]+",")
        f.write(str(int(predict_val)))
        #f.write(dict['Nouns'][i])
        if (i==len(predict_vals)-1): break
        f.write('\n')
print("save predicted file at",os.path.join(pre_label_path,"p1_valid.txt"))
