# python predict.py test_features.pt ../../hw4_data/TrimmedVideos/label/gt_valid.csv best_46.pth ./
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
test_features = np.array(torch.load(test_features_path))
print("test_features",test_features.shape)


# load model
my_net = torch.load(model_path)
my_net = my_net.eval()
my_net = my_net.cuda()


# validation
accuracy_val = 0
BATCH_SIZE = 64
datalen = len(test_features)
predict_vals_org_order = np.zeros(datalen).astype(int)
with torch.no_grad():
    for batch_idx, batch_val in enumerate(range(0,datalen ,BATCH_SIZE)):
        # get the batch items
        if batch_val+BATCH_SIZE > datalen: test_features_batch = test_features[batch_val:]
        else: test_features_batch = test_features[batch_val:batch_val+BATCH_SIZE]

        # sort the content in batch items by video length(how much frame/2048d inside)
        lengths = np.array([len(x) for x in test_features_batch])
        sorted_indexes = np.argsort(lengths)[::-1] # decreasing
        test_features_batch = test_features_batch[sorted_indexes]

        # pack as a torch batch and put into cuda
        test_features_batch = torch.nn.utils.rnn.pad_sequence(test_features_batch, batch_first=True)

        # predict
        predict_labels,_,_ = my_net(test_features_batch,torch.LongTensor(sorted(lengths)[::-1]))
        predict_vals = torch.argmax(predict_labels,1).cpu().data

        # restored original order (not sorted by length)
        for i,predict_val in enumerate(predict_vals):
            predict_vals_org_order[sorted_indexes[i]+batch_val] = int(predict_val)
    print(predict_vals_org_order)


# get test label csv content
#dict = getVideoList(os.path.join(test_label_path))


# output as csv file
with open(os.path.join(pre_label_path,"p2_result.txt"),'w') as f:
    #f.write("Video_index,Video_name,Video_category,Start_times,End_times,Action_labels,Nouns\n")
    for i,predict_val in enumerate(predict_vals_org_order):
        #f.write(dict['Video_index'][i]+','+dict['Video_name'][i]+','+dict['Video_category'][i]+','+dict['Start_times'][i]+','+dict['End_times'][i]+",")
        f.write(str(int(predict_val)))
        #f.write(dict['Nouns'][i])
        if (i==len(predict_vals_org_order)-1): break
        f.write('\n')
print("save predicted file at",os.path.join(pre_label_path,"p2_result.txt"))
