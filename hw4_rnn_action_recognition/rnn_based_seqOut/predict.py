#python predict.py valid_features.pt valid_infos.pt result/best_568.pth ./
import sys
import os
import numpy as np
import torch
from torchvision import transforms

test_features_path = sys.argv[1]
test_infos_path = sys.argv[2]
model_path = sys.argv[3]
pre_label_path = sys.argv[4]


# loading features extracted by pretrain model
print("\nloading videos feature...")
test_features = np.array(torch.load(test_features_path))
print("test_features",test_features.shape)

# loading video infos including names and len(frame number)
print("\nloading videos infos...")
test_infos = torch.load(test_infos_path)
test_dir_names = np.array(test_infos[0])
test_lens = np.array(test_infos[1])
print("test_dir_names",test_dir_names)
print("test_lens",test_lens)


# load model
my_net = torch.load(model_path)
my_net = my_net.eval()
my_net = my_net.cuda()


# testation
accuracy_val = 0
BATCH_SIZE = 16
datalen = len(test_features)
predict_vals_org_order = [None]*datalen
with torch.no_grad():
    for batch_idx, batch_val in enumerate(range(0,datalen ,BATCH_SIZE)):
        # get the batch items
        if batch_val+BATCH_SIZE > datalen: test_features_batch = test_features[batch_val:]
        else: test_features_batch = test_features[batch_val:batch_val+BATCH_SIZE]

        # sort the content in batch items by video length(how much frame/2048d inside)
        lengths = np.array([len(x) for x in test_features_batch])
        sorted_indexes = np.argsort(lengths)[::-1] # decreasing
        test_features_batch = [test_features_batch[i] for i in sorted_indexes]

        # pack as a torch batch and put into cuda
        test_features_batch = torch.nn.utils.rnn.pad_sequence(test_features_batch, batch_first=True)
        test_features_batch = test_features_batch.cuda()

        # predict
        predict_labels = my_net(test_features_batch,torch.LongTensor(sorted(lengths)[::-1]))
        predict_labels_result = []
        for i in range(len(predict_labels)):
            predict_vals = torch.argmax(predict_labels[i],1).cpu().data
            predict_labels_result.append(predict_vals)

        # restored original order (not sorted by length)
        for i,predict_vals in enumerate(predict_labels_result):
            predict_vals_org_order[sorted_indexes[i]+batch_val] = predict_vals
print([len(i) for i in predict_vals_org_order])


# output as csv file
# open videos amount .txt file stored at files list
files = []
for test_len, test_dir_name in zip(test_lens, test_dir_names):
    files.append(open(os.path.join(pre_label_path,test_dir_name+".txt"),'w'))

# writing each parted video batch(200) result to correspond txt
f_index = 0
local_len = test_lens[f_index]
for predict_vals in predict_vals_org_order:
    local_len -= 200
    # do not writing all content (because we already padded)
    if local_len < 0:
        for i,predict_val in enumerate(predict_vals[:local_len+200]):
            files[f_index].write(str(int(predict_val)))
            if (i==len(predict_vals[:local_len+200])-1):
                if f_index < len(test_lens)-1:
                    f_index += 1
                    local_len = test_lens[f_index]
                break
            files[f_index].write('\n')
    # writing all content
    else:
        for predict_val in predict_vals:
            files[f_index].write(str(int(predict_val)))
            files[f_index].write('\n')

print("save predicted files at",os.path.join(pre_label_path))
