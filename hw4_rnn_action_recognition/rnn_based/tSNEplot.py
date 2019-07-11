#python tSNEplot.py valid_features.pt ../../hw4_data/TrimmedVideos/label/gt_valid.csv result/best.pth

import sys
import os
import numpy as np
import torch.backends.cudnn as cudnn
import torch.utils.data
from torchvision import transforms
from torchvision import datasets
from reader import getVideoList
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt

test_features_path = sys.argv[1]
test_label_path = sys.argv[2] #TrimmedVideos/label/gt_valid.csv
model_path = sys.argv[3]


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
predict_features = []
for i in range(datalen): predict_features.append(torch.randn(256))
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
        _,_,predict_fs = my_net(test_features_batch,torch.LongTensor(sorted(lengths)[::-1]))

        # restored original order (not sorted by length)
        for i,predict_f in enumerate(predict_fs):
            predict_features[sorted_indexes[i]+batch_val] = predict_f
predict_features = torch.stack(predict_features)

# get test label csv content
dict = getVideoList(os.path.join(test_label_path))
action_labels = (dict['Action_labels'])


# tSNE to visualize
#x_t = (t_features.cpu()).numpy()
#y_t = (t_label.cpu()).numpy()
X = np.array(predict_features.tolist())
Y = np.array(dict['Action_labels']).astype(int)

tsne = TSNE(n_components=2, random_state=0)

#Project the data in 2D
X_2d = tsne.fit_transform(X)

#Visualize the data
target_names = ['0others','1Inspect/Read','2Open','3Take','4Cut','5Put','6Close','7Move_around','8Divide/Pull_apart','9Pour','10Transfer']
target_ids = range(len(target_names)) # 0~10 digits
fig1 = plt.figure(figsize=(12, 10)).suptitle('tSNE plot of cnn_based action recognition')
colors = 'r', 'g', 'b', 'c', 'm', 'y', 'k', 'peru', 'orange', 'purple' ,'indigo'
for i, c, label in zip(target_ids, colors, target_names):
    plt.scatter(X_2d[Y== i, 0], X_2d[Y == i, 1], c=c, label=label)
plt.legend()
plt.savefig('result/tSNEplot.png')
