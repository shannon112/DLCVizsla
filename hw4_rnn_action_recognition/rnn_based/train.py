import os
import sys
import numpy as np
import datetime
import matplotlib.pyplot as plt

import torchvision
import torch
import torchvision.transforms as transforms
import torch.nn as nn

from model import Resnet50
from model import LSTM
from reader import readShortVideo
from reader import getVideoList


# loading features extracted by pretrain model
train_features = np.array(torch.load('train_features.pt'))
valid_features = np.array(torch.load('valid_features.pt'))
train_vals = torch.load('train_vals.pt').type(torch.LongTensor)
valid_vals = torch.load('valid_vals.pt').type(torch.LongTensor)
print("train_features",train_features.shape)
print("train_vals",train_vals.shape)
print("valid_features",valid_features.shape)
print("valid_vals",valid_vals.shape)

# model, optimzer, loss function
feature_size = 2048
model = LSTM(feature_size).cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=0.00002)
loss_function = nn.CrossEntropyLoss()


# some training parameters
BATCH_SIZE = 64
EPOCH_NUM = 100
datalen = len(train_features)
datalen_valid = len(valid_features)
max_accuracy = 0
logfile = open('log.txt', 'w')
now = datetime.datetime.now()
logfile.writelines("start training at:"+str(now)+"\n")
logfile.flush()

def save_checkpoint(checkpoint_path, model, optimizer):
    state = {'state_dict': model.state_dict(),
             'optimizer' : optimizer.state_dict()}
    torch.save(state, checkpoint_path)
    print('model saved to %s' % checkpoint_path)

# start training
model.train()
train_loss = []
valid_acc = []
for epoch in range(EPOCH_NUM):
    logfile.writelines("Epoch:"+str(epoch+1)+"\n")
    logfile.flush()
    print("Epoch:", epoch+1)
    total_loss = 0.0
    total_batchnum = 0

    # shuffle data
    perm_index = torch.randperm(datalen)
    train_features_sfl = train_features[perm_index]
    train_vals_sfl = train_vals[perm_index]

    # training as batches
    for batch_idx, batch_val in enumerate(range(0,datalen ,BATCH_SIZE)):
        if batch_val+BATCH_SIZE > datalen: break
        optimizer.zero_grad()  # zero the parameter gradients

        # get the batch items
        input_features = train_features_sfl[batch_val:batch_val+BATCH_SIZE]
        input_vals = train_vals_sfl[batch_val:batch_val+BATCH_SIZE]

        # sort the content in batch items by video length(how much frame/2048d inside)
        lengths = np.array([len(x) for x in input_features])
        sorted_indexes = np.argsort(lengths)[::-1] # decreasing
        input_features = input_features[sorted_indexes]
        input_vals = input_vals.numpy()[sorted_indexes]

        # pack as a torch batch and put into cuda
        input_features = torch.nn.utils.rnn.pad_sequence(input_features, batch_first=True)
        input_vals = torch.from_numpy(input_vals)
        input_vals = input_vals.cuda()
        input_features = input_features.cuda()

        # forward + backward + optimize
        predict_labels,_,_ = model(input_features,torch.LongTensor(sorted(lengths)[::-1])) #size 64x11
        loss = loss_function(predict_labels, input_vals) #size 64x11 vs 64
        loss.backward()
        optimizer.step()
        total_loss += loss.cpu().data.numpy()
        total_batchnum = batch_idx+1
    print("avg training loss:",total_loss / total_batchnum)
    logfile.writelines("avg training loss:"+ str(total_loss / total_batchnum)+"\n")
    logfile.flush()
    train_loss.append(total_loss / total_batchnum)

    # validation
    accuracy_val = 0
    with torch.no_grad():
        model.eval()
        for batch_idx, batch_val in enumerate(range(0,datalen_valid ,BATCH_SIZE)):
            # get the batch items
            if batch_val+BATCH_SIZE > datalen_valid:
                valid_features_batch = valid_features[batch_val:]
                valid_vals_batch = valid_vals[batch_val:]
            else:
                valid_features_batch = valid_features[batch_val:batch_val+BATCH_SIZE]
                valid_vals_batch = valid_vals[batch_val:batch_val+BATCH_SIZE]

            # sort the content in batch items by video length(how much frame/2048d inside)
            lengths = np.array([len(x) for x in valid_features_batch])
            sorted_indexes = np.argsort(lengths)[::-1] # decreasing
            valid_features_batch = valid_features_batch[sorted_indexes]
            valid_vals_batch = valid_vals_batch.numpy()[sorted_indexes]

            # pack as a torch batch and put into cuda
            valid_features_batch = torch.nn.utils.rnn.pad_sequence(valid_features_batch, batch_first=True)
            valid_vals_batch = torch.from_numpy(valid_vals_batch)

            # predict
            predict_labels,_,_ = model(valid_features_batch,torch.LongTensor(sorted(lengths)[::-1]))
            predict_vals = torch.argmax(predict_labels,1).cpu().data
            accuracy_val += np.sum((predict_vals == valid_vals_batch).numpy())
        accuracy = accuracy_val / float(datalen_valid)
        print("validation accuracy: ",accuracy)
        logfile.writelines("validation accuracy: "+str(accuracy)+"\n")
        logfile.flush()
        valid_acc.append(accuracy)

    # saving best acc model as best.pth
    if accuracy > max_accuracy:
        max_accuracy = accuracy
        save_checkpoint('best_loading.pth',model,optimizer)
        #torch.save(model, 'best.pth')
        logfile.writelines("save as best.pth\n")
        logfile.flush()
    model.train()

now = datetime.datetime.now()
logfile.writelines("end training at:"+str(now)+"\n")
logfile.flush()

# plot loss and acc graph
plt.figure(figsize=(15,5))
plt.subplot(1,2,1)
plt.plot(train_loss)
plt.title("cross entropy training loss")
plt.ylabel("loss")
plt.xlabel("epoch")
plt.subplot(1,2,2)
plt.plot(valid_acc)
plt.title("validation accuracy")
plt.ylabel("accuracy")
plt.xlabel("epoch")
plt.savefig("p2_curve.png")
plt.show()
