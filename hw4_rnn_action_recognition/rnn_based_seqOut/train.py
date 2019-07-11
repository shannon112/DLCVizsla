import os
import sys
import numpy as np
import datetime
import matplotlib.pyplot as plt
from random import shuffle

import torchvision
import torch
import torchvision.transforms as transforms
import torch.nn as nn

from model import Resnet50
from model import LSTM
from model import CustomLoss

# loading features extracted by pretrain model
train_features = torch.load('train_features.pt') # list of torch
valid_features = torch.load('valid_features.pt') # list of torch
train_vals = torch.load('train_vals.pt') # list of torchs
valid_vals = torch.load('valid_vals.pt') # list of torchs
print("train_features",len(train_features))
print("train_vals",len(train_vals))
print("valid_features",len(valid_features))
print("valid_vals",len(valid_vals))


# model, optimzer, loss function
feature_size = 2048
learning_rate = 0.00025
model = LSTM(feature_size).cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
loss_function = CustomLoss()


def load_checkpoint(checkpoint_path, model,optimizer):
    state = torch.load(checkpoint_path) # for cuda
    #state = torch.load(checkpoint_path, map_location=device) #for cpu
    model.load_state_dict(state['state_dict'])
    optimizer.load_state_dict(state['optimizer'])
    print('model loaded from %s' % checkpoint_path)
load_checkpoint("../rnn_based/result/best_498_loading.pth", model, optimizer)
#load_checkpoint("result/best_629.pth", model, optimizer)

for param_group in optimizer.param_groups:
    param_group['lr'] = learning_rate

def save_checkpoint(checkpoint_path, model, optimizer):
    state = {'state_dict': model.state_dict(),
             'optimizer' : optimizer.state_dict()}
    torch.save(state, checkpoint_path)
    print('model saved to %s' % checkpoint_path)

# some training parameters
BATCH_SIZE = 32
EPOCH_NUM = 100
datalen = len(train_features)
datalen_valid = len(valid_features)
max_accuracy = 0
logfile = open('log.txt', 'w')
now = datetime.datetime.now()
logfile.writelines("start training at:"+str(now)+"\n")
logfile.flush()


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
    train_features_sfl = train_features.copy()
    train_vals_sfl = train_vals.copy()
    temp = list(zip(train_features_sfl,train_vals_sfl))
    shuffle(temp)
    train_features_sfl, train_vals_sfl = zip(*temp)

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
        input_features = [input_features[i] for i in sorted_indexes]
        input_vals = [input_vals[i] for i in sorted_indexes]

        # pack as a torch batch and put into cuda
        input_features = torch.nn.utils.rnn.pad_sequence(input_features, batch_first=True)
        input_features = input_features.cuda()
        input_vals = [input_val.cuda() for input_val in input_vals]
        input_vals = torch.nn.utils.rnn.pad_sequence(input_vals, batch_first=True)
        input_vals = input_vals.cuda()

        # forward + backward + optimize
        predict_labels = model(input_features,torch.LongTensor(sorted(lengths)[::-1])) #size= batch_size x video_long(diff) x 11
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
            valid_features_batch = [valid_features_batch[i] for i in sorted_indexes]
            valid_vals_batch = [valid_vals_batch[i] for i in sorted_indexes]

            # pack as a torch batch and put into cuda
            valid_features_batch = torch.nn.utils.rnn.pad_sequence(valid_features_batch, batch_first=True)
            valid_features_batch = valid_features_batch.cuda()
            valid_vals_batch = [valid_vals_bat.cuda() for valid_vals_bat in valid_vals_batch]
            valid_vals_batch = torch.nn.utils.rnn.pad_sequence(valid_vals_batch, batch_first=True)
            valid_vals_batch = valid_vals_batch.cuda()

            # predict
            predict_labels = model(valid_features_batch,torch.LongTensor(sorted(lengths)[::-1]))
            accuracy_val_part = 0
            for i in range(len(predict_labels)):
                predict_vals = torch.argmax(predict_labels[i],1).cpu().data
                valid_vals_batch = valid_vals_batch.cpu().data
                accuracy_val_part += np.mean((predict_vals == valid_vals_batch[i]).numpy())
            accuracy_part = accuracy_val_part / len(predict_labels)
            #print(accuracy_part)
            accuracy_val += accuracy_part
        accuracy = accuracy_val / (batch_idx+1)
        print("validation accuracy: ",accuracy)
        logfile.writelines("validation accuracy: "+str(accuracy)+"\n")
        logfile.flush()
        valid_acc.append(accuracy)

    # saving best acc model as best.pth
    if accuracy > max_accuracy:
        max_accuracy = accuracy
        torch.save(model, 'best.pth')
        #save_checkpoint('best.pth',model,optimizer)
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
plt.savefig("p3_curve.png")
plt.show()
