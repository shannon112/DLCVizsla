# extract train feature and label
#python featureExtractor.py ../../hw4_data/TrimmedVideos/video/train ../../hw4_data/TrimmedVideos/label/gt_train.csv ./ train

# extract valid feature and label
#python featureExtractor.py ../../hw4_data/TrimmedVideos/video/valid ../../hw4_data/TrimmedVideos/label/gt_valid.csv ./ valid

import os
import numpy as np
import sys

import torch
import torchvision.transforms as transforms

from tqdm import tqdm
from model import Resnet50
from reader import readShortVideo
from reader import getVideoList
tqdm.pandas()


# data source
video_path = sys.argv[1] #TrimmedVideos/video/valid
test_label_path = sys.argv[2] #TrimmedVideos/label/gt_valid.csv
save_path = sys.argv[3] #TrimmedVideos/label/gt_valid.csv
mode = sys.argv[4] #test
dict = getVideoList(test_label_path)
if mode != "test": action_labels = (dict['Action_labels'])
video_names = (dict['Video_name'])
video_categorys = (dict['Video_category'])
total_num = len(video_names)


# loading videos
test_videos = []
test_labels = []
print("\nloading videos...")
with tqdm(total=total_num) as pbar:
    for i,(video_category, video_name) in enumerate(zip(video_categorys,video_names)):
            train_video = readShortVideo(video_path, video_category, video_name)
            test_videos.append(train_video)
            pbar.update(1)
if mode != "test":
    print("\nloading labels...")
    for i,action_label in enumerate(action_labels):
        test_labels.append(int(action_label))
print("test_videos_len:",len(test_videos))


# extracting features
cnn_feature_extractor = Resnet50().cuda() # to 2048 dims
transform=transforms.Compose([
                        transforms.ToPILImage(),
                        transforms.Pad((0,40),fill=0,padding_mode='constant'),
                        transforms.Resize(224),
                        transforms.ToTensor(),
                        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                        ])
cnn_feature_extractor.eval()
train_features = []
with torch.no_grad():
    print("\nextracting videos feature...")
    with tqdm(total=total_num) as pbar:
        for train_video in test_videos:
            local_batch = []
            for frame in train_video:
                frame = transform(frame)
                local_batch.append(frame)
            local_batch = torch.stack(local_batch)
            features = cnn_feature_extractor(local_batch.cuda())
            train_features.append(features) # here is the different, we output hole batch feature as nx2048d dim
            #print(features.shape)
            pbar.update(1)


# save feature torch as file
# load feature file as torch for test
torch.save(train_features, os.path.join(save_path,mode+'_features.pt'))
train_features = torch.load(os.path.join(save_path,mode+'_features.pt'))
print(mode+"_features:",len(train_features))

if mode != "test":
    test_labels = torch.LongTensor(test_labels)
    torch.save(test_labels, os.path.join(save_path,mode+'_vals.pt'))
    test_labels = torch.load(os.path.join(save_path,mode+'_vals.pt'))
    print(mode+"_labels:",test_labels.shape)
