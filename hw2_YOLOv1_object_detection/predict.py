# usage:
# python predict2.py ../hw2_train_val/val1500/images/ ../hw2_train_val/val1500/labelpre/
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import os
import sys
import numpy as np

# import self-made function
from dataset_predict import hw2DataSet
from models import Yolov1_vgg16bn
from yoloLoss import yoloLoss

def nms(bboxes, nms_threshold):
    '''
    # bboxes (?,7)
    # [x1,y1,x2,y2,bbox_conf_score,preclass_index,class_spec_conf_score]
    '''
    if bboxes.shape[0]==0:
        return np.array([])
    x1,y1 = bboxes[:, 0], bboxes[:, 1]
    x2,y2 = bboxes[:, 2], bboxes[:, 3]
    scores = bboxes[:, 4]
    #scores = bboxes[:, 6]

    area_boxes = (x2 - x1 + 1) * (y2 - y1 + 1)
    order_indexs = scores.argsort()[::-1]
    final_keep = []
    while order_indexs.size > 0:
        selected_index = order_indexs[0]
        final_keep.append(selected_index)
        xx1 = np.maximum(x1[selected_index], x1[order_indexs[1:]])
        yy1 = np.maximum(y1[selected_index], y1[order_indexs[1:]])
        xx2 = np.minimum(x2[selected_index], x2[order_indexs[1:]])
        yy2 = np.minimum(y2[selected_index], y2[order_indexs[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        area_overlap = w * h
        iou = area_overlap / (area_boxes[selected_index] + area_boxes[order_indexs[1:]] - area_overlap)
        keep_indexs = np.where(iou <= nms_threshold)[0]  # here index is 1: not contain selected_index
        order_indexs = order_indexs[keep_indexs + 1] # so we +1 back to origin

    return bboxes[final_keep]


def load_checkpoint(checkpoint_path, model,device):
    state = torch.load(checkpoint_path, map_location="cuda") # for cuda
    #state = torch.load(checkpoint_path, map_location=device) #for cpu
    model.load_state_dict(state['state_dict'])
    print('model loaded from %s' % checkpoint_path)


def writeToFile(final_bboxes,pre_label_dir,label_fn,classes):
    ans_lines = []
    for j in range(len(final_bboxes)):
        x1 = str(final_bboxes[j][0])
        y1 = str(final_bboxes[j][1])
        x2 = str(final_bboxes[j][2])
        y2 = str(final_bboxes[j][3])
        class_ = classes[int(final_bboxes[j][5])]
        score = str(final_bboxes[j][6])
        ans_line = "{} {} {} {} {} {} {} {} {} {}".format(
                    x1,y1,x2,y1,x2,y2,x1,y2,class_,score)
        ans_lines.append(ans_line)

    with open(os.path.join(pre_label_dir,label_fn),'w') as f:
        for j,line in enumerate(ans_lines):
            f.write(line)
            if (j==len(ans_lines)-1): break
            f.write('\n')

def predict(device,model,predictset_loader,predictset,classes,pre_label_dir):
    # arguements
    filter_lowScore = 0.1
    nms_threshold = 0.4

    with torch.no_grad():
        loader=0
        for batch_idx,(images,labels_fn) in enumerate(predictset_loader):
            images = images.to(device)
            batchsize = len(labels_fn)
            preds = model(images)
            preds = preds.to(device)
            loader+= batchsize

            for i in range(batchsize): # load one prediction in batch
                pred = preds[i]
                label_fn = labels_fn[i]
                bboxes = predictset.decoder(pred,filter_lowScore)
                final_bboxes = nms(bboxes,nms_threshold)
                writeToFile(final_bboxes,pre_label_dir,label_fn,classes)

            if(loader%10==0): print ("%.2f" % (100*loader/len(predictset)),"%")


def main():
    # read images and labels
    image_dir, pre_label_dir, model_fn = sys.argv[1], sys.argv[2], sys.argv[3]
    os.system('rm -rf '+pre_label_dir)
    os.system('mkdir -p '+pre_label_dir)

    # using cuda
    use_cuda = torch.cuda.is_available()
    torch.manual_seed(123)
    device = torch.device("cuda" if use_cuda else "cpu")
    print('Device used:', device)

    # loading predictset images and label_file_names
    predictset = hw2DataSet(root=image_dir, transform=transforms.ToTensor())
    print('# images in predictset:', len(predictset))
    predictset_loader = DataLoader(predictset, batch_size=16, shuffle=False, num_workers=1)
    classes = predictset.classesL

    # loading model
    model = Yolov1_vgg16bn(pretrained=True).to(device)
    load_checkpoint(model_fn,model,device)
    model.eval()

    # predict
    predict(device,model,predictset_loader,predictset,classes,pre_label_dir)

if __name__ == '__main__':
    main()
