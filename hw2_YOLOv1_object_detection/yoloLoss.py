import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class yoloLoss(nn.Module):
    def __init__(self,l_coord,l_noobj):
        super(yoloLoss,self).__init__()
        self.l_coord = l_coord # 5
        self.l_noobj = l_noobj # 0.5

    def compute_iou(self, box_pred, box_tar):
        '''Compute the  overlapsection over union of two set of boxes, each box is [x1,y1,x2,y2].
        Args:
          box_pred: (tensor) bounding boxes, sized [N,4].
          box_tar: (tensor) bounding boxes, sized [M,4].
        Return:
          (tensor) iou, sized [N,M].
        '''
        N = box_pred.size(0)
        M = box_tar.size(0)
        box_pred_lt = box_pred[:,:2].unsqueeze(1).expand(N,M,2)  # [N,2] -> [N,1,2] -> [N,M,2]
        box_pred_rb = box_pred[:,2:].unsqueeze(1).expand(N,M,2)  # [N,2] -> [N,1,2] -> [N,M,2]
        box_tar_lt = box_tar[:,:2].unsqueeze(0).expand(N,M,2)  # [M,2] -> [1,M,2] -> [N,M,2]
        box_tar_rb = box_tar[:,2:].unsqueeze(0).expand(N,M,2)  # [M,2] -> [1,M,2] -> [N,M,2]
        box_lt = torch.max(box_pred_lt, box_tar_lt)
        box_rb = torch.min(box_pred_rb, box_tar_rb)

        overlap_wh = box_rb - box_lt  # [N,M,2]
        overlap_wh[overlap_wh<0] = 0  # clip at 0
        area_overlap = overlap_wh[:,:,0] * overlap_wh[:,:,1]  # [N,M]

        box_pred_wh = box_pred[:,2:] - box_pred[:,:2]
        area_pred = box_pred_wh[:,0] * box_pred_wh[:,1]
        area_pred = area_pred.unsqueeze(1).expand_as( area_overlap)  # [N,] -> [N,1] -> [N,M]

        box_tar_wh = box_tar[:,2:] - box_tar[:,:2]
        area_tar = box_tar_wh[:,0] * box_tar_wh[:,1]
        area_tar = area_tar.unsqueeze(0).expand_as( area_overlap)  # [M,] -> [1,M] -> [N,M]
        return area_overlap / (area_pred + area_tar -  area_overlap)

    def forward(self,pred_tensor,target_tensor):
        '''
        pred_tensor: (tensor) size(batchsize,S,S,Bx5+16=26) [x,y,w,h,c]
        target_tensor: (tensor) size(batchsize,S,S,26)
        '''
        batchsize = pred_tensor.size()[0] # batch size
        # mask
        # coo(c=1, have object) + noo(c=0, no object) = all
        # bool array over 7x7patches. 4 is confidence of box#1 torch.Size([batch, 7, 7])
        coo_mask = target_tensor[:,:,:,4] > 0
        noo_mask = target_tensor[:,:,:,4] == 0
        # squeeze torch.Size([batch ,7, 7, 1])
        # expand_as torch.Size([batch ,7, 7, 26]) fill 26 as origin bool 0 or 1
        #coo_mask = coo_mask.unsqueeze(-1).expand_as(target_tensor)
        #noo_mask = noo_mask.unsqueeze(-1).expand_as(target_tensor)

        # coo pred boxes = (2*remainPatches,5)
        # coo pred classes = (remain,16)
        # noo pred 26 = (remain,26)
        coo_pred = pred_tensor[coo_mask].view(-1,26) # torch.Size([remain patches, 26])
        coo_box_pred = coo_pred[:,:10].contiguous().view(-1,5) # box[x1,y1,w1,h1,c1] [x2,y2,w2,h2,c2]
        coo_class_pred = coo_pred[:,10:]                       # class
        noo_pred = pred_tensor[noo_mask].view(-1,26)

        # coo target boxes = (2*remainPatches,5)
        # coo target classes = (remain,16)
        # noo target 26 = (remain,26)
        coo_target = target_tensor[coo_mask].view(-1,26)
        coo_box_target = coo_target[:,:10].contiguous().view(-1,5)
        coo_class_target = coo_target[:,10:]
        noo_target = target_tensor[noo_mask].view(-1,26)


        ###############################################
        # compute not contain obj loss (noo_obj_c_loss)   penalize box#2
        ###############################################
        noo_pred_mask = torch.cuda.ByteTensor(noo_pred.size())
        noo_pred_mask.zero_()
        noo_pred_mask[:,4], noo_pred_mask[:,9] = 1, 1
        #only compute box#1 and box#2 c loss size[-1,2]
        noo_pred_c = noo_pred[noo_pred_mask]
        noo_target_c = noo_target[noo_pred_mask]
        noo_obj_c_loss = F.mse_loss(noo_pred_c,noo_target_c,reduction='sum')


        ###############################################
        # class loss   (class_loss)(purple)
        ###############################################
        class_loss = F.mse_loss(coo_class_pred,coo_class_target,reduction='sum')


        ###############################################
        # compute contain obj loss (xy_loss, wh_loss, confi_loss_response, confi_loss_not_response)
        ###############################################
        # initialize mask
        coo_response_mask = torch.cuda.ByteTensor(coo_box_target.size())
        coo_response_mask.zero_()
        coo_not_response_mask = torch.cuda.ByteTensor(coo_box_target.size())
        coo_not_response_mask.zero_()
        coo_box_target_iou = torch.cuda.FloatTensor(coo_box_target.size())
        coo_box_target_iou.zero_()
        patch_size = 1. / 7

        # choose the best iou box
        for i in range(0,coo_box_target.size()[0],2): # step=2
            box_pred = coo_box_pred[i:i+2] # two box of same patch
            box_pred_xyxy = torch.cuda.FloatTensor(box_pred.size())
            box_pred_xyxy[:,:2] = box_pred[:,:2]*patch_size - box_pred[:,2:4]*0.5 # left-top
            box_pred_xyxy[:,2:4] = box_pred[:,:2]*patch_size + box_pred[:,2:4]*0.5 # right-bottom

            box_tar = coo_box_target[i].view(-1,5)
            box_tar_xyxy = torch.cuda.FloatTensor(box_tar.size())
            box_tar_xyxy[:,:2] = box_tar[:,:2]*patch_size - box_tar[:,2:4]*0.5 # left-top
            box_tar_xyxy[:,2:4] = box_tar[:,:2]*patch_size + box_tar[:,2:4]*0.5 # right-bottom

            iou = self.compute_iou(box_pred_xyxy[:,:4],box_tar_xyxy[:,:4]) #[2,1]
            max_iou, max_index = torch.max(iou,0)  # index = 0 or 1
            coo_response_mask[i+max_index]=1   #get boxes responsible for the objects
            coo_not_response_mask[i+1-max_index]=1  #get boxes not responsible for the objects
            coo_box_target_iou[i+max_index,4] = max_iou # save iou

        # confidence score = 1*IoU (between the predicted box and the target(ground truth) box)
        #1.response loss (blue+red)
        coo_box_pred_response = coo_box_pred[coo_response_mask].view(-1,5)
        coo_box_target_response = coo_box_target[coo_response_mask].view(-1,5)
        coo_box_target_response_iou = coo_box_target_iou[coo_response_mask].view(-1,5)
        xy_loss = F.mse_loss(coo_box_pred_response[:,:2], coo_box_target_response[:,:2],reduction='sum')
        wh_loss = F.mse_loss(torch.sqrt(coo_box_pred_response[:,2:4]),torch.sqrt(coo_box_target_response[:,2:4]),reduction='sum')
        confi_loss_response = F.mse_loss(coo_box_pred_response[:,4],coo_box_target_response_iou[:,4],reduction='sum')

        #2.not response loss (yellow)
        coo_box_pred_not_response = coo_box_pred[coo_not_response_mask].view(-1,5)
        coo_box_target_not_response = coo_box_target[coo_not_response_mask].view(-1,5)
        coo_box_target_not_response[:,4]= 0  # not responsible for object c = 0
        confi_loss_not_response = F.mse_loss(coo_box_pred_not_response[:,4], coo_box_target_not_response[:,4],reduction='sum')

        #print(self.l_coord*xy_loss , self.l_coord*wh_loss , confi_loss_response , confi_loss_not_response , self.l_noobj*noo_obj_c_loss , class_loss)
        #print( (self.l_coord*xy_loss + self.l_coord*wh_loss + 2*confi_loss_response + confi_loss_not_response + self.l_noobj*noo_obj_c_loss + class_loss) / batchsize)
        return (self.l_coord*xy_loss + self.l_coord*wh_loss + 2*confi_loss_response + confi_loss_not_response + self.l_noobj*noo_obj_c_loss + class_loss )/ batchsize
