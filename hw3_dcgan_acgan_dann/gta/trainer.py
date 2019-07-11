import torch
import torch.nn as nn
from torch.autograd import Variable, Function
import torch.optim as optim
import torchvision.utils as vutils
import datetime
import numpy as np
import models
import utils
import constants as consts


class GTA(object):

    def __init__(self, mean, std, source_trainloader, source_valloader, targetloader):

        self.source_trainloader = source_trainloader
        self.source_valloader = source_valloader
        self.targetloader = targetloader
        self.mean = mean
        self.std = std
        self.best_val = 0
        self.cuda = True if torch.cuda.is_available() else False

        # Defining networks and optimizers
        self.netG = models._netG()
        self.netD = models._netD()
        self.netF = models._netF()
        self.netC = models._netC()

        # Weight initialization
        self.netG.apply(utils.weights_init)
        self.netD.apply(utils.weights_init)
        self.netF.apply(utils.weights_init)
        self.netC.apply(utils.weights_init)

        # Defining loss criterions
        self.criterion_c = nn.CrossEntropyLoss()
        self.criterion_s = nn.BCELoss()

        if self.cuda:
            self.netD.cuda()
            self.netG.cuda()
            self.netF.cuda()
            self.netC.cuda()
            self.criterion_c.cuda()
            self.criterion_s.cuda()

        # Defining optimizers
        self.optimizerD = optim.Adam(self.netD.parameters(), lr=consts.lr, betas=(consts.beta1, 0.999))
        self.optimizerG = optim.Adam(self.netG.parameters(), lr=consts.lr, betas=(consts.beta1, 0.999))
        self.optimizerF = optim.Adam(self.netF.parameters(), lr=consts.lr, betas=(consts.beta1, 0.999))
        self.optimizerC = optim.Adam(self.netC.parameters(), lr=consts.lr, betas=(consts.beta1, 0.999))

        # Other variables
        self.real_label_val = 1
        self.fake_label_val = 0

    """
    Validation function
    """
    def validate(self, epoch):

        self.netF.eval()
        self.netC.eval()
        total = 0
        correct = 0

        # Testing the model
        for i, datas in enumerate(self.source_valloader):
            inputs, labels = datas
            inputv, labelv = Variable(inputs.cuda(), volatile=True), Variable(labels.cuda())

            outC = self.netC(self.netF(inputv))
            _, predicted = torch.max(outC.data, 1)
            total += labels.size(0)
            correct += ((predicted == labels.cuda()).sum())

        val_acc = 100*float(correct)/total
        print('%s| Epoch: %d, Val Accuracy: %f %%' % (datetime.datetime.now(), epoch, val_acc))

        # Saving checkpoints
        torch.save(self.netF.state_dict(), 'models/netF.pth')
        torch.save(self.netC.state_dict(), 'models/netC.pth')

        if val_acc>self.best_val:
            self.best_val = val_acc
            torch.save(self.netF.state_dict(), 'models/model_best_netF.pth')
            torch.save(self.netC.state_dict(), 'models/model_best_netC.pth')


    """
    Train function
    """
    def train(self):

        curr_iter = 0

        reallabel = torch.FloatTensor(consts.batch_size).fill_(self.real_label_val)
        fakelabel = torch.FloatTensor(consts.batch_size).fill_(self.fake_label_val)
        if self.cuda:
            reallabel, fakelabel = reallabel.cuda(), fakelabel.cuda()
        reallabelv = Variable(reallabel)
        fakelabelv = Variable(fakelabel)

        print(self.netG)
        print(self.netF)
        print(self.netC)
        print(self.netD)

        for epoch in range(consts.num_epochs):

            self.netG.train()
            self.netF.train()
            self.netC.train()
            self.netD.train()

            for i, (datas, datat) in enumerate(zip(self.source_trainloader, self.targetloader)):

                ###########################
                # Forming input variables
                ###########################

                src_inputs, src_labels = datas
                tgt_inputs, __ = datat
                src_inputs_unnorm = (((src_inputs*self.std[0]) + self.mean[0]) - 0.5)*2

                # Creating one hot vector
                labels_onehot = np.zeros((consts.batch_size, consts.nclasses+1), dtype=np.float32)
                for num in range(consts.batch_size):
                    labels_onehot[num, src_labels[num]] = 1
                src_labels_onehot = torch.from_numpy(labels_onehot)

                labels_onehot = np.zeros((consts.batch_size, consts.nclasses+1), dtype=np.float32)
                for num in range(consts.batch_size):
                    labels_onehot[num, consts.nclasses] = 1
                tgt_labels_onehot = torch.from_numpy(labels_onehot)

                if self.cuda:
                    src_inputs, src_labels = src_inputs.cuda(), src_labels.cuda()
                    src_inputs_unnorm = src_inputs_unnorm.cuda()
                    tgt_inputs = tgt_inputs.cuda()
                    src_labels_onehot = src_labels_onehot.cuda()
                    tgt_labels_onehot = tgt_labels_onehot.cuda()

                # Wrapping in variable
                src_inputsv, src_labelsv = Variable(src_inputs), Variable(src_labels)
                src_inputs_unnormv = Variable(src_inputs_unnorm)
                tgt_inputsv = Variable(tgt_inputs)
                src_labels_onehotv = Variable(src_labels_onehot)
                tgt_labels_onehotv = Variable(tgt_labels_onehot)

                ###########################
                # Updates
                ###########################

                # Updating D network

                self.netD.zero_grad()
                src_emb = self.netF(src_inputsv)
                src_emb_cat = torch.cat((src_labels_onehotv, src_emb), 1)
                src_gen = self.netG(src_emb_cat)

                tgt_emb = self.netF(tgt_inputsv)
                tgt_emb_cat = torch.cat((tgt_labels_onehotv, tgt_emb),1)
                tgt_gen = self.netG(tgt_emb_cat)

                src_realoutputD_s, src_realoutputD_c = self.netD(src_inputs_unnormv)
                errD_src_real_s = self.criterion_s(src_realoutputD_s, reallabelv)
                errD_src_real_c = self.criterion_c(src_realoutputD_c, src_labelsv)

                src_fakeoutputD_s, src_fakeoutputD_c = self.netD(src_gen)
                errD_src_fake_s = self.criterion_s(src_fakeoutputD_s, fakelabelv)

                tgt_fakeoutputD_s, tgt_fakeoutputD_c = self.netD(tgt_gen)
                errD_tgt_fake_s = self.criterion_s(tgt_fakeoutputD_s, fakelabelv)

                errD = errD_src_real_c + errD_src_real_s + errD_src_fake_s + errD_tgt_fake_s
                errD.backward(retain_graph=True)
                self.optimizerD.step()


                # Updating G network

                self.netG.zero_grad()
                src_fakeoutputD_s, src_fakeoutputD_c = self.netD(src_gen)
                errG_c = self.criterion_c(src_fakeoutputD_c, src_labelsv)
                errG_s = self.criterion_s(src_fakeoutputD_s, reallabelv)
                errG = errG_c + errG_s
                errG.backward(retain_graph=True)
                self.optimizerG.step()


                # Updating C network

                self.netC.zero_grad()
                outC = self.netC(src_emb)
                errC = self.criterion_c(outC, src_labelsv)
                errC.backward(retain_graph=True)
                self.optimizerC.step()


                # Updating F network

                self.netF.zero_grad()
                errF_fromC = self.criterion_c(outC, src_labelsv)

                src_fakeoutputD_s, src_fakeoutputD_c = self.netD(src_gen)
                errF_src_fromD = self.criterion_c(src_fakeoutputD_c, src_labelsv)*(consts.adv_weight)

                tgt_fakeoutputD_s, tgt_fakeoutputD_c = self.netD(tgt_gen)
                errF_tgt_fromD = self.criterion_s(tgt_fakeoutputD_s, reallabelv)*(consts.adv_weight*consts.alpha)

                errF = errF_fromC + errF_src_fromD + errF_tgt_fromD
                errF.backward()
                self.optimizerF.step()

                curr_iter += 1

                # Learning rate scheduling
                if consts.lrd:
                    self.optimizerD = utils.exp_lr_scheduler(self.optimizerD, epoch, consts.lr, consts.lrd, curr_iter)
                    self.optimizerF = utils.exp_lr_scheduler(self.optimizerF, epoch, consts.lr, consts.lrd, curr_iter)
                    self.optimizerC = utils.exp_lr_scheduler(self.optimizerC, epoch, consts.lr, consts.lrd, curr_iter)

            # Validate every epoch
            self.validate(epoch+1)
