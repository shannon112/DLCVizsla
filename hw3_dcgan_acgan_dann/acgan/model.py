import constants as consts
import torch.nn as nn
import torch.nn.functional as F
import math
import torch
import numpy as np
import constants as consts

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.label_emb = nn.Embedding(consts.n_classes, consts.nz)
        self.conv1 = nn.ConvTranspose2d(consts.nz, consts.ngf * 8, 4, 1, 0)
        self.bn1 = nn.BatchNorm2d(consts.ngf * 8)

        self.conv2 = nn.ConvTranspose2d(consts.ngf * 8, consts.ngf * 4, 4, 2, 1)
        self.bn2 = nn.BatchNorm2d(consts.ngf * 4)

        self.conv3 = nn.ConvTranspose2d(consts.ngf * 4, consts.ngf * 2, 4, 2, 1)
        self.bn3 = nn.BatchNorm2d(consts.ngf * 2)

        self.conv4 = nn.ConvTranspose2d(consts.ngf * 2, consts.ngf * 1, 4, 2, 1)
        self.bn4 = nn.BatchNorm2d(consts.ngf * 1)

        self.conv5 = nn.ConvTranspose2d(consts.ngf * 1, consts.nc, 4, 2, 1)

        self.relu = nn.ReLU(True)
        self.tanh = nn.Tanh()

    def forward(self, noise, labels):
        gen_input = torch.mul(self.label_emb(labels), noise)
        x = gen_input.view(gen_input.size(0), -1, 1, 1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)

        x = self.conv5(x)
        output = self.tanh(x)
        return output


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.ndf = consts.ndf
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)
        self.conv1 = nn.Conv2d(consts.nc, consts.ndf, 4, 2, 1)
        self.conv2 = nn.Conv2d(consts.ndf, consts.ndf * 2, 4, 2, 1)
        self.bn2 = nn.BatchNorm2d(consts.ndf * 2)
        self.conv3 = nn.Conv2d(consts.ndf * 2, consts.ndf * 4, 4, 2, 1)
        self.bn3 = nn.BatchNorm2d(consts.ndf * 4)
        self.conv4 = nn.Conv2d(consts.ndf * 4, consts.ndf * 8, 4, 2, 1)
        self.bn4 = nn.BatchNorm2d(consts.ndf * 8)
        self.conv5 = nn.Conv2d(consts.ndf * 8, consts.ndf * 1, 4, 1, 0)
        self.gan_linear = nn.Linear(consts.ndf * 1, 1)
        self.aux_linear = nn.Linear(consts.ndf * 1, consts.n_classes)

        self.sigmoid = nn.Sigmoid()

    def forward(self, input):

        x = self.conv1(input)
        x = self.lrelu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.lrelu(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.lrelu(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.lrelu(x)

        x = self.conv5(x)
        x = x.view(-1, self.ndf * 1)
        c = self.aux_linear(x)

        s = self.gan_linear(x)
        s = self.sigmoid(s)
        return s.squeeze(1), c.squeeze(1)
