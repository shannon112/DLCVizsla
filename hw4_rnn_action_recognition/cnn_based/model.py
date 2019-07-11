import torch.nn as nn
import torchvision.models as models
#from torchsummary import summary


class Net(nn.Module):
    def __init__(self, feature_size):
        super(Net, self).__init__()

        self.linear1 = nn.Linear(feature_size,4096)
        self.linear2 = nn.Linear(4096,1024)
        self.linear3 = nn.Linear(1024, 11)
        self.softmax = nn.Softmax(1)
        self.relu = nn.ReLU()
        self.bn_1 = nn.BatchNorm1d(4096)
        self.bn_2 = nn.BatchNorm1d(1024)

    def forward(self, x):
        x = self.relu(self.bn_1(self.linear1(x))) # same as relu output
        x = self.relu(self.bn_2(self.linear2(x)))
        y_pred = self.softmax(self.linear3(x))
        return y_pred,x


class Vgg16(nn.Module):
    def __init__(self):
        super(Vgg16, self).__init__()
        model = models.vgg16(pretrained=True)
        self.feature = nn.Sequential(*list(model.children())[:-1])

    def forward(self, x):
        """
            input shape: 224 x 224
            output shape: batch size x 512 x 7 x 7
        """
        output = self.feature(x)
        return output


class Resnet50(nn.Module):
    def __init__(self):
        super(Resnet50, self).__init__()
        model = models.resnet50(pretrained=True)
        self.feature = nn.Sequential(*list(model.children())[:-1])

    def forward(self, x):
        """
            input shape: 224 x 224
            output shape: batch size x 2048 x 1 x 1
        """
        output = self.feature(x)
        output = output.view(-1,2048)
        return output


if __name__ == '__main__':
    net = Vgg16().cuda()
    #summary(net, input_size=(3, 224, 224))

    net2 = Resnet50().cuda()
    #summary(net2, input_size=(3, 224, 224))
