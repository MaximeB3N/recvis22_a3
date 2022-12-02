import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

nclasses = 20 

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv3 = nn.Conv2d(20, 20, kernel_size=5)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, nclasses)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.relu(F.max_pool2d(self.conv3(x), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

class ResNetFeatures(nn.Module):
    def __init__(self, pathFeatures=None):
        super(ResNetFeatures, self).__init__()
        
        if pathFeatures is not None:
            self.features = self.load_model(pathFeatures)

        else:
            resnet = torchvision.models.resnet34(pretrained=True)

            self.features = nn.Sequential(*list(resnet.children())[:-1])

            # for param in self.features.parameters():
            #     param.requires_grad = False

            # self.features.eval()
        
    def forward(self, x):
        x = self.features(x)
        # print(x.shape)
        return x

    def load_model(self, path):
        model = ResNetRetrain()
        model.load_state_dict(torch.load(path))

        # remove the last layer
        model = nn.Sequential(*list(model.children())[:-1],
                              list(model.children())[-1][0])

        # print(model)
        return model


class NN2(nn.Module):
    def __init__(self, layers = [1024, 256, 128], in_layer=2048):
        super(NN2, self).__init__()

        self.in_layer = in_layer
        self.dropout0 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(self.in_layer, layers[0])
        self.batchnorm1 = nn.BatchNorm1d(layers[0])
        self.dropout1 = nn.Dropout(0.5)

        self.fc2 = nn.Linear(layers[0], layers[1])
        self.batchnorm2 = nn.BatchNorm1d(layers[1])
        self.dropout2 = nn.Dropout(0.3)

        self.fc3 = nn.Linear(layers[1], layers[2])
        self.batchnorm3 = nn.BatchNorm1d(layers[2])
        self.dropout3 = nn.Dropout(0.1)

        self.fc4 = nn.Linear(layers[2], nclasses)

        print('The number of parameters in the model is: {}'.format(sum([p.numel() for p in self.parameters()])))

    def forward(self, x):
        x = x.view(-1, self.in_layer)
        x = self.dropout0(x)
        x = F.relu(self.batchnorm1(self.fc1(x)))
        x = self.dropout1(x)
        x = F.relu(self.batchnorm2(self.fc2(x)))
        x = self.dropout2(x)
        x = F.relu(self.batchnorm3(self.fc3(x)))
        x = self.dropout3(x)
        x = self.fc4(x)

        return x # F.softmax(x, dim=1)

class NN(nn.Module):
    def __init__(self):
        super(NN, self).__init__()
        self.fc1 = nn.Linear(2048, 512)
        self.batchnorm1 = nn.BatchNorm1d(512)
        self.dropout1 = nn.Dropout(0.7)

        self.fc2 = nn.Linear(512, 512)
        self.batchnorm2 = nn.BatchNorm1d(512)
        self.dropout2 = nn.Dropout(0.5)

        self.fc3 = nn.Linear(512, 256)
        self.batchnorm3 = nn.BatchNorm1d(256)
        self.dropout3 = nn.Dropout(0.3)

        self.fc4 = nn.Linear(256, 128)
        self.batchnorm4 = nn.BatchNorm1d(128)
        self.dropout4 = nn.Dropout(0.1)

        self.fc5 = nn.Linear(128, nclasses)

    def forward(self, x):
        x = x.view(-1, 2048)
        x = F.relu(self.batchnorm1(self.fc1(x)))
        x = self.dropout1(x)
        x = F.relu(self.batchnorm2(self.fc2(x)))
        x = self.dropout2(x)
        x = F.relu(self.batchnorm3(self.fc3(x)))
        x = self.dropout3(x)
        x = F.relu(self.batchnorm4(self.fc4(x)))
        x = self.dropout4(x)
        x = self.fc5(x)
        return x # F.softmax(x, dim=1)

class ResNetRetrain(nn.Module):
    def __init__(self) -> None:
        super(ResNetRetrain, self).__init__()

        resnet = torchvision.models.resnet50(pretrained=True)

        # initialize the three layers

        self.before_layer4 = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            nn.Sequential(
                *list(resnet.layer4.children())[:-1]
            )
        )
        
        # freeze the layers
        for param in self.before_layer4.parameters():
            param.requires_grad = False

        # print(before_layer4)

        self.layer4 = list(resnet.layer4.children())[-1]

        for param in self.layer4.parameters():
            param.requires_grad = False

        self.layer4.conv3.__init__(resnet.inplanes, 512)
        # print(layer4)

        self.classififier = torch.nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(2048, nclasses)
        )

        print('The number of parameters in the model is: {}'.format(sum([p.numel() for p in self.parameters() if p.requires_grad])))

    def forward(self, x):
        x = self.before_layer4(x)
        x = self.layer4(x)
        x = self.classififier(x)
        return x




class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.resnet = torchvision.models.resnet50(pretrained=True)
        self.num_ftrs = self.resnet.fc.in_features
        self.ninter = 256
        
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])

        # Freeze model weights

        for param in self.resnet.parameters():
            param.requires_grad = False

        #
        
        # Replace the last fully-connected layer
        # Parameters of newly constructed modules have requires_grad=True by default
        
        # print(num_ftrs)

        # self
        self.dropout1 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(self.num_ftrs, self.ninter)
        self.fc2 = nn.Linear(self.ninter, nclasses)
        self.dropout2 = nn.Dropout(0.2)
        self.batchnorm_lin = nn.BatchNorm1d(self.ninter)


    def forward(self, x):
        x = self.resnet(x)
        print(x.shape)
        x = F.relu(x)
        x = self.dropout1(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.batchnorm_lin(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return x