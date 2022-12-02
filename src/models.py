import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

nclasses = 20

class ResNetFeatures(nn.Module):
    def __init__(self, pathFeatures=None):
        super(ResNetFeatures, self).__init__()
        
        if pathFeatures is not None:
            self.features = self.load_model(pathFeatures)

        else:
            resnet = torchvision.models.resnet50(pretrained=True)

            self.features = nn.Sequential(*list(resnet.children())[:-1])

        for param in self.features.parameters():
            param.requires_grad = False

        self.features.eval()
        
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


class NN(nn.Module):
    def __init__(self, layers = [1024, 256, 128], in_layer=2048):
        super(NN, self).__init__()

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