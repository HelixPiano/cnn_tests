import torch
import torch.nn as nn
import torchvision.models


class Network(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=32, kernel_size=(3, 7)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=(1, 1)),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 7)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=(1, 1)),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 7)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=(1, 1)),
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=29, kernel_size=(3, 7)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=(1, 1)),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        # self.layer5 = nn.Sequential(
        #     nn.Linear(in_features=256, out_features=29),
        # )

    def forward(self, nn_input: torch.FloatTensor):
        x = self.layer1(nn_input)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = torch.flatten(x, 1)
        # x = self.layer5(x)
        return x


class SCNNB(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=32, kernel_size=(3, 3)),
            nn.LazyBatchNorm2d(),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2)),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3)),
            nn.LazyBatchNorm2d(),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2)),
        )
        self.layer3 = nn.Sequential(
            nn.LazyLinear(out_features=1280),
            nn.Dropout(0.5),
            nn.LazyLinear(out_features=29),
        )

    def forward(self, nn_input: torch.FloatTensor):
        x = self.layer1(nn_input)
        x = self.layer2(x)
        x = torch.flatten(x, 1)
        x = self.layer3(x)
        return x


def load_efficientnet():
    network = torchvision.models.efficientnet_v2_s()
    network.features[0][0] = nn.Conv2d(in_channels=2, out_channels=24, kernel_size=(3, 3), stride=(2, 2),
                                       padding=(1, 1), bias=False)
    network.classifier[1] = nn.Linear(in_features=1280, out_features=29, bias=True)
    return network


def load_convnext():
    network = torchvision.models.convnext_tiny()
    network.classifier[2] = nn.Linear(in_features=768, out_features=29, bias=True)
    return network


def load_maxvit():
    network = torchvision.models.maxvit_t()
    network.classifier[2] = nn.Linear(in_features=512, out_features=29, bias=True)
    return network


def load_swintransformer():
    network = torchvision.models.swin_t()
    network.features[0][0] = nn.Conv2d(2, 96, kernel_size=(4, 4), stride=(4, 4))
    network.head = nn.Linear(in_features=768, out_features=29, bias=True)
    return network


def load_regnet():
    network = torchvision.models.regnet_y_400mf()
    network.stem[0] = nn.Conv2d(2, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
    network.fc = nn.Linear(in_features=440, out_features=29, bias=True)
    return network


def load_wideresnet():
    network = torchvision.models.wide_resnet50_2()
    network.conv1 = nn.Conv2d(2, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    network.fc = nn.Linear(in_features=2048, out_features=29, bias=True)
    return network


def load_squeezenet():
    network = torchvision.models.squeezenet1_1()
    network.classifier[1] = nn.Conv2d(512, 29, kernel_size=(1, 1), stride=(1, 1))
    return network
