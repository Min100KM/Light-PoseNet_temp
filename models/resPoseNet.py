import torch
import torch.nn as nn
from torch.nn import init
from torch.nn import functional as F
import functools
from torch.autograd import Variable
from torch.optim import lr_scheduler
import numpy as np
from torchsummary import summary


###############################################################################
# Functions
###############################################################################

def weight_init_resnet(key, module, weights=None):

    if weights is None:
        init.constant_(module.bias.data, 0.0)
        if key == "XYZ":
            init.normal_(module.weight.data, 0.0, 0.5)
        elif key == "WPQR":
            init.normal_(module.weight.data, 0.0, 0.01)
        else:
            init.normal_(module.weight.data, 0.0, 0.01)
    else:
        # print(key, weights[(key+"_1").encode()].shape, module.bias.size())
        module.bias.data[...] = torch.from_numpy(weights[(key + "_1").encode()])
        module.weight.data[...] = torch.from_numpy(weights[(key + "_0").encode()])
    return module


def get_scheduler(optimizer, opt):
    if opt.lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + 1 + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def define_network(input_nc, model, init_from=None, isKD=False, isTest=False, gpu_ids=[]):
    netG = None
    use_gpu = len(gpu_ids) > 0

    if use_gpu:
        assert (torch.cuda.is_available())

    if model == 'resnet18':
        netG = ResNet18(3, ResBlock, weights=init_from, isKD=False)

    elif model == 'resnet34':
        netG = ResNet34(3, ResBlock, weights=init_from)

    elif model == 'resnet50':
        netG = ResNet50(3, ResBottleneckBlock, weights=init_from)

    elif model == 'resnet101':
        netG = ResNet101(3, ResBottleneckBlock, weights=init_from)

    else:
        raise NotImplementedError('Model name [%s] is not recognized' % model)
    if len(gpu_ids) > 0:
        netG.cuda(gpu_ids[0])
    return netG


# defines the regression module for resnet
class Regression(nn.Module):
    def __init__(self, weights=None, input_cls=512, mid_cls=1024):
        super(Regression, self).__init__()

        dropout_rate = 0.5
        #dropout_rate = 0.7
        #input_cls = [512, 2048]

        self.dropout = nn.Dropout(p=dropout_rate)

        #self.projection = nn.AvgPool2d(kernel_size=7, stride=1)
        self.cls_fc_pose = nn.Sequential(*[weight_init_resnet("pose", nn.Linear(input_cls, mid_cls)),
                                           nn.ReLU(inplace=True)])
        self.cls_fc_xy = weight_init_resnet("XYZ", nn.Linear(mid_cls, 3))
        self.cls_fc_wpqr = weight_init_resnet("WPQR", nn.Linear(mid_cls, 4))


    def forward(self, input):
        #output = self.projection(input)
        #output = self.cls_fc_pose(output.view(output.size(0), -1))
        output = self.cls_fc_pose(input.view(input.size(0), -1))
        output = self.dropout(output)
        output_xy = self.cls_fc_xy(output)
        output_wpqr = self.cls_fc_wpqr(output)
        output_wpqr = F.normalize(output_wpqr, p=2, dim=1)
        return [output_xy, output_wpqr]


###############################################
#              ResNet_PoseNet                 #
###############################################
# Implementation Link : https://towardsdev.com/implement-resnet-with-pytorch-a9fb40a77448


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample, weights=None):
        super().__init__()
        if downsample:
            """
            weight_init_googlenet("inception_" + incp + "/5x5",
                                  nn.Conv2d(x5_reduce_nc, x5_nc, kernel_size=5, padding=2), weights),
            """
            self.conv1 = weight_init_resnet("conv1", nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1), weights)

            self.shortcut = nn.Sequential(
                weight_init_resnet("conv2", nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2),weights),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.conv1 = weight_init_resnet("conv1", nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1), weights)
            self.shortcut = nn.Sequential()

        self.conv2 = weight_init_resnet("conv2", nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1), weights)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, input):
        shortcut = self.shortcut(input)
        input = nn.ReLU()(self.bn1(self.conv1(input)))
        input = nn.ReLU()(self.bn2(self.conv2(input)))
        input = input + shortcut
        return nn.ReLU()(input)


class ResBottleneckBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample, weights=None):
        super().__init__()
        self.downsample = downsample
        self.conv1 = weight_init_resnet("conv1", nn.Conv2d(in_channels, out_channels // 4, kernel_size=1, stride=1), weights)
        self.conv2 = weight_init_resnet("conv2", nn.Conv2d(out_channels // 4, out_channels // 4, kernel_size=3, stride=2 if downsample else 1,
                               padding=1), weights)
        self.conv3 = weight_init_resnet("conv3", nn.Conv2d(out_channels // 4, out_channels, kernel_size=1, stride=1), weights)
        self.shortcut = nn.Sequential()

        if self.downsample or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                weight_init_resnet("shortcut", nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2 if self.downsample else 1), weights),
                nn.BatchNorm2d(out_channels)
            )

        self.bn1 = nn.BatchNorm2d(out_channels // 4)
        self.bn2 = nn.BatchNorm2d(out_channels // 4)
        self.bn3 = nn.BatchNorm2d(out_channels)

    def forward(self, input):
        shortcut = self.shortcut(input)
        input = nn.ReLU()(self.bn1(self.conv1(input)))
        input = nn.ReLU()(self.bn2(self.conv2(input)))
        input = nn.ReLU()(self.bn3(self.conv3(input)))
        input = input + shortcut
        return nn.ReLU()(input)


class ResNet18(nn.Module):
    def __init__(self, in_channels, resblock=ResBlock, weights=None, isKD=False):
        super().__init__()

        self.isKD = isKD

        self.layer0 = nn.Sequential(
            weight_init_resnet("layer0" , nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3), weights=weights),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.layer1 = nn.Sequential(
            ResBlock(64, 64, downsample=False, weights=weights),
            resblock(64, 64, downsample=False, weights=weights)
        )

        self.layer2 = nn.Sequential(
            resblock(64, 128, downsample=True, weights=weights),
            resblock(128, 128, downsample=False, weights=weights)
        )

        self.layer3 = nn.Sequential(
            resblock(128, 256, downsample=True, weights=weights),
            resblock(256, 256, downsample=False, weights=weights)
        )


        self.layer4 = nn.Sequential(
            resblock(256, 512, downsample=True, weights=weights),
            resblock(512, 512, downsample=False, weights=weights)
        )

        self.gap = torch.nn.AdaptiveAvgPool2d(1)
        self.fc = Regression(weights=None, input_cls=512, mid_cls=1024)

    def forward(self, input):
        feature1 = self.layer0(input)
        feature2 = self.layer1(feature1)
        feature3 = self.layer2(feature2)
        feature4 = self.layer3(feature3)
        feature5 = self.layer4(feature4)
        feature6 = self.gap(feature5)
        #input = torch.flatten(input)
        output = self.fc(feature6)

        if self.isKD:
            return output, [feature1] + [feature2] + [feature3] + [feature4] + [feature5]
        else:
            return output



class ResNet34(nn.Module):
    def __init__(self, in_channels, resblock, weights, isKD=False):
        super().__init__()

        self.isKD = isKD

        self.layer0 = nn.Sequential(
            weight_init_resnet("layer0", nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3), weights=weights),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.layer1 = nn.Sequential(
            resblock(64, 64, downsample=False, weights=weights),
            resblock(64, 64, downsample=False, weights=weights),
            resblock(64, 64, downsample=False, weights=weights)
        )

        self.layer2 = nn.Sequential(
            resblock(64, 128, downsample=True, weights=weights),
            resblock(128, 128, downsample=False, weights=weights),
            resblock(128, 128, downsample=False, weights=weights),
            resblock(128, 128, downsample=False, weights=weights)
        )

        self.layer3 = nn.Sequential(
            resblock(128, 256, downsample=True, weights=weights),
            resblock(256, 256, downsample=False, weights=weights),
            resblock(256, 256, downsample=False, weights=weights),
            resblock(256, 256, downsample=False, weights=weights),
            resblock(256, 256, downsample=False, weights=weights),
            resblock(256, 256, downsample=False, weights=weights)
        )


        self.layer4 = nn.Sequential(
            resblock(256, 512, downsample=True, weights=weights),
            resblock(512, 512, downsample=False, weights=weights),
            resblock(512, 512, downsample=False, weights=weights),
        )

        self.gap = torch.nn.AdaptiveAvgPool2d(1)
        self.fc = Regression(weights=None, input_cls=512, mid_cls=1024)

    def forward(self, input):
        feature1 = self.layer0(input)
        feature2 = self.layer1(feature1)
        feature3 = self.layer2(feature2)
        feature4 = self.layer3(feature3)
        feature5 = self.layer4(feature4)
        feature6 = self.gap(feature5)  # torch.Size([75, 512, 1, 1])
        #input = torch.flatten(input)

        output = self.fc(feature6)

        if self.isKD:
            return output, [feature1] + [feature2] + [feature3] + [feature4] + [feature5]
        else:
            return output


class ResNet50(nn.Module):
    def __init__(self, in_channels, resblock, weights, isKD=False):
        super().__init__()

        self.isKD = isKD

        self.layer0 = nn.Sequential(
            weight_init_resnet("layer0", nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3), weights),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        filters = [64, 256, 512, 1024, 2048]
        # [3, 4, 6, 3]

        self.layer1 = nn.Sequential(
            resblock(64, 256, downsample=False, weights=weights),
            resblock(256, 256, downsample=False, weights=weights),
            resblock(256, 256, downsample=False, weights=weights),
        )

        self.layer2 = nn.Sequential(
            resblock(256, 512, downsample=True, weights=weights),
            resblock(512, 512, downsample=False, weights=weights),
            resblock(512, 512, downsample=False, weights=weights),
            resblock(512, 512, downsample=False, weights=weights),
        )

        self.layer3 = nn.Sequential(
            resblock(512, 1024, downsample=True, weights=weights),
            resblock(1024, 1024, downsample=False, weights=weights),
            resblock(1024, 1024, downsample=False, weights=weights),
            resblock(1024, 1024, downsample=False, weights=weights),
            resblock(1024, 1024, downsample=False, weights=weights),
            resblock(1024, 1024, downsample=False, weights=weights),
        )

        self.layer4 = nn.Sequential(
            resblock(1024, 2048, downsample=True, weights=weights),
            resblock(2048, 2048, downsample=False, weights=weights),
            resblock(2048, 2048, downsample=False, weights=weights),
        )

        self.gap = torch.nn.AdaptiveAvgPool2d(1)
        self.fc = Regression(weights=None, input_cls=2048, mid_cls=1024)

    def forward(self, input):
        feature1 = self.layer0(input)
        feature2 = self.layer1(feature1)
        feature3 = self.layer2(feature2)
        feature4 = self.layer3(feature3)
        feature5 = self.layer4(feature4)
        feature6 = self.gap(feature5)
        #print("self.gap(input) :", self.gap(input).shape)
        #input = torch.flatten(input, start_dim=1)
        output = self.fc(feature6)

        if self.isKD:
            return output, [feature1] + [feature2] + [feature3] + [feature4] + [feature5]
        else:
            return output


class ResNet101(nn.Module):
    def __init__(self, in_channels, resblock, weights, isKD):
        super().__init__()

        self.isKD = isKD

        self.layer0 = nn.Sequential(
            weight_init_resnet("layer0", nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3), weights=weights),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        filters = [64, 256, 512, 1024, 2048]
        #[3, 4, 23, 3]

        self.layer1 = nn.Sequential(
            resblock(64, 256, downsample=False, weights=weights),
            resblock(256, 256, downsample=False, weights=weights),
            resblock(256, 256, downsample=False, weights=weights),
        )

        self.layer2 = nn.Sequential(
            resblock(256, 512, downsample=True, weights=weights),
            resblock(512, 512, downsample=False, weights=weights),
            resblock(512, 512, downsample=False, weights=weights),
            resblock(512, 512, downsample=False, weights=weights),
        )

        self.layer3 = nn.Sequential(
            resblock(512, 1024, downsample=True, weights=weights),
        )
        for i in range(22):
            self.layer3.add_module("layer2_%d"%(i+1), resblock(1024, 1024, downsample=False, weights=weights))

        self.layer4 = nn.Sequential(
            resblock(1024, 2048, downsample=True, weights=weights),
            resblock(2048, 2048, downsample=False, weights=weights),
            resblock(2048, 2048, downsample=False, weights=weights),
        )

        self.gap = torch.nn.AdaptiveAvgPool2d(1)
        self.fc = Regression(weights=None, input_cls=2048, mid_cls=1024)

    def forward(self, input):
        feature1 = self.layer0(input)
        feature2 = self.layer1(feature1)
        feature3 = self.layer2(feature2)
        feature4 = self.layer3(feature3)
        feature5 = self.layer4(feature4)
        feature6 = self.gap(feature5)
        #input = torch.flatten(input, start_dim=1)
        output = self.fc(feature6)

        if self.isKD:
            return output, [feature1] + [feature2] + [feature3] + [feature4] + [feature5]
        else:
            return output








