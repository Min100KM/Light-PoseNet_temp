import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from collections import OrderedDict
from torch.autograd import Variable
import util.util as util
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import resPoseNet
import pickle
import numpy



class DistillKL_Feature(nn.Module):
    # Distilling the Knowledge in a ResNet Module (Hint / Guided Module)
    def __init__(self):
        super(DistillKL_Feature, self).__init__()
    def forward(self, feature_T, feature_S):
        feature_S = F.normalize(feature_S, p=2, dim=1)
        feature_T = F.normalize(feature_T, p=2, dim=1)

        feature_S = F.log_softmax(feature_S, dim=1)
        feature_T = F.softmax(feature_T, dim=1)
        loss = F.kl_div(feature_S, feature_T, reduction='batchmean')
        return loss


class Distill_CS(nn.Module):
    def __init__(self, isKL):
        super(Distill_CS, self).__init__()
        self.isKL = isKL
        self.MSE = torch.nn.MSELoss
    def forward(self, CS_T, CS_S):
        if self.isKL:
            CS_S = F.log_softmax(CS_S, dim=1)
            CS_T = F.log_softmax(CS_T, dim=1)

            loss = F.kl_div(CS_S, CS_T, reduction='batchmean')
            return loss
        else:
            loss = self.MSE(CS_S, CS_T)
            return loss



class PoseNetModel(BaseModel):
    def name(self):
        return 'PoseNetModel'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.isTrain = opt.isTrain

        self.isKD = opt.isKD

        # define tensors
        self.input_A = self.Tensor(opt.batchSize, opt.input_nc,
                                   opt.fineSize, opt.fineSize)
        self.input_B = self.Tensor(opt.batchSize, opt.output_nc)

        # load/define networks
        resnet_weights = None
        if self.isTrain and opt.init_weights != '':
            resnet_file = open(opt.init_weights, "rb")
            resnet_weights = pickle.load(resnet_file, encoding="bytes")
            resnet_file.close()
            print('initializing the weights from ' + opt.init_weights)
        self.mean_image = np.load(os.path.join(opt.dataroot, 'mean_image.npy'))


        self.netG = resPoseNet.define_network(opt.input_nc, opt.model,
                                                init_from=resnet_weights, isTest=not self.isTrain,
                                                isKD=self.isKD,
                                                gpu_ids=self.gpu_ids)


        if not self.isTrain or opt.continue_train:
            if not opt.isKD:  # 얘 추가
                self.load_network(self.netG, 'G', opt.which_epoch)

        if self.isTrain:
            self.old_lr = opt.lr
            # define loss functions

            criterion = torch.nn.ModuleList([])

            criterion_MSE = torch.nn.MSELoss()
            criterion_KLfeatures = DistillKL_Feature()
            criterion_CS = Distill_CS()

            if self.isKD:
                criterion.append(criterion_MSE)
                criterion.append(criterion_KLfeatures)
                criterion.append(criterion_CS)

                self.hintmodule = opt.hintmodule
                self.CSmodule = opt.CSmodule

            else:
                criterion.append(criterion_MSE)

            self.criterion = criterion
            self.criterion.cuda()

            # initialize optimizers
            self.schedulers = []
            self.optimizers = []
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                                lr=opt.lr, eps=1,
                                                weight_decay=0.0625,
                                                betas=(self.opt.adambeta1, self.opt.adambeta2))
            self.optimizers.append(self.optimizer_G)
            # for optimizer in self.optimizers:
            #     self.schedulers.append(networks.get_scheduler(optimizer, opt))

        print('---------- Networks initialized -------------')
        # networks.print_network(self.netG)
        # print('-----------------------------------------------')

    def set_input(self, input):
        input_A = input['A']
        input_B = input['B']
        self.image_paths = input['A_paths']

        self.batch_index = input['index_N']

        self.input_A.resize_(input_A.size()).copy_(input_A)
        self.input_B.resize_(input_B.size()).copy_(input_B)

    def forward(self):
        if self.isKD:
            # output, [feature1] + [feature2] + [feature3] + [feature4] + [feature5]
            self.pred_B, self.features = self.netG(self.input_A)

        else:
            self.pred_B = self.netG(self.input_A)

    # no backprop gradients
    def test(self):
        self.forward()

    # get image paths
    def get_image_paths(self):
        return self.image_paths

    def backward(self):
        self.PoseNet_backward()
"""
            if self.isKD:
                criterion.append(criterion_MSE)
                criterion.append(criterion_KLfeatures)
                criterion.append(criterion_CS)

            else:
                criterion.append(criterion_MSE)

            self.criterion = criterion
            self.criterion.cuda()
"""
    def PoseNet_backward(self):
        self.loss_Gt = 0
        self.loss_pos = 0
        self.loss_ori = 0

        if self.isKD:
            self.loss_feature = 0
            self.loss_CS = 0

            criterion_MSE, criterion_KL, criterion_CS = self.criterion()
            # Gt_Loss
            mse_pos = criterion_MSE(self.pred_B[0], self.input_B[:, 0:3])
            ori_gt = F.normalize(self.input_B[:, 3:], p=2, dim=1)
            mse_ori = self.criterion(self.pred_B[1], ori_gt)
            self.loss_Gt = (mse_pos + mse_ori * self.opt.beta)
            self.loss_pos = mse_pos.item()
            self.loss_ori += mse_ori.item() * self.opt.beta

            #Feature_Loss
            """
            def forward(self, g_s, g_t):
                g_t = [g_t[i] for i in self.guide_layers]
                g_s = [g_s[i] for i in self.hint_layers]
                loss = self.attention(g_s, g_t)
                return sum(loss)
                
                self.hintmodule = opt.hintmodule
                self.CSmodule = opt.CSmodule
            """

        else:
            criterion = self.criterion()


        mse_pos = self.criterion(self.pred_B[0], self.input_B[:, 0:3])
        ori_gt = F.normalize(self.input_B[:, 3:], p=2, dim=1)
        mse_ori = self.criterion(self.pred_B[1], ori_gt)
        self.loss_G = (mse_pos + mse_ori * self.opt.beta)
        self.loss_pos = mse_pos.item()
        self.loss_ori += mse_ori.item() * self.opt.beta

        self.loss_G.backward()


    def optimize_parameters(self):
        self.forward()
        self.optimizer_G.zero_grad()
        self.backward()
        self.optimizer_G.step()

    def get_current_errors(self):
        if self.opt.isTrain:
            return OrderedDict([('pos_err', self.loss_pos),
                                ('ori_err', self.loss_ori),
                                ])

        pos_err = torch.dist(self.pred_B[0], self.input_B[:, 0:3])
        ori_gt = F.normalize(self.input_B[:, 3:], p=2, dim=1)
        abs_distance = torch.abs((ori_gt.mul(self.pred_B[1])).sum())
        ori_err = 2 * 180 / numpy.pi * torch.acos(abs_distance)
        return [pos_err.item(), ori_err.item()]

    def get_current_pose(self):
        return numpy.concatenate((self.pred_B[0].data[0].cpu().numpy(),
                                  self.pred_B[1].data[0].cpu().numpy()))

    def get_current_visuals(self):
        input_A = util.tensor2im(self.input_A.data)
        # pred_B = util.tensor2im(self.pred_B.data)
        # input_B = util.tensor2im(self.input_B.data)
        return OrderedDict([('input_A', input_A)])

    def save(self, label):
        self.save_network(self.netG, 'G', label, self.gpu_ids)