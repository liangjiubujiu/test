import os
import torch
import torch.nn as nn
from .Unet3DMINI import UNet3Dmini,Discriminator
from lib.medzoo.base_model import PixBaseModel
from ..losses3D.ganloss3d import GANLoss,get_scheduler
from ..losses3D import DiceLoss
def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)

class CBCTpix(PixBaseModel):
    def __init__(self,in_channels,num_classes,base_n_filter):
        super(CBCTpix, self).__init__()
        self.netG = UNet3Dmini(in_channels=in_channels, n_classes=num_classes, base_n_filter=base_n_filter)


        self.netD = Discriminator(in_channels=in_channels*2,n_classes=2,base_n_filter=2)

        # define loss functions
        self.criterionGAN = GANLoss(tensor=torch.cuda.FloatTensor)
        self.criterionL1 = torch.nn.L1Loss()
        self.criterionDICE = DiceLoss(classes=2)
        # initialize optimizers
        self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=0.002, betas=(0.9, 0.999))
        self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=0.008, betas=(0.9, 0.999))

        self.optimizers = []
        self.schedulers = []

        self.optimizers.append(self.optimizer_G)
        self.optimizers.append(self.optimizer_D)

        for optimizer in self.optimizers:
            self.schedulers.append(get_scheduler(optimizer))

        print('---------- Networks initialized -------------')
        print_network(self.netG)


        print_network(self.netD)

        print('-----------------------------------------------')

    def set_input(self, input,target):


        self.input_img = input
        self.input_mask = target



    def forward(self):
        # self.pred_mask =nn.parallel.data_parallel(self.netG, self.input_img, [0,1])
        # self.netG.cuda()
        self.pred_mask=self.netG(self.input_img)
        # self.pred_mask = self.netG(self.input_img)
        return self.pred_mask

    def backward_D(self):
        fake_AB = torch.cat((self.input_img, self.pred_mask[:,1,:,:,:].unsqueeze(dim=1)), 1)
        # print('fake AB SHAPE',fake_AB.shape)
        pred_fake = self.netD(fake_AB.detach())

        self.loss_D_fake, per_ch_score_fake = self.criterionGAN(pred_fake, False)

        real_AB = torch.cat((self.input_img, self.input_mask.unsqueeze(dim=1)), 1)
        pred_real = self.netD(real_AB)
        # print('real AB SHAPE', pred_real.shape)
        self.loss_D_real,per_ch_score_real = self.criterionGAN(pred_real, True)

        # combine loss and calculate gradients

        self.loss_D = (self.loss_D_fake + self.loss_D_real)*0.5
        print('loss D',self.loss_D)
        print('D_total_loss: %f D_REAL: %f D_FAKE: %f' % (self.loss_D,self.loss_D_real, self.loss_D_fake))
        self.loss_D.backward()

    def backward_G(self):
        fake_AB = torch.cat((self.input_img, self.pred_mask[:,1,:,:,:].unsqueeze(dim=1)), 1)
        pred_fake = self.netD(fake_AB)
        self.loss_G_GAN ,per_ch_score= self.criterionGAN(pred_fake, True)

        # Second, G(A) = B
        self.loss_G_L1 = self.criterionL1(self.pred_mask[:,1,:,:,:].unsqueeze(dim=1), self.input_mask.unsqueeze(dim=1))



        self.loss_dice, per_ch_score = self.criterionDICE(self.pred_mask,self.input_mask)

        # combine loss and calculate gradients
        # self.loss_G_GAN=0
        # self.loss_G_L1=0
        self.loss_G = self.loss_G_GAN + self.loss_G_L1*70+self.loss_dice*100
        print('G_total_loss: %f G_GAN: %f G_L1: %f DICE: %f' % (self.loss_G, self.loss_G_GAN, self.loss_G_L1, self.loss_dice))
        #0.6,0.03,0.5
        self.loss_G.backward()

        return self.loss_dice, per_ch_score

    def optimize_parameters(self):
        self.forward()  # compute fake images: G(A)
        # update D
        self.set_requires_grad(self.netD, True)  # enable backprop for D
        self.optimizer_D.zero_grad()  # set D's gradients to zero
        self.backward_D()  # calculate gradients for D
        self.optimizer_D.step()  # update D's weights
        # update G
        self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
        self.optimizer_G.zero_grad()  # set G's gradients to zero
        loss_dice, per_ch_score=self.backward_G()  # calculate graidents for G
        self.optimizer_G.step()  # udpate G's weights
        torch.cuda.empty_cache()
        return loss_dice, per_ch_score


    def count_params(self):
        r"""
        Computes the number of parameters in this model.

        Args: None

        Returns:
            int: Total number of weight parameters for this model.
            int: Total number of trainable parameters for this model.

        """
        num_total_params = sum(p.numel() for p in self.parameters())
        num_trainable_params = sum(p.numel() for p in self.parameters()
                                   if p.requires_grad)

        return num_total_params, num_trainable_params
