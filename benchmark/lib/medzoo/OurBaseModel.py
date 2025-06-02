import torch.nn as nn
import torch
import torch.nn.functional as F
from .attunet.networks_other import init_weights
from torch.nn import init
from collections import OrderedDict


class OurBaseModel(nn.Module):

    def __init__(self, feature_scale=4, n_classes=21, is_deconv=True, in_channels=3,
                 nonlocal_mode='concatenation', attention_dsample=(2, 2, 2), is_batchnorm=True):
        super(OurBaseModel, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale

        filters = [8, 16, 16]

        # downsampling
        self.conv1 = UnetConv31(self.in_channels, filters[0], self.is_batchnorm, kernel_size=(3, 3, 3),
                                padding_size=(1, 1, 1))
        # self.maxpool1 = nn.MaxPool3d(kernel_size=(2, 2, 2))
        self.maxpool1 = Maxpooling1()
        self.conv2 = UnetConv32(filters[0], filters[1], self.is_batchnorm, kernel_size=(3, 3, 3),
                                padding_size=(1, 1, 1))
        # self.maxpool2 = nn.MaxPool3d(kernel_size=(2, 2, 2))
        self.maxpool2 = Maxpooling2()

        self.center = UnetConv33(filters[1], filters[2], self.is_batchnorm, kernel_size=(3, 3, 3),
                                 padding_size=(1, 1, 1))
        self.gating = UnetGridGatingSignal3(filters[2], filters[2], kernel_size=(1, 1, 1),
                                            is_batchnorm=self.is_batchnorm)


        self.up_concat2 = UnetUp3_CT2(filters[2], filters[1], is_batchnorm)
        self.up_concat1 = UnetUp3_CT1(filters[1], filters[0], is_batchnorm)

        self.dsv2 = UnetDsv32(in_size=filters[1], out_size=n_classes, scale_factor=2)
        self.dsv1 = UnetDsv31(in_size=filters[0], out_size=n_classes)

        # final conv (without any concat)
        self.final = Final(in_size=n_classes * 2, out_size=n_classes)


        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm3d):
                init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        # Feature Extraction

        conv1 = self.conv1(inputs)
        maxpool1 = self.maxpool1(conv1)

        conv2 = self.conv2(maxpool1)
        maxpool2 = self.maxpool2(conv2)

        # Gating Signal Generation
        center = self.center(maxpool2)


        g_conv2 = conv2
        g_center = center

        up2 = self.up_concat2(g_conv2, g_center)
        up1 = self.up_concat1(conv1, up2)



        dsv2 = self.dsv2(up2)
        dsv1 = self.dsv1(up1)
        final = self.final(torch.cat([dsv1, dsv2], dim=1))

        return final







class Maxpooling1(nn.Module):
    def __init__(self):
        super(Maxpooling1, self).__init__()
        self.maxpooling = nn.MaxPool3d(kernel_size=(2, 2, 2))

    def forward(self, input1):
        return self.maxpooling(input1)


class Maxpooling2(nn.Module):
    def __init__(self):
        super(Maxpooling2, self).__init__()
        self.maxpooling = nn.MaxPool3d(kernel_size=(2, 2, 2))

    def forward(self, input1):
        return self.maxpooling(input1)


class UnetConv31(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm, kernel_size=(3, 3, 3), padding_size=(1, 1, 1),
                 init_stride=(1, 1, 1)):
        super(UnetConv31, self).__init__()

        if is_batchnorm:
            self.conv1 = nn.Sequential(nn.InstanceNorm3d(in_size),
                                       nn.ReLU(inplace=True),
                                       nn.Conv3d(in_size, out_size, kernel_size=3, stride=1, padding=1),
                                       )

            self.conv2 = nn.Sequential(nn.InstanceNorm3d(out_size),
                                       nn.ReLU(inplace=True),
                                       nn.Conv3d(out_size, out_size, kernel_size=3, stride=1, padding=1),
                                       )

            self.conv3 = nn.Sequential(nn.InstanceNorm3d(out_size),
                                       nn.ReLU(inplace=True),
                                       nn.Conv3d(out_size, out_size, kernel_size=3, stride=1, padding=1),
                                       )
            self.conv4 = nn.Sequential(nn.InstanceNorm3d(out_size),
                                       nn.Conv3d(out_size, out_size, kernel_size=3, stride=1, padding=1),
                                       nn.ReLU(inplace=True),
                                       )
            self.conv5 = nn.Sequential(nn.InstanceNorm3d(out_size),
                                       nn.Conv3d(out_size, out_size, kernel_size=3, stride=1, padding=1),
                                       nn.ReLU(inplace=True),
                                       )
            self.conv6 = nn.Sequential(nn.InstanceNorm3d(out_size),
                                       nn.Conv3d(out_size, out_size, kernel_size=3, stride=1, padding=1),
                                       nn.ReLU(inplace=True),
                                       )

        else:
            self.conv1 = nn.Sequential(nn.ReLU(inplace=True),
                                       nn.Conv3d(in_size, out_size, kernel_size, init_stride, padding_size),
                                       )

            self.conv2 = nn.Sequential(nn.ReLU(inplace=True),
                                       nn.Conv3d(out_size, out_size, kernel_size, 1, padding_size),
                                       )

            self.conv3 = nn.Sequential(nn.ReLU(inplace=True),
                                       nn.Conv3d(out_size, out_size, kernel_size, 1, padding_size),
                                       )
            self.conv4 = nn.Sequential(nn.ReLU(inplace=True),
                                       nn.Conv3d(out_size, out_size, kernel_size=3, stride=1, padding=1),
                                       )
            self.conv5 = nn.Sequential(nn.ReLU(inplace=True),
                                       nn.Conv3d(out_size, out_size, kernel_size=3, stride=1, padding=1),

                                       )
            self.conv6 = nn.Sequential(nn.ReLU(inplace=True),
                                       nn.Conv3d(out_size, out_size, kernel_size=3, stride=1, padding=1),

                                       )

        # initialise the blocks
        for m in self.children():
            init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        output1 = self.conv1(inputs)
        outputs = self.conv2(output1)
        outputs = self.conv3(outputs)
        outputs = self.conv4(outputs)
        outputs = self.conv5(outputs)
        outputs = self.conv6(outputs)
        # residual connection
        # outputs = outputs + output1
        # print('outputs', outputs.shape)

        return outputs


class UnetConv32(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm, kernel_size=(3, 3, 3), padding_size=(1, 1, 1),
                 init_stride=(1, 1, 1)):
        super(UnetConv32, self).__init__()

        if is_batchnorm:
            self.conv1 = nn.Sequential(nn.InstanceNorm3d(in_size),
                                       nn.ReLU(inplace=True),
                                       nn.Conv3d(in_size, out_size, kernel_size=3, stride=1, padding=1),
                                       )

            self.conv2 = nn.Sequential(nn.InstanceNorm3d(out_size),
                                       nn.ReLU(inplace=True),
                                       nn.Conv3d(out_size, out_size, kernel_size=3, stride=1, padding=1),
                                       )

            self.conv3 = nn.Sequential(nn.InstanceNorm3d(out_size),
                                       nn.ReLU(inplace=True),
                                       nn.Conv3d(out_size, out_size, kernel_size=3, stride=1, padding=1),
                                       )
            self.conv4 = nn.Sequential(nn.InstanceNorm3d(out_size),
                                       nn.Conv3d(out_size, out_size, kernel_size=3, stride=1, padding=1),
                                       nn.ReLU(inplace=True),
                                       )
            self.conv5 = nn.Sequential(nn.InstanceNorm3d(out_size),
                                       nn.Conv3d(out_size, out_size, kernel_size=3, stride=1, padding=1),
                                       nn.ReLU(inplace=True),
                                       )
            self.conv6 = nn.Sequential(nn.InstanceNorm3d(out_size),
                                       nn.Conv3d(out_size, out_size, kernel_size=3, stride=1, padding=1),
                                       nn.ReLU(inplace=True),
                                       )

        else:
            self.conv1 = nn.Sequential(nn.ReLU(inplace=True),
                                       nn.Conv3d(in_size, out_size, kernel_size, init_stride, padding_size),
                                       )

            self.conv2 = nn.Sequential(nn.ReLU(inplace=True),
                                       nn.Conv3d(out_size, out_size, kernel_size, 1, padding_size),
                                       )

            self.conv3 = nn.Sequential(nn.ReLU(inplace=True),
                                       nn.Conv3d(out_size, out_size, kernel_size, 1, padding_size),
                                       )
            self.conv4 = nn.Sequential(nn.ReLU(inplace=True),
                                       nn.Conv3d(out_size, out_size, kernel_size=3, stride=1, padding=1),
                                       )
            self.conv5 = nn.Sequential(nn.ReLU(inplace=True),
                                       nn.Conv3d(out_size, out_size, kernel_size=3, stride=1, padding=1),

                                       )
            self.conv6 = nn.Sequential(nn.ReLU(inplace=True),
                                       nn.Conv3d(out_size, out_size, kernel_size=3, stride=1, padding=1),

                                       )

        # initialise the blocks
        for m in self.children():
            init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        output1 = self.conv1(inputs)
        outputs = self.conv2(output1)
        outputs = self.conv3(outputs)
        outputs = self.conv4(outputs)
        outputs = self.conv5(outputs)
        outputs = self.conv6(outputs)
        # residual connection
        # outputs = outputs + output1


        return outputs


class UnetConv33(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm, kernel_size=(3, 3, 3), padding_size=(1, 1, 1),
                 init_stride=(1, 1, 1)):
        super(UnetConv33, self).__init__()

        if is_batchnorm:
            self.conv1 = nn.Sequential(nn.InstanceNorm3d(in_size),
                                       nn.ReLU(inplace=True),
                                       nn.Conv3d(in_size, out_size, kernel_size=3, stride=1, padding=1),
                                       )

            self.conv2 = nn.Sequential(nn.InstanceNorm3d(out_size),
                                       nn.ReLU(inplace=True),
                                       nn.Conv3d(out_size, out_size, kernel_size=3, stride=1, padding=1),
                                       )

            self.conv3 = nn.Sequential(nn.InstanceNorm3d(out_size),
                                       nn.ReLU(inplace=True),
                                       nn.Conv3d(out_size, out_size, kernel_size=3, stride=1, padding=1),
                                       )
            self.conv4 = nn.Sequential(nn.InstanceNorm3d(out_size),
                                       nn.Conv3d(out_size, out_size, kernel_size=3, stride=1, padding=1),
                                       nn.ReLU(inplace=True),
                                       )
            self.conv5 = nn.Sequential(nn.InstanceNorm3d(out_size),
                                       nn.Conv3d(out_size, out_size, kernel_size=3, stride=1, padding=1),
                                       nn.ReLU(inplace=True),
                                       )
            self.conv6 = nn.Sequential(nn.InstanceNorm3d(out_size),
                                       nn.Conv3d(out_size, out_size, kernel_size=3, stride=1, padding=1),
                                       nn.ReLU(inplace=True),
                                       )

        else:
            self.conv1 = nn.Sequential(nn.ReLU(inplace=True),
                                       nn.Conv3d(in_size, out_size, kernel_size, init_stride, padding_size),
                                       )

            self.conv2 = nn.Sequential(nn.ReLU(inplace=True),
                                       nn.Conv3d(out_size, out_size, kernel_size, 1, padding_size),
                                       )

            self.conv3 = nn.Sequential(nn.ReLU(inplace=True),
                                       nn.Conv3d(out_size, out_size, kernel_size, 1, padding_size),
                                       )
            self.conv4 = nn.Sequential(nn.ReLU(inplace=True),
                                       nn.Conv3d(out_size, out_size, kernel_size=3, stride=1, padding=1),
                                       )
            self.conv5 = nn.Sequential(nn.ReLU(inplace=True),
                                       nn.Conv3d(out_size, out_size, kernel_size=3, stride=1, padding=1),

                                       )
            self.conv6 = nn.Sequential(nn.ReLU(inplace=True),
                                       nn.Conv3d(out_size, out_size, kernel_size=3, stride=1, padding=1),

                                       )

        # initialise the blocks
        for m in self.children():
            init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        output1 = self.conv1(inputs)
        outputs = self.conv2(output1)
        outputs = self.conv3(outputs)
        outputs = self.conv4(outputs)
        outputs = self.conv5(outputs)
        outputs = self.conv6(outputs)
        # residual connection
        # outputs = outputs + output1

        return outputs


class UnetUp3_CT1(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm=True):
        super(UnetUp3_CT1, self).__init__()
        self.conv = nn.Conv3d(in_size + out_size, out_size, kernel_size=3, padding=1)
        self.up = nn.Upsample(scale_factor=(2, 2, 2), mode='trilinear')
        self.norm1 = nn.InstanceNorm3d(in_size)
        self.norm2 = nn.InstanceNorm3d(out_size)
        self.relu = nn.LeakyReLU()

        # initialise the blocks
        for m in self.children():
            if m.__class__.__name__.find('UnetConv3') != -1: continue
            init_weights(m, init_type='kaiming')

    def forward(self, inputs1, inputs2):
        inputs2 = self.norm1(inputs2)
        outputs2 = self.up(inputs2)
        # print('input2',inputs2.shape)
        # print('input1',inputs1.shape)
        # print('outputs2', outputs2.shape)
        # exit()
        outputs = self.conv(torch.cat([inputs1, outputs2], 1))
        outputs = self.norm2(outputs)
        outputs = self.relu(outputs)

        return outputs


class UnetUp3_CT2(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm=True):
        super(UnetUp3_CT2, self).__init__()
        self.conv = nn.Conv3d(in_size + out_size, out_size, kernel_size=3, padding=1)
        self.up = nn.Upsample(scale_factor=(2, 2, 2), mode='trilinear')
        self.norm1 = nn.InstanceNorm3d(in_size)
        self.norm2 = nn.InstanceNorm3d(out_size)
        self.relu = nn.LeakyReLU()

        # initialise the blocks
        for m in self.children():
            if m.__class__.__name__.find('UnetConv3') != -1: continue
            init_weights(m, init_type='kaiming')

    def forward(self, inputs1, inputs2):
        inputs2 = self.norm1(inputs2)
        outputs2 = self.up(inputs2)
        # print('input2',inputs2.shape)
        # print('input1',inputs1.shape)
        # print('outputs2', outputs2.shape)
        # exit()
        outputs = self.conv(torch.cat([inputs1, outputs2], 1))
        outputs = self.norm2(outputs)
        outputs = self.relu(outputs)

        return outputs


class UnetDsv32(nn.Module):
    def __init__(self, in_size, out_size, scale_factor):
        super(UnetDsv32, self).__init__()
        self.dsv = nn.Sequential(nn.Conv3d(in_size, out_size, kernel_size=1, stride=1, padding=0),
                                 nn.Upsample(scale_factor=scale_factor, mode='trilinear'), )

    def forward(self, input):
        return self.dsv(input)


class UnetDsv31(nn.Module):
    def __init__(self, in_size, out_size):
        super(UnetDsv31, self).__init__()
        self.dsv = nn.Sequential(nn.Conv3d(in_size, out_size, kernel_size=1, stride=1, padding=0),
                                 )

    def forward(self, input):
        return self.dsv(input)


class Final(nn.Module):
    def __init__(self, in_size, out_size):
        super(Final, self).__init__()
        self.dsv = nn.Sequential(nn.Conv3d(in_size, out_size, kernel_size=1, stride=1, padding=0),
                                 )

    def forward(self, input):
        return self.dsv(input)


class UnetGridGatingSignal3(nn.Module):
    def __init__(self, in_size, out_size, kernel_size=(1, 1, 1), is_batchnorm=True):
        super(UnetGridGatingSignal3, self).__init__()

        if is_batchnorm:
            self.conv1 = nn.Sequential(nn.Conv3d(in_size, out_size, kernel_size, (1, 1, 1), (0, 0, 0)),
                                       nn.BatchNorm3d(out_size),
                                       nn.ReLU(inplace=True),
                                       )
        else:
            self.conv1 = nn.Sequential(nn.Conv3d(in_size, out_size, kernel_size, (1, 1, 1), (0, 0, 0)),
                                       nn.ReLU(inplace=True),
                                       )

        # initialise the blocks
        for m in self.children():
            init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        return outputs

