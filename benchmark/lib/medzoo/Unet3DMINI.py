# import torch.nn as nn
# import torch
# from torchsummary import summary
# import torchsummaryX
# from lib.medzoo.BaseModelClass import BaseModel
#
#
# class UNet3Dmini(BaseModel):
#     """
#     Implementations based on the Unet3D paper: https://arxiv.org/abs/1606.06650
#     """
#
#     def __init__(self, in_channels, n_classes, base_n_filter=8):
#         super(UNet3Dmini, self).__init__()
#         self.in_channels = in_channels
#         self.n_classes = n_classes
#         self.base_n_filter = base_n_filter
#
#         self.lrelu = nn.LeakyReLU()
#         self.dropout3d = nn.Dropout3d(p=0.6)
#         self.upsacle = nn.Upsample(scale_factor=2, mode='nearest')
#         self.softmax = nn.Softmax(dim=1)
#
#         self.conv3d_c1_1 = nn.Conv3d(self.in_channels, self.base_n_filter, kernel_size=3, stride=1, padding=1,
#                                      bias=False)
#         self.conv3d_c1_2 = nn.Conv3d(self.base_n_filter, self.base_n_filter, kernel_size=3, stride=1, padding=1,
#                                      bias=False)
#         self.lrelu_conv_c1 = self.lrelu_conv(self.base_n_filter, self.base_n_filter)
#         self.inorm3d_c1 = nn.InstanceNorm3d(self.base_n_filter)
#
#         self.conv3d_c2 = nn.Conv3d(self.base_n_filter, self.base_n_filter * 2, kernel_size=3, stride=2, padding=1,
#                                    bias=False)
#         self.norm_lrelu_conv_c2 = self.norm_lrelu_conv(self.base_n_filter * 2, self.base_n_filter * 2)
#         self.inorm3d_c2 = nn.InstanceNorm3d(self.base_n_filter * 2)
#
#         self.conv3d_c5 = nn.Conv3d(self.base_n_filter * 2, self.base_n_filter * 4, kernel_size=3, stride=2, padding=1,
#                                    bias=False)
#         self.norm_lrelu_conv_c5 = self.norm_lrelu_conv(self.base_n_filter * 4, self.base_n_filter * 4)
#         self.norm_lrelu_upscale_conv_norm_lrelu_l0 = self.norm_lrelu_upscale_conv_norm_lrelu(self.base_n_filter * 4,
#                                                                                              self.base_n_filter * 2)
#
#         self.conv3d_l0 = nn.Conv3d(self.base_n_filter * 2, self.base_n_filter * 2, kernel_size=1, stride=1, padding=0,
#                                    bias=False)
#         self.inorm3d_l0 = nn.InstanceNorm3d(self.base_n_filter * 2)
#
#         self.conv_norm_lrelu_l3 = self.conv_norm_lrelu(self.base_n_filter * 4, self.base_n_filter * 4)
#         self.conv3d_l3 = nn.Conv3d(self.base_n_filter * 4, self.base_n_filter * 2, kernel_size=1, stride=1, padding=0,
#                                    bias=False)
#         self.norm_lrelu_upscale_conv_norm_lrelu_l3 = self.norm_lrelu_upscale_conv_norm_lrelu(self.base_n_filter * 2,
#                                                                                              self.base_n_filter)
#
#         self.conv_norm_lrelu_l4 = self.conv_norm_lrelu(self.base_n_filter * 2, self.base_n_filter * 2)
#         self.conv3d_l4 = nn.Conv3d(self.base_n_filter * 2, self.n_classes, kernel_size=1, stride=1, padding=0,
#                                    bias=False)
#
#         self.ds2_1x1_conv3d = nn.Conv3d(self.base_n_filter * 8, self.n_classes, kernel_size=1, stride=1, padding=0,
#                                         bias=False)
#         self.ds3_1x1_conv3d = nn.Conv3d(self.base_n_filter * 4, self.n_classes, kernel_size=1, stride=1, padding=0,
#                                         bias=False)
#         self.sigmoid = nn.Sigmoid()
#
#     def conv_norm_lrelu(self, feat_in, feat_out):
#         return nn.Sequential(
#             nn.Conv3d(feat_in, feat_out, kernel_size=3, stride=1, padding=1, bias=False),
#             nn.InstanceNorm3d(feat_out),
#             nn.LeakyReLU())
#
#     def norm_lrelu_conv(self, feat_in, feat_out):
#         return nn.Sequential(
#             nn.InstanceNorm3d(feat_in),
#             nn.LeakyReLU(),
#             nn.Conv3d(feat_in, feat_out, kernel_size=3, stride=1, padding=1, bias=False))
#
#     def lrelu_conv(self, feat_in, feat_out):
#         return nn.Sequential(
#             nn.LeakyReLU(),
#             nn.Conv3d(feat_in, feat_out, kernel_size=3, stride=1, padding=1, bias=False))
#
#     def norm_lrelu_upscale_conv_norm_lrelu(self, feat_in, feat_out):
#         return nn.Sequential(
#             nn.InstanceNorm3d(feat_in),
#             nn.LeakyReLU(),
#             nn.Upsample(scale_factor=2, mode='nearest'),
#             # should be feat_in*2 or feat_in
#             nn.Conv3d(feat_in, feat_out, kernel_size=3, stride=1, padding=1, bias=False),
#             nn.InstanceNorm3d(feat_out),
#             nn.LeakyReLU())
#
#     def forward(self, x):
#         #  Level 1 context pathway
#         out = self.conv3d_c1_1(x)
#         residual_1 = out
#         out = self.lrelu(out)
#         out = self.conv3d_c1_2(out)
#         out = self.dropout3d(out)
#         out = self.lrelu_conv_c1(out)
#         # Element Wise Summation
#         out += residual_1
#         context_1 = self.lrelu(out)
#         out = self.inorm3d_c1(out)
#         out = self.lrelu(out)
#         # print("content1-l4")
#         # print(context_1.shape)
#
#         # Level 2 context pathway
#         out = self.conv3d_c2(out)
#         residual_2 = out
#         out = self.norm_lrelu_conv_c2(out)
#         out = self.dropout3d(out)
#         out = self.norm_lrelu_conv_c2(out)
#         out += residual_2
#         out = self.inorm3d_c2(out)
#         out = self.lrelu(out)
#         context_2 = out
#         # print("content2-l3")
#         # print(context_2.shape)
#
#         # Level 5
#         out = self.conv3d_c5(out)
#         residual_5 = out
#         out = self.norm_lrelu_conv_c5(out)
#         out = self.dropout3d(out)
#         out = self.norm_lrelu_conv_c5(out)
#         out += residual_5
#         # print("content5-l0")
#         # print(out.shape)
#         out = self.norm_lrelu_upscale_conv_norm_lrelu_l0(out)
#         out = self.conv3d_l0(out)
#         out = self.inorm3d_l0(out)
#         out = self.lrelu(out)
#         # print(out.shape)
#
#
#         # Level 3 localization pathway
#
#         out = torch.cat([out, context_2], dim=1)
#         # print(out.shape)
#         out = self.conv_norm_lrelu_l3(out)
#         # print(out.shape)
#         ds3 = out
#         out = self.conv3d_l3(out)
#         # print(out.shape)
#         out = self.norm_lrelu_upscale_conv_norm_lrelu_l3(out)
#         # print(out.shape)
#
#         # Level 4 localization pathway
#
#         out = torch.cat([out, context_1], dim=1)
#         # print(out.shape)
#         out = self.conv_norm_lrelu_l4(out)
#         # print(out.shape)
#         out_pred = self.conv3d_l4(out)
#         # print(out.shape)
#
#
#         ds3_1x1_conv = self.ds3_1x1_conv3d(ds3)
#         ds1_ds2_sum_upscale_ds3_sum = ds3_1x1_conv
#         ds1_ds2_sum_upscale_ds3_sum_upscale = self.upsacle(ds1_ds2_sum_upscale_ds3_sum)
#         # print("combination")
#         #
#         # print( ds3_1x1_conv.shape)
#         # print( ds1_ds2_sum_upscale_ds3_sum.shape)
#         # print(ds1_ds2_sum_upscale_ds3_sum_upscale.shape)
#
#         out = out_pred + ds1_ds2_sum_upscale_ds3_sum_upscale
#         # print(out.shape)
#         seg_layer = out
#         return seg_layer
#
#     def test(self,device='cpu'):
#
#         input_tensor = torch.rand(1, 2, 32, 32, 32)
#         ideal_out = torch.rand(1, self.n_classes, 32, 32, 32)
#         out = self.forward(input_tensor)
#         assert ideal_out.shape == out.shape
#         summary(self.to(torch.device(device)), (2, 32, 32, 32),device='cpu')
#         # import torchsummaryX
#         # torchsummaryX.summary(self, input_tensor.to(device))
#         print("Unet3D test is complete")
#
#


import torch.nn as nn
import torch
from torchsummary import summary
import torchsummaryX
from lib.medzoo.BaseModelClass import BaseModel


import torch.nn as nn
class UNet3Dmini(BaseModel):
    """
    Implementations based on the Unet3D paper: https://arxiv.org/abs/1606.06650
    """

    def __init__(self, in_channels, n_classes, base_n_filter=8):
        super(UNet3Dmini, self).__init__()
        self.in_channels = in_channels
        self.n_classes = n_classes
        self.base_n_filter = base_n_filter

        self.lrelu = nn.LeakyReLU()
        self.dropout3d = nn.Dropout3d(p=0.6)
        self.upsacle = nn.Upsample(scale_factor=2, mode='nearest')
        self.softmax = nn.Softmax(dim=1)

        self.conv3d_c1_1 = nn.Conv3d(self.in_channels, self.base_n_filter, kernel_size=3, stride=1, padding=1,
                                     bias=False)
        self.conv3d_c1_2 = nn.Conv3d(self.base_n_filter, self.base_n_filter, kernel_size=3, stride=1, padding=1,
                                     bias=False)
        self.lrelu_conv_c1 = self.lrelu_conv(self.base_n_filter, self.base_n_filter)
        self.inorm3d_c1 = nn.InstanceNorm3d(self.base_n_filter)

        self.conv3d_c2 = nn.Conv3d(self.base_n_filter, self.base_n_filter * 2, kernel_size=3, stride=2, padding=1,
                                   bias=False)
        self.norm_lrelu_conv_c2 = self.norm_lrelu_conv(self.base_n_filter * 2, self.base_n_filter * 2)
        self.inorm3d_c2 = nn.InstanceNorm3d(self.base_n_filter * 2)

        self.conv3d_c5 = nn.Conv3d(self.base_n_filter * 2, self.base_n_filter * 4, kernel_size=3, stride=2, padding=1,
                                   bias=False)
        self.norm_lrelu_conv_c5 = self.norm_lrelu_conv(self.base_n_filter * 4, self.base_n_filter * 4)
        self.norm_lrelu_upscale_conv_norm_lrelu_l0 = self.norm_lrelu_upscale_conv_norm_lrelu(self.base_n_filter * 4,
                                                                                             self.base_n_filter * 2)

        self.conv3d_l0 = nn.Conv3d(self.base_n_filter * 2, self.base_n_filter * 2, kernel_size=1, stride=1, padding=0,
                                   bias=False)
        self.inorm3d_l0 = nn.InstanceNorm3d(self.base_n_filter * 2)

        self.conv_norm_lrelu_l3 = self.conv_norm_lrelu(self.base_n_filter * 4, self.base_n_filter * 4)
        self.conv3d_l3 = nn.Conv3d(self.base_n_filter * 4, self.base_n_filter * 2, kernel_size=1, stride=1, padding=0,
                                   bias=False)
        self.norm_lrelu_upscale_conv_norm_lrelu_l3 = self.norm_lrelu_upscale_conv_norm_lrelu(self.base_n_filter * 2,
                                                                                             self.base_n_filter)

        self.conv_norm_lrelu_l4 = self.conv_norm_lrelu(self.base_n_filter * 2, self.base_n_filter * 2)
        self.conv3d_l4 = nn.Conv3d(self.base_n_filter * 2, self.n_classes, kernel_size=1, stride=1, padding=0,
                                   bias=False)

        self.ds2_1x1_conv3d = nn.Conv3d(self.base_n_filter * 8, self.n_classes, kernel_size=1, stride=1, padding=0,
                                        bias=False)
        self.ds3_1x1_conv3d = nn.Conv3d(self.base_n_filter * 4, self.n_classes, kernel_size=1, stride=1, padding=0,
                                        bias=False)


    def conv_norm_lrelu(self, feat_in, feat_out):
        return nn.Sequential(
            nn.Conv3d(feat_in, feat_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm3d(feat_out),
            nn.LeakyReLU())

    def norm_lrelu_conv(self, feat_in, feat_out):
        return nn.Sequential(
            nn.InstanceNorm3d(feat_in),
            nn.LeakyReLU(),
            nn.Conv3d(feat_in, feat_out, kernel_size=3, stride=1, padding=1, bias=False))

    def lrelu_conv(self, feat_in, feat_out):
        return nn.Sequential(
            nn.LeakyReLU(),
            nn.Conv3d(feat_in, feat_out, kernel_size=3, stride=1, padding=1, bias=False))

    def norm_lrelu_upscale_conv_norm_lrelu(self, feat_in, feat_out):
        return nn.Sequential(
            nn.InstanceNorm3d(feat_in),
            nn.LeakyReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            # should be feat_in*2 or feat_in
            nn.Conv3d(feat_in, feat_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm3d(feat_out),
            nn.LeakyReLU())

    def forward(self, x):
        #  Level 1 context pathway
        out = self.conv3d_c1_1(x)
        residual_1 = out
        out = self.lrelu(out)
        out = self.conv3d_c1_2(out)
        out = self.dropout3d(out)
        out = self.lrelu_conv_c1(out)
        # Element Wise Summation
        out += residual_1
        context_1 = self.lrelu(out)
        out = self.inorm3d_c1(out)
        out = self.lrelu(out)
        # print("content1-l4")
        # print(context_1.shape)

        # Level 2 context pathway
        out = self.conv3d_c2(out)
        residual_2 = out
        out = self.norm_lrelu_conv_c2(out)
        out = self.dropout3d(out)
        out = self.norm_lrelu_conv_c2(out)
        out += residual_2
        out = self.inorm3d_c2(out)
        out = self.lrelu(out)
        context_2 = out
        # print("content2-l3")
        # print(context_2.shape)

        # Level 5
        out = self.conv3d_c5(out)
        residual_5 = out
        out = self.norm_lrelu_conv_c5(out)
        out = self.dropout3d(out)
        out = self.norm_lrelu_conv_c5(out)
        out += residual_5
        # print("content5-l0")
        # print(out.shape)
        out = self.norm_lrelu_upscale_conv_norm_lrelu_l0(out)
        out = self.conv3d_l0(out)
        out = self.inorm3d_l0(out)
        out = self.lrelu(out)
        # print(out.shape)


        # Level 3 localization pathway

        out = torch.cat([out, context_2], dim=1)
        # print(out.shape)
        out = self.conv_norm_lrelu_l3(out)
        # print(out.shape)
        ds3 = out
        out = self.conv3d_l3(out)
        # print(out.shape)
        out = self.norm_lrelu_upscale_conv_norm_lrelu_l3(out)
        # print(out.shape)

        # Level 4 localization pathway

        out = torch.cat([out, context_1], dim=1)
        # print(out.shape)
        out = self.conv_norm_lrelu_l4(out)
        # print(out.shape)
        out_pred = self.conv3d_l4(out)
        # print(out.shape)


        ds3_1x1_conv = self.ds3_1x1_conv3d(ds3)
        ds1_ds2_sum_upscale_ds3_sum = ds3_1x1_conv
        ds1_ds2_sum_upscale_ds3_sum_upscale = self.upsacle(ds1_ds2_sum_upscale_ds3_sum)
        # print("combination")
        #
        # print( ds3_1x1_conv.shape)
        # print( ds1_ds2_sum_upscale_ds3_sum.shape)
        # print(ds1_ds2_sum_upscale_ds3_sum_upscale.shape)

        out = out_pred + ds1_ds2_sum_upscale_ds3_sum_upscale
        # print(out.shape)
        seg_layer = out
        return seg_layer

    def test(self,device='cpu'):

        input_tensor = torch.rand(1, 2, 32, 32, 32)
        ideal_out = torch.rand(1, self.n_classes, 32, 32, 32)
        out = self.forward(input_tensor)
        assert ideal_out.shape == out.shape
        summary(self.to(torch.device(device)), (2, 32, 32, 32),device='cpu')
        # import torchsummaryX
        # torchsummaryX.summary(self, input_tensor.to(device))
        print("Unet3D test is complete")

class Discriminator(BaseModel):
    """
    Implementations based on the Unet3D paper: https://arxiv.org/abs/1606.06650
    """

    def __init__(self, in_channels, n_classes, base_n_filter=2):
        super(Discriminator, self).__init__()
        self.in_channels = in_channels
        self.n_classes = n_classes
        self.base_n_filter = base_n_filter

        self.lrelu = nn.LeakyReLU()


        self.conv3d_c1_1 = nn.Conv3d(self.in_channels, self.base_n_filter, kernel_size=1, stride=1, padding=1,
                                     bias=False)


        self.inorm3d_c1 = nn.InstanceNorm3d(self.base_n_filter)

        self.conv3d_l4 = nn.Conv3d(self.base_n_filter, 1, kernel_size=1, stride=1, padding=0,
                                   bias=False)



    def forward(self, x):
        #  Level 1 context pathway
        out = self.conv3d_c1_1(x)
        out = self.inorm3d_c1(out)
        out_pred = self.conv3d_l4(out)

        return out_pred

    def test(self,device='cpu'):

        pass




