import torch.nn as nn
import torch
import torch.nn as nn
import torch
from torchsummary import summary
import torchsummaryX
from lib.medzoo.BaseModelClass import BaseModel

class SELayer(nn.Module):
    def __init__(self, channel, reduction=2):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            # nn.ReLU(inplace=True),
            nn.LeakyReLU(),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1, 1)
        return x * y.expand_as(x)

class Seunetmini(BaseModel):
    """
    Implementations based on the Unet3D paper: https://arxiv.org/abs/1606.06650
    """

    def __init__(self, in_channels, n_classes, base_n_filter=8):
        super(Seunetmini, self).__init__()
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
        self.ds4_1x1_conv3d = nn.Conv3d(self.base_n_filter * 2, self.n_classes, kernel_size=1, stride=1, padding=0,
                                        bias=False)
        self.sigmoid = nn.Sigmoid()

        self.se=SELayer(n_classes*2,reduction=self.n_classes)

        self.output = nn.Conv3d(n_classes*2, self.n_classes, kernel_size=1, stride=1, padding=0,bias=False)

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
        ds4=out
        # print(out.shape)
        out_pred = self.conv3d_l4(out)
        # print(out.shape)


        ds3_1x1_conv = self.ds3_1x1_conv3d(ds3)
        # print('ds3_1x1',ds3_1x1_conv.shape)
        ds1_ds2_sum_upscale_ds3_sum_upscale = self.upsacle(ds3_1x1_conv)
        # print('ds1_ds2_sum_upscale_ds3_sum_upscale',ds1_ds2_sum_upscale_ds3_sum_upscale.shape)

        ds4_1x1=self.ds4_1x1_conv3d(ds4)
        # print('ds4_1x1', ds4_1x1.shape)
        out = out_pred + ds1_ds2_sum_upscale_ds3_sum_upscale+ds4_1x1

        seg_layer=out
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





# class SEBlock(nn.Module):
#     def __init__(self, inplanes, planes, kernel_size=3, stride=2, padding=1, bias=False):
#         super(SEBlock, self).__init__()
#         self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
#
#         self.norm = nn.InstanceNorm3d(planes)
#         # self.relu = nn.ReLU(inplace=True)
#         self.relu = nn.LeakyReLU()
#         self.dropout3d = nn.Dropout3d(p=0.6)
#         self.conv2 = nn.Conv3d(planes, planes, kernel_size=kernel_size, stride=1, padding=padding, bias=bias)
#
#         self.se = SELayer(planes)
#         self.downsample = nn.Sequential(
#             nn.Conv3d(inplanes, planes, kernel_size=1, stride=stride, padding=0, bias=bias),
#             nn.InstanceNorm3d(planes))
#
#         self.stride = stride
#
#     def forward(self, x):
#         residual = x
#
#         out = self.conv1(x)
#         out = self.norm(out)
#         out = self.relu(out)
#
#         out = self.dropout3d(out)
#
#         out = self.conv2(out)
#         out = self.norm(out)
#         out = self.relu(out)
#
#
#         out = self.se(out)
#
#         residual = self.downsample(residual)
#         # print('residual shape',residual.shape)
#         out += residual
#
#         out = self.norm(out)
#         out = self.relu(out)
#
#         return out
#
#
# class BasicBlock(nn.Module):
#     def __init__(self, inplanes, planes, kernel_size=3, stride=2, padding=1, bias=False):
#         super(BasicBlock, self).__init__()
#         self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
#
#         self.norm = nn.InstanceNorm3d(planes)
#         # self.relu = nn.ReLU(inplace=True)
#         self.relu = nn.LeakyReLU()
#         self.dropout3d = nn.Dropout3d(p=0.6)
#         self.conv2 = nn.Conv3d(planes, planes, kernel_size=kernel_size, stride=1, padding=padding, bias=bias)
#
#         self.se = SELayer(planes)
#         self.downsample = nn.Sequential(
#             nn.Conv3d(inplanes, planes, kernel_size=1, stride=stride, padding=0, bias=bias),
#             nn.InstanceNorm3d(planes))
#
#         self.stride = stride
#
#     def forward(self, x):
#         residual = x
#
#         out = self.conv1(x)
#         out = self.norm(out)
#         out = self.relu(out)
#
#         out = self.dropout3d(out)
#
#         out = self.conv2(out)
#         out = self.norm(out)
#         out = self.relu(out)
#
#
#
#
#         residual = self.downsample(residual)
#         # print('residual shape',residual.shape)
#         out += residual
#
#         out = self.norm(out)
#         out = self.relu(out)
#
#         return out
#
#
# class Seunetmini(nn.Module):
#     def __init__(self, in_channels, classes, base_n_filter=8):
#         super(Seunetmini, self).__init__()
#         self.downblock0 = SEBlock(in_channels, base_n_filter,stride=1)
#         self.downblock1 = SEBlock(base_n_filter, base_n_filter)
#         self.downblock2 = SEBlock(base_n_filter, base_n_filter * 2)
#         self.downblock3 = SEBlock(base_n_filter * 2, base_n_filter * 4)
#
#         self.midblock0 = BasicBlock(base_n_filter * 4, base_n_filter * 4,stride=1)
#
#         self.midblock1 = BasicBlock(base_n_filter * 8, base_n_filter * 4,stride=1)
#         self.upblock1 = self.upblock(base_n_filter * 4, base_n_filter * 2)
#
#         self.midblock2 = BasicBlock(base_n_filter * 4, base_n_filter * 2,stride=1)
#         self.upblock2 = self.upblock(base_n_filter * 2, base_n_filter)
#
#
#         self.midblock3 = BasicBlock(base_n_filter * 2, base_n_filter,stride=1)
#         self.upblock3 = self.upblock(base_n_filter, base_n_filter)
#
#         self.output = nn.Conv3d(base_n_filter, classes, kernel_size=1, stride=1, padding=0, bias=False)
#
#         self.sigmoid = nn.Sigmoid()
#
#     def upblock(self, feat_in, feat_out):
#         return nn.Sequential(
#             nn.InstanceNorm3d(feat_in),
#             nn.LeakyReLU(),
#             nn.Upsample(scale_factor=2, mode='nearest'),
#             nn.Conv3d(feat_in, feat_out, kernel_size=3, stride=1, padding=1, bias=False),
#             nn.InstanceNorm3d(feat_out),
#             nn.LeakyReLU())
#
#     def forward(self, x):
#         # print('x', x.shape)
#         db0 = self.downblock0(x)
#         # print('db0', db0.shape)
#         db1 = self.downblock1(db0)
#         # print('db1', db1.shape)
#         db2 = self.downblock2(db1)
#         # print('db2', db2.shape)
#         db3 = self.downblock3(db2)
#         # print('db3', db3.shape)
#
#         mid0 = self.midblock0(db3)
#         # print('mid0', mid0.shape)
#
#         up1 = torch.cat([mid0, db3], dim=1)
#         # print('up1', up1.shape)
#         mid1=self.midblock1(up1)
#         mid1 = self.upblock1(mid1)
#         # print('mid1', mid1.shape)
#
#
#         up2 = torch.cat([mid1, db2], dim=1)
#         # print('up2', up2.shape)
#         mid2 = self.midblock2(up2)
#         mid2=self.upblock2(mid2)
#         # print('mid3', mid2.shape)
#
#
#         up3 = torch.cat([mid2, db1], dim=1)
#         # print('up3', up3.shape)
#         mid3 = self.midblock3(up3)
#         mid3=self.upblock3(mid3)
#         # print('mid3', mid3.shape)
#
#         out=self.sigmoid(self.output(mid3))
#         # print('out',out.shape)
#
#         return out
