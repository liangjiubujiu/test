import torch.nn as nn
import torch
import torch.nn.functional as F
from .attunet.networks_other import init_weights
from torch.nn import init


class A2Attention(nn.Module):

    def __init__(self, in_channels,c_m,c_n,reconstruct = True):
        super().__init__()
        self.in_channels=in_channels
        self.reconstruct = reconstruct
        self.c_m=c_m
        self.c_n=c_n
        self.convA=nn.Conv3d(in_channels,c_m,1)
        self.convB=nn.Conv3d(in_channels,c_n,1)
        self.convV=nn.Conv3d(in_channels,c_n,1)
        if self.reconstruct:
            self.conv_reconstruct = nn.Conv3d(c_m, in_channels, kernel_size = 1)
        self.init_weights()


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        b, c, d,h,w=x.shape
        assert c==self.in_channels
        A=self.convA(x) #b,c_m,d,h,w
        B=self.convB(x) #b,c_n,d,h,w
        V=self.convV(x) #b,c_n,d,h,w
        tmpA=A.view(b,self.c_m,-1)
        attention_maps=F.softmax(B.view(b,self.c_n,-1))
        # print('tmp',tmpA.shape,attention_maps.shape)
        # exit()
        attention_vectors=F.softmax(V.view(b,self.c_n,-1))
        # step 1: feature gating
        global_descriptors=torch.bmm(tmpA,attention_maps.permute(0,2,1)) #b.c_m,c_n
        # step 2: feature distribution
        tmpZ = global_descriptors.matmul(attention_vectors) #b,c_m,d*h*w
        tmpZ=tmpZ.view(b,self.c_m,d,h,w) #b,c_m,h,w
        if self.reconstruct:
            tmpZ=self.conv_reconstruct(tmpZ)

        return tmpZ

class CBAMAttention(nn.Module):

    def __init__(self, channel=512, reduction=16, kernel_size=49):
        super().__init__()
        self.ca = ChannelAttention(channel=channel, reduction=reduction)
        self.sa = SpatialAttention(kernel_size=kernel_size)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        b, c, _, _,_ = x.size()
        residual = x
        out = x * self.ca(x)
        out = out * self.sa(out)
        return out + residual

class DAAttention(nn.Module):

    def __init__(self, d_model=16, kernel_size=3, H=7, W=7):
        super().__init__()
        self.position_attention_module = PositionAttentionModule(d_model=d_model, kernel_size=3, H=64, W=64)
        self.channel_attention_module = ChannelAttentionModule(d_model=d_model, kernel_size=3, H=64, W=64)

    def forward(self, inputt):
        out=torch.zeros_like(inputt).cuda()
        for i in range(out.shape[2]):
            input=inputt[:,:,i,:,:]
            bs, c, h, w = input.shape
            p_out = self.position_attention_module(input)
            c_out = self.channel_attention_module(input)
            p_out = p_out.permute(0, 2, 1).view(bs, c, h, w)
            c_out = c_out.view(bs, c, h, w)
            out[:,:,i,:,:]=c_out+p_out
        return out




class ECAAttention(nn.Module):

    def __init__(self, kernel_size=3):
        super().__init__()
        self.gap=nn.AdaptiveAvgPool3d(1)
        self.conv=nn.Conv1d(1,1,kernel_size=kernel_size,padding=(kernel_size-1)//2)
        self.sigmoid=nn.Sigmoid()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        y=self.gap(x) #bs,c,1,1

        y=y.squeeze(-1).squeeze(-1).permute(0,2,1) #bs,1,c
        y=self.conv(y) #bs,1,c
        y=self.sigmoid(y) #bs,1,c
        y=y.permute(0,2,1).unsqueeze(-1).unsqueeze(-1) #bs,c,1,1
        return x*y.expand_as(x)


class _GridAttentionBlockND(nn.Module):
    def __init__(self, in_channels, gating_channels, inter_channels=None, dimension=3, mode='concatenation',
                 sub_sample_factor=(2, 2, 2)):
        super(_GridAttentionBlockND, self).__init__()

        assert dimension in [2, 3]
        assert mode in ['concatenation', 'concatenation_debug', 'concatenation_residual']

        # Downsampling rate for the input featuremap
        if isinstance(sub_sample_factor, tuple):
            self.sub_sample_factor = sub_sample_factor
        elif isinstance(sub_sample_factor, list):
            self.sub_sample_factor = tuple(sub_sample_factor)
        else:
            self.sub_sample_factor = tuple([sub_sample_factor]) * dimension

        # Default parameter set
        self.mode = mode
        self.dimension = dimension
        self.sub_sample_kernel_size = self.sub_sample_factor
        # print('self.sub_sample_kernel_size', self.sub_sample_kernel_size)
        # Number of channels (pixel dimensions)
        self.in_channels = in_channels
        self.gating_channels = gating_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        if dimension == 3:
            conv_nd = nn.Conv3d
            bn = nn.BatchNorm3d
            self.upsample_mode = 'trilinear'
        elif dimension == 2:
            conv_nd = nn.Conv2d
            bn = nn.BatchNorm2d
            self.upsample_mode = 'bilinear'
        else:
            raise NotImplemented

        # Output transform
        self.W = nn.Sequential(
            conv_nd(in_channels=self.in_channels, out_channels=self.in_channels, kernel_size=1, stride=1, padding=0),
            bn(self.in_channels),
        )

        # Theta^T * x_ij + Phi^T * gating_signal + bias
        self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=self.sub_sample_kernel_size, stride=self.sub_sample_factor, padding=0,
                             bias=False)
        self.phi = conv_nd(in_channels=self.gating_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0, bias=True)
        self.psi = conv_nd(in_channels=self.inter_channels, out_channels=1, kernel_size=1, stride=1, padding=0,
                           bias=True)
        self.attentionblock = ExternalAttention(d_model=16,S=8)
        self.layernorm=nn.LayerNorm([64,64])
        # Initialise weights
        for m in self.children():
            init_weights(m, init_type='kaiming')

    def forward(self, x, g):
        '''
        :param x: (b, c, t, h, w)
        :param g: (b, g_d)
        :return:
        '''
        input_size = x.size()
        batch_size = input_size[0]
        assert batch_size == g.size(0)

        # theta => (b, c, t, h, w) -> (b, i_c, t, h, w) -> (b, i_c, thw)
        # phi   => (b, g_d) -> (b, i_c)
        theta_x = self.theta(x)
        theta_x_size = theta_x.size()
        # print('x', x.shape)
        # print('theta_x', theta_x.shape)

        # g (b, c, t', h', w') -> phi_g (b, i_c, t', h', w')
        #  Relu(theta_x + phi_g + bias) -> f = (b, i_c, thw) -> (b, i_c, t/s1, h/s2, w/s3)
        phi_g = self.phi(g)
        # print('phi(g)', phi_g.shape)
        phi_g = F.upsample(phi_g, size=theta_x_size[2:], mode=self.upsample_mode)
        # print('upsample phi(g)', phi_g.shape)


        #  psi^T * f -> (b, psi_i_c, t/s1, h/s2, w/s3)
        f = F.relu(self.layernorm(theta_x + phi_g), inplace=True)


        f_t=f.permute(0,2,3,4,1).contiguous()
        # print('f',f.shape)

        sigm_psi_f = self.attentionblock(f_t)

        sigm_psi_f=sigm_psi_f.permute(0,4,1,2,3).contiguous()
        # print('sigm', sigm_psi_f.shape)
        # exit()
        # upsample the attentions and multiply
        sigm_psi_f = F.upsample(sigm_psi_f+f, size=input_size[2:], mode=self.upsample_mode)

        return sigm_psi_f,sigm_psi_f

        return output




class GridAttentionBlock3D(_GridAttentionBlockND):
    def __init__(self, in_channels, gating_channels, inter_channels=None, mode='concatenation',
                 sub_sample_factor=(2, 2, 2)):
        super(GridAttentionBlock3D, self).__init__(in_channels,
                                                   inter_channels=inter_channels,
                                                   gating_channels=gating_channels,
                                                   dimension=3, mode=mode,
                                                   sub_sample_factor=sub_sample_factor,
                                                   )



class ExternalAttention(nn.Module):

    def __init__(self, d_model,S=64):
        super().__init__()
        self.mk=nn.Linear(d_model,S,bias=False)
        self.mv=nn.Linear(S,d_model,bias=False)
        self.softmax=nn.Softmax(dim=1)
        self.init_weights()


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, queries):
        attn=self.mk(queries) #bs,n,S
        attn=self.softmax(attn) #bs,n,S
        attn=attn/torch.sum(attn,dim=2,keepdim=True) #bs,n,S
        out=self.mv(attn) #bs,n,d_model

        return out

class GCAttention(nn.Module):

    def __init__(self,
                 inplanes,
                 ratio,
                 pooling_type='att',
                 fusion_types=('channel_add', )):
        super(GCAttention, self).__init__()
        assert pooling_type in ['avg', 'att']
        assert isinstance(fusion_types, (list, tuple))
        valid_fusion_types = ['channel_add', 'channel_mul']
        assert all([f in valid_fusion_types for f in fusion_types])
        assert len(fusion_types) > 0, 'at least one fusion should be used'
        self.inplanes = inplanes
        self.ratio = ratio
        self.planes = int(inplanes * ratio)
        self.pooling_type = pooling_type
        self.fusion_types = fusion_types
        if pooling_type == 'att':
            self.conv_mask = nn.Conv2d(inplanes, 1, kernel_size=1)
            self.softmax = nn.Softmax(dim=2)
        else:
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
        if 'channel_add' in fusion_types:
            self.channel_add_conv = nn.Sequential(
                nn.Conv2d(self.inplanes, self.planes, kernel_size=1),
                nn.LayerNorm([self.planes, 1, 1]),
                nn.ReLU(inplace=True),  # yapf: disable
                nn.Conv2d(self.planes, self.inplanes, kernel_size=1))
        else:
            self.channel_add_conv = None
        if 'channel_mul' in fusion_types:
            self.channel_mul_conv = nn.Sequential(
                nn.Conv2d(self.inplanes, self.planes, kernel_size=1),
                nn.LayerNorm([self.planes, 1, 1]),
                nn.ReLU(inplace=True),  # yapf: disable
                nn.Conv2d(self.planes, self.inplanes, kernel_size=1))
        else:
            self.channel_mul_conv = None
        # initialise the blocks
        for m in self.children():
            init_weights(m, init_type='kaiming')

    def init_weights(net, init_type='normal'):
        # print('initialization method [%s]' % init_type)
        if init_type == 'normal':
            net.apply(weights_init_normal)
        elif init_type == 'xavier':
            net.apply(weights_init_xavier)
        elif init_type == 'kaiming':
            net.apply(weights_init_kaiming)
        elif init_type == 'orthogonal':
            net.apply(weights_init_orthogonal)
        else:
            raise NotImplementedError('initialization method [%s] is not implemented' % init_type)

    def spatial_pool(self, x):
        batch, channel, height, width = x.size()
        if self.pooling_type == 'att':
            input_x = x
            # [N, C, H * W]
            input_x = input_x.view(batch, channel, height * width)
            # [N, 1, C, H * W]
            input_x = input_x.unsqueeze(1)
            # [N, 1, H, W]
            context_mask = self.conv_mask(x)
            # [N, 1, H * W]
            context_mask = context_mask.view(batch, 1, height * width)
            # [N, 1, H * W]
            context_mask = self.softmax(context_mask)
            # [N, 1, H * W, 1]
            context_mask = context_mask.unsqueeze(-1)
            # [N, 1, C, 1]
            context = torch.matmul(input_x, context_mask)
            # [N, C, 1, 1]
            context = context.view(batch, channel, 1, 1)
        else:
            # [N, C, 1, 1]
            context = self.avg_pool(x)

        return context

    def forward(self, x):
        # [N, C, 1, 1]
        context = self.spatial_pool(x)

        out = x
        if self.channel_mul_conv is not None:
            # [N, C, 1, 1]
            channel_mul_term = torch.sigmoid(self.channel_mul_conv(context))
            out = out * channel_mul_term
        if self.channel_add_conv is not None:
            # [N, C, 1, 1]
            channel_add_term = self.channel_add_conv(context)
            out = out + channel_add_term

        return out

class OutlookAttention(nn.Module):

    def __init__(self,dim,num_heads=1,kernel_size=3,padding=1,stride=1,qkv_bias=False,
                    attn_drop=0.1):
        super().__init__()
        self.dim=dim
        self.num_heads=num_heads
        self.head_dim=dim//num_heads
        self.kernel_size=kernel_size
        self.padding=padding
        self.stride=stride
        self.scale=self.head_dim**(-0.5)

        self.v_pj=nn.Linear(dim,dim,bias=qkv_bias)
        self.attn=nn.Linear(dim,kernel_size**4*num_heads)

        self.attn_drop=nn.Dropout(attn_drop)
        self.proj=nn.Linear(dim,dim)
        self.proj_drop=nn.Dropout(attn_drop)

        self.unflod=nn.Unfold(kernel_size,padding,stride) #手动卷积
        self.pool=nn.AvgPool2d(kernel_size=stride,stride=stride,ceil_mode=True)
        self.out = nn.Conv3d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1)

    def forward(self, xx) :
        outt=torch.zeros_like(xx).cuda()
        for i in range(xx.shape[2]):
            x=xx[:,:,i,:,:].permute(0,2,3,1)
            B,H,W,C=x.shape

            #映射到新的特征v
            v=self.v_pj(x).permute(0,3,1,2) #B,C,H,W
            h,w=math.ceil(H/self.stride),math.ceil(W/self.stride)
            v=self.unflod(v).reshape(B,self.num_heads,self.head_dim,self.kernel_size*self.kernel_size,h*w).permute(0,1,4,3,2) #B,num_head,H*W,kxk,head_dim

            #生成Attention Map
            attn=self.pool(x.permute(0,3,1,2)).permute(0,2,3,1) #B,H,W,C
            attn=self.attn(attn).reshape(B,h*w,self.num_heads,self.kernel_size*self.kernel_size \
                        ,self.kernel_size*self.kernel_size).permute(0,2,1,3,4) #B，num_head，H*W,kxk,kxk
            attn=self.scale*attn
            attn=attn.softmax(-1)
            attn=self.attn_drop(attn)

            #获取weighted特征
            out=(attn @ v).permute(0,1,4,3,2).reshape(B,C*self.kernel_size*self.kernel_size,h*w) #B,dimxkxk,H*W
            out=F.fold(out,output_size=(H,W),kernel_size=self.kernel_size,
                        padding=self.padding,stride=self.stride) #B,C,H,W
            out=self.proj(out.permute(0,2,3,1)) #B,H,W,C
            out=self.proj_drop(out)
            out=out.permute(0,3,1,2)
            outt[:,:,i,:,:]=out
            outt=self.out(outt)



        return outt


class SequentialPolarizedSelfAttention(nn.Module):

    def   __init__(self, channel=512):
        super().__init__()
        self.ch_wv=nn.Conv2d(channel,channel//2,kernel_size=(1,1))
        self.ch_wq=nn.Conv2d(channel,1,kernel_size=(1,1))
        self.softmax_channel=nn.Softmax(1)
        self.softmax_spatial=nn.Softmax(-1)
        self.ch_wz=nn.Conv2d(channel//2,channel,kernel_size=(1,1))
        self.ln=nn.LayerNorm(channel)
        self.sigmoid=nn.Sigmoid()
        self.sp_wv=nn.Conv2d(channel,channel//2,kernel_size=(1,1))
        self.sp_wq=nn.Conv2d(channel,channel//2,kernel_size=(1,1))
        self.agp=nn.AdaptiveAvgPool2d((1,1))

    def forward(self, xx):
        out=torch.zeros_like(xx).cuda()
        for i in range(xx.shape[2]):
            x=xx[:,:,i,:,:]
            b, c, h, w = x.size()

            #Channel-only Self-Attention
            channel_wv=self.ch_wv(x) #bs,c//2,h,w
            channel_wq=self.ch_wq(x) #bs,1,h,w
            channel_wv=channel_wv.reshape(b,c//2,-1) #bs,c//2,h*w
            channel_wq=channel_wq.reshape(b,-1,1) #bs,h*w,1
            channel_wq=self.softmax_channel(channel_wq)
            channel_wz=torch.matmul(channel_wv,channel_wq).unsqueeze(-1) #bs,c//2,1,1
            channel_weight=self.sigmoid(self.ln(self.ch_wz(channel_wz).reshape(b,c,1).permute(0,2,1))).permute(0,2,1).reshape(b,c,1,1) #bs,c,1,1
            channel_out=channel_weight*x

            #Spatial-only Self-Attention
            spatial_wv=self.sp_wv(channel_out) #bs,c//2,h,w
            spatial_wq=self.sp_wq(channel_out) #bs,c//2,h,w
            spatial_wq=self.agp(spatial_wq) #bs,c//2,1,1
            spatial_wv=spatial_wv.reshape(b,c//2,-1) #bs,c//2,h*w
            spatial_wq=spatial_wq.permute(0,2,3,1).reshape(b,1,c//2) #bs,1,c//2
            spatial_wq=self.softmax_spatial(spatial_wq)
            spatial_wz=torch.matmul(spatial_wq,spatial_wv) #bs,1,h*w
            spatial_weight=self.sigmoid(spatial_wz.reshape(b,1,h,w)) #bs,1,h,w
            spatial_out=spatial_weight*channel_out
            out[:,:,i,:,:]=spatial_out
        return out



class PSAAttention(nn.Module):

    def __init__(self, channel=16, reduction=2, S=4):
        super().__init__()
        self.S = S

        self.convs = nn.Sequential(nn.Conv3d(channel // S, channel // S, kernel_size=3, padding=1),#repeat 4 times
                                   # kernel_size=2 * (i + 1) + 1, padding=i + 1
                                   nn.Conv3d(channel // S, channel // S, kernel_size=5, padding=2),
                                  )

        self.se_blocks=nn.Sequential(#repeat 4 times
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(channel // S, channel // (S * reduction), kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(channel // (S * reduction), channel // S, kernel_size=1, bias=False),
            nn.Sigmoid(),

        )

        self.softmax = nn.Softmax(dim=1)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        b, c, d, h, w = x.size()

        # Step1:SPC module
        SPC_outt = x.view(b, self.S, c // self.S, d, h, w)  # bs,s,ci,d,h,w

        SPC_out=torch.zeros_like(SPC_outt).cuda()
        # print('spc', SPC_out.shape)
        for idx in range(self.S):
            SPC_out[:, idx, :, :, :, :] = self.convs(SPC_outt[:, idx, :, :, :, :])
        # print('spc', SPC_out.shape)


        # Step2:SE weight
        se_out = []
        for idx in range(self.S):
            se_out.append(self.se_blocks(SPC_out[:, idx, :, :, :, :]))

        SE_out = torch.stack(se_out, dim=1)
        # print('se',SE_out.shape)
        SE_out = SE_out.expand_as(SPC_out)
        # print('se after', SE_out.shape)
        # Step3:Softmax
        softmax_out = self.softmax(SE_out)

        # Step4:SPA
        PSA_out = SPC_out * softmax_out
        PSA_out = PSA_out.view(b, -1, d, h, w)

        return PSA_out



class S2Attention(nn.Module):

    def __init__(self, channels=16):
        super().__init__()
        self.mlp1 = nn.Linear(channels, channels * 3)
        self.mlp2 = nn.Linear(channels, channels)
        self.split_attention = SplitAttention(channel=channels)

    def forward(self, xx):
        out=torch.zeros_like(xx).cuda()
        for i in range(xx.shape[2]):
            x=xx[:,:,i,:,:]
            b, c, w, h = x.size()
            # print(x.shape)
            x = x.permute(0, 2, 3, 1)
            x = self.mlp1(x)
            x1 = spatial_shift1(x[:, :, :, :c])
            x2 = spatial_shift2(x[:, :, :, c:c * 2])
            x3 = x[:, :, :, c * 2:]
            x_all = torch.stack([x1, x2, x3], 1)
            # print('xall',x_all.shape)
            a = self.split_attention(x_all)
            x = self.mlp2(a)
            x = x.permute(0, 3, 1, 2)
            out[:,:,i,:,:]=x
        return out


class SEAttention(nn.Module):

    def __init__(self, channel=512, reduction=2):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        b, c, _, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1, 1)
        return x * y.expand_as(x)

    @staticmethod
    def apply_argmax_softmax(pred):
        log_p = F.softmax(pred, dim=1)

        return log_p



class SpatialGroupEnhanceAttention(nn.Module):

    def __init__(self, groups):
        super().__init__()
        self.groups = groups
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.weight = nn.Parameter(torch.zeros(1, groups,1, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, groups,1, 1, 1))
        self.sig = nn.Sigmoid()
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        b, c,d, h, w = x.shape
        x = x.view(b * self.groups, -1,d, h, w)  # bs*g,dim//g,d,h,w
        xn = x * self.avg_pool(x)  # bs*g,dim//g,d,h,w
        xn = xn.sum(dim=1, keepdim=True)  # bs*g,1,d,h,w
        t = xn.view(b * self.groups, -1)  # bs*g,d*h*w

        t = t - t.mean(dim=1, keepdim=True)  # bs*g,d*h*w
        std = t.std(dim=1, keepdim=True) + 1e-5
        t = t / std  # bs*g,d*h*w
        t = t.view(b, self.groups,d, h, w)  # bs,g,d*h*w

        t = t * self.weight + self.bias  # bs,g,d*h*w
        t = t.view(b * self.groups, 1, d,h, w)  # bs*g,1,d*h*w
        x = x * self.sig(t)
        x = x.view(b, c, d,h, w)

        return x


class ShuffleAttention(nn.Module):

    def __init__(self, channel=512,reduction=16,G=8):
        super().__init__()
        self.G=G
        self.channel=channel
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.gn = nn.GroupNorm(channel // (2 * G), channel // (2 * G))
        self.cweight = Parameter(torch.zeros(1, channel // (2 * G), 1, 1))
        self.cbias = Parameter(torch.ones(1, channel // (2 * G), 1, 1))
        self.sweight = Parameter(torch.zeros(1, channel // (2 * G), 1, 1))
        self.sbias = Parameter(torch.ones(1, channel // (2 * G), 1, 1))
        self.sigmoid=nn.Sigmoid()


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)


    @staticmethod
    def channel_shuffle(x, groups):
        b, c, d,h, w = x.shape
        x = x.reshape(b, groups, -1, d,h, w)
        x = x.permute(0, 2, 1, 3, 4,5)

        # flatten
        x = x.reshape(b, -1, d,h, w)

        return x

    def forward(self, x):
        b, c, d,h, w = x.size()
        #group into subfeatures
        x=x.view(b*self.G,-1,d,h,w) #bs*G,c//G,d,h,w

        #channel_split
        x_0,x_1=x.chunk(2,dim=1) #bs*G,c//(2*G),d,h,w

        #channel attention
        x_channel=self.avg_pool(x_0) #bs*G,c//(2*G),1,1,1
        x_channel=self.cweight*x_channel+self.cbias #bs*G,c//(2*G),1,1,1
        x_channel=x_0*self.sigmoid(x_channel)

        #spatial attention
        x_spatial=self.gn(x_1) #bs*G,c//(2*G),d,h,w
        x_spatial=self.sweight*x_spatial+self.sbias #bs*G,c//(2*G),d,h,w
        x_spatial=x_1*self.sigmoid(x_spatial) #bs*G,c//(2*G),d,h,w

        # concatenate along channel axis
        out=torch.cat([x_channel,x_spatial],dim=1)  #bs*G,c//G,d,h,w
        out=out.contiguous().view(b,-1,d,h,w)

        # channel shuffle
        out = self.channel_shuffle(out, 2)
        return out


class SRMAttention(nn.Module):
    def __init__(self, channel):
        super(SRMAttention, self).__init__()

        self.cfc = Parameter(torch.Tensor(channel, 2))
        self.cfc.data.fill_(0)

        self.bn = nn.LayerNorm(channel)
        self.activation = nn.Sigmoid()

        # setattr(self.cfc, 'srm_param', True)
        # setattr(self.bn.weight, 'srm_param', True)
        # setattr(self.bn.bias, 'srm_param', True)

    def _style_pooling(self, x, eps=1e-5):
        N, C, _, _= x.size()

        channel_mean = x.view(N, C, -1).mean(dim=2, keepdim=True)
        channel_var = x.view(N, C, -1).var(dim=2, keepdim=True) + eps
        channel_std = channel_var.sqrt()

        t = torch.cat((channel_mean, channel_std), dim=2)
        return t

    def _style_integration(self, t):
        # print(self.cfc[None, :,:].shape)
        # print(t.shape)
        z = t * self.cfc[None, :,:]  # B x C x 2
        z = torch.sum(z, dim=2)[:, :, None, None]  # B x C x 1 x 1

        # z_hat = self.bn(z)
        z_hat=z
        g = self.activation(z_hat)

        return g

    def forward(self, xx):
        # B x C x 2
        out=torch.zeros_like(xx).cuda()
        for i in range(out.shape[2]):
            x=xx[:,:,i,:,:]
            t = self._style_pooling(x)
            # print(t.shape)
            # B x C x 1 x 1
            g = self._style_integration(t)
            out[:,:,i,:,:]=x * g

        return out



class TripletAttention(nn.Module):
    def __init__(self, no_spatial=False):
        super(TripletAttention, self).__init__()
        self.cw = AttentionGate()
        self.hc = AttentionGate()
        self.no_spatial=no_spatial
        self.out=nn.Conv3d(in_channels=16,out_channels=16,kernel_size=3,stride=1,padding=1)
        if not no_spatial:
            self.hw = AttentionGate()
    def forward(self, xx):
        out = torch.zeros_like(xx).cuda()
        for i in range(xx.shape[2]):
            x = xx[:, :, i, :, :]
            x_perm1 = x.permute(0,2,1,3).contiguous()
            x_out1 = self.cw(x_perm1)
            x_out11 = x_out1.permute(0,2,1,3).contiguous()
            x_perm2 = x.permute(0,3,2,1).contiguous()
            x_out2 = self.hc(x_perm2)
            x_out21 = x_out2.permute(0,3,2,1).contiguous()
            if not self.no_spatial:
                x_out = self.hw(x)
                x_out = 1/3 * (x_out + x_out11 + x_out21)
            else:
                x_out = 1/2 * (x_out11 + x_out21)
            out[:, :, i, :, :] = x_out

        return out



class SKAttention(nn.Module):

    def __init__(self, channel=512,kernels=[1,3,5,7],reduction=16,group=1,L=16):
        super().__init__()
        self.d=max(L,channel//reduction)
        # print('self.d',self.d)
        self.convs=nn.ModuleList([])
        for k in kernels:
            self.convs.append(
                nn.Sequential(OrderedDict([
                    ('conv',nn.Conv3d(channel,channel,kernel_size=k,padding=k//2,groups=group)),
                    ('bn',nn.BatchNorm3d(channel)),
                    ('relu',nn.ReLU())
                ]))
            )
        self.fc=nn.Linear(channel,self.d)
        self.fcs=nn.ModuleList([])
        for i in range(len(kernels)):
            self.fcs.append(nn.Linear(self.d,channel))
        self.softmax=nn.Softmax(dim=1)


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        bs, c, _, _,_ = x.size()
        conv_outs=[]
        ### split
        for conv in self.convs:
            conv_outs.append(conv(x))
        feats=torch.stack(conv_outs,0)#k,bs,channel,h,w

        ### fuse
        U=sum(conv_outs) #bs,c,h,w

        ### reduction channel
        S=U.mean(-1).mean(-1).mean(-1) #bs,c
        # print('s SHAPE',S.shape,U.shape)

        Z=self.fc(S) #bs,d
        # print(S.shape, Z.shape)
        ### calculate attention weight
        weights=[]
        for fc in self.fcs:
            weight=fc(Z)
            weights.append(weight.view(bs,c,1,1,1)) #bs,channel
        attention_weughts=torch.stack(weights,0)#k,bs,channel,1,1
        attention_weughts=self.softmax(attention_weughts)#k,bs,channel,1,1

        ### fuse
        V=(attention_weughts*feats).sum(0)
        return V