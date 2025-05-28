import torch.optim as optim

from .DenseVoxelNet import DenseVoxelNet
from .Densenet3D import DualPathDenseNet, DualSingleDenseNet, SinglePathDenseNet
from .HighResNet3D import HighResNet3D
from .HyperDensenet import HyperDenseNet, HyperDenseNet_2Mod
from .ResNet3DMedNet import generate_resnet3d
from .ResNet3D_VAE import ResNet3dVAE
from .SkipDenseNet3D import SkipDenseNet3D
from .Unet2D import Unet
from .Unet3D import UNet3D
from .Unet3DMINI import UNet3Dmini
from .TransMINI import Transmini
from .Vnet import VNet, VNetLight
from .CBCTpix import CBCTpix
from .TransBTS.TransBTS.TransBTS_downsample8x_skipconnection import TransBTS
# from .ToothPix.Config.gan_xre_options import gan_xre_3d
# from .ToothPix.Model import create_toothpix
from .swin.Swin import Swin

from .Seunetmini import Seunetmini
model_list = ['SWIN','CBCTpix','SEUNETMINI','UNET3DMINI','UNET3D', 'DENSENET1', "UNET2D", 'DENSENET2', 'DENSENET3', 'HYPERDENSENET', "SKIPDENSENET3D",
              "DENSEVOXELNET", 'VNET', 'VNET2', "RESNET3DVAE", "RESNETMED3D", "COVIDNET1", "COVIDNET2", "CNN",
              "HIGHRESNET",'NONLOCALUNET3D','CCATTENTIONUNET3D','TransBTS','ToothPix','TransMINI','resnet', 'preresnet', 'wideresnet', 'resnext', 'densenet','senet']


def create_model(args):
    model_name = args.model
    model_name_depth=args.model_depth
    assert model_name in model_list
    optimizer_name = args.opt
    lr = args.lr
    in_channels = args.inChannels
    num_classes = args.classes
    weight_decay = 0#0.0002
    print("Building ToothPix . . . . . . . ." + model_name)

    if model_name == 'VNET2':
        model = VNetLight(in_channels=in_channels, elu=False, classes=num_classes)
    elif model_name == 'VNET':
        model = VNet(in_channels=in_channels, elu=False, classes=num_classes)
    elif model_name == 'UNET3D':
        model = UNet3D(in_channels=in_channels, classes=num_classes, base_n_filter=8)
    elif model_name == 'UNET3DMINI':
        model = UNet3Dmini(in_channels=in_channels, n_classes=num_classes, base_n_filter=8)
    elif model_name=='SWIN':
        model = Swin()
    elif model_name == 'DENSENET1':
        model = SinglePathDenseNet(in_channels=in_channels, classes=num_classes)
    elif model_name == 'DENSENET2':
        model = DualPathDenseNet(in_channels=in_channels, classes=num_classes)
    elif model_name == 'DENSENET3':
        model = DualSingleDenseNet(in_channels=in_channels, drop_rate=0.1, classes=num_classes)
    elif model_name == "UNET2D":
        model = Unet(in_channels, num_classes)
    elif model_name == "RESNET3DVAE":
        model = ResNet3dVAE(in_channels=in_channels, classes=num_classes, dim=args.dim)
    elif model_name == "SKIPDENSENET3D":
        model = SkipDenseNet3D(growth_rate=16, num_init_features=32, drop_rate=0.1, classes=num_classes)

    elif model_name == "HYPERDENSENET":
        if in_channels == 2:
            model = HyperDenseNet_2Mod(classes=num_classes)
        elif in_channels == 3:
            model = HyperDenseNet(classes=num_classes)
        else:
            raise NotImplementedError
    elif model_name == "DENSEVOXELNET":
        model = DenseVoxelNet(in_channels=in_channels, classes=num_classes)
    elif model_name == "HIGHRESNET":
        model = HighResNet3D(in_channels=in_channels, classes=num_classes)
    elif model_name == "RESNETMED3D":
        depth = 18
        model = generate_resnet3d(in_channels=in_channels, classes=num_classes, model_depth=depth)
    elif model_name=='TransBTS':
        _, model = TransBTS(args,dataset='brats', _conv_repr=True, _pe_type="learned")
    elif model_name=='TransMINI':
        model=Transmini(in_channels=in_channels, classes=num_classes, base_n_filter=8)
    elif model_name=='CBCTpix':
        model=CBCTpix(in_channels=in_channels, num_classes=num_classes, base_n_filter=8)
    elif model_name=='SEUNETMINI':
        model=Seunetmini(in_channels=in_channels, n_classes=num_classes, base_n_filter=8)

    print(model_name, 'Number of params: {}'.format(
        sum([p.data.nelement() for p in model.parameters()])))

    if optimizer_name == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.5, weight_decay=weight_decay)
    elif optimizer_name == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=lr, betas = (0.9, 0.999), eps = 1e-08, weight_decay=weight_decay)
    elif optimizer_name == 'rmsprop':
        optimizer = optim.RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay)

    return model, optimizer
