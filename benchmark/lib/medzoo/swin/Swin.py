import torch.nn as nn
from .swin_transformer import SwinTransformer
from .uper_head import UPerHead


class Swin(nn.Module):
    def __init__(self):
        self.swin = SwinTransformer(embed_dim=96,
                                    depths=[2, 2, 6, 2],
                                    num_heads=[3, 6, 12, 24],
                                    window_size=7,
                                    ape=False,
                                    drop_path_rate=0.3,
                                    patch_norm=True,
                                    use_checkpoint=False)
        self.uperhead = UPerHead(in_channels=[96, 192, 384, 768],
                                 num_classes=150)

    def forward(self,x):
        x1=self.swin(x)
        out=self.uperhead(x1)
        return out

