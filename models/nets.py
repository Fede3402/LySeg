import torch 
import torch.nn as nn 
from monai.networks.nets import BasicUNet
from .blocks import LiverConditioning

class ConitionedBasicUNet(BasicUNet):
    def __init__(self, spatial_dims, in_channels, out_channels, features, **kwargs):
        super().__init__(
            spatial_dims=spatial_dims, 
            in_channels=in_channels, 
            out_channels=out_channels, 
            features=features, 
            **kwargs
        )
        self.features = features

        bottleneck_channels = self.features[4]

        self.liver_conditioning = LiverConditioning(bottleneck_channels)
    
    def forward(self, x: torch.Tensor, suv_prior: torch.Tensor): 
        
        # Encoder (feature extraxction)
        x1 = self.conv_0(x)
        x2 = self.down_1(x1)
        x3 = self.down_2(x2)
        x4 = self.down_3(x3)
        x5 = self.down_4(x4)
        
        # Liver conditioning
        x5 = self.liver_conditioning(x5, suv_prior)

        # Decoder 
        x_up = self.upcat_4(x5, x4)
        x_up = self.upcat_3(x_up, x3)
        x_up = self.upcat_2(x_up, x2)
        x_up = self.upcat_1(x_up, x1)

        logits = self.final_conv(x_up)

        return logits




    