import torch 
import torch.nn as nn 

from monai.networks.nets import BasicUNet
from monai.networks.blocks import ResidualUnit, Convolution

from .blocks import LiverConditioning, FiLMconditioning


class ConditionedBasicUNet(BasicUNet):
    def __init__(self, spatial_dims, in_channels, out_channels, features=(16, 32, 64, 128, 256, 16), **kwargs):
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
        
        # Encoder (feature extraction)
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
    
class PriorUNet(nn.Module):
    def __init__(
            self, 
            spatial_dims = 3, 
            in_channels = 2, 
            out_channels = 1, 
            features = (16,32,64,128,256),
            num_prior_stats = 1,
            dropout = 0.2
    ):
        super().__init__()
        norm = ("instance", {"affine": True})

        def _get_res_block(in_c, out_c, stride=1):
            return ResidualUnit(
                spatial_dims=spatial_dims,
                in_channels=in_c,
                out_channels=out_c,
                strides=stride,
                kernel_size=3,
                subunits=2,
                norm=norm,
                dropout=dropout,
            )
        
        def _get_upsample(in_c, out_c): 
            return Convolution(
                spatial_dims=spatial_dims,
                in_channels=in_c,
                out_channels=out_c,
                strides=2,
                kernel_size=3,
                is_transposed=True,
                norm=norm,
                dropout=dropout,
            )
        
        # Encoder 
        self.enc1 = _get_res_block(in_channels, features[0], stride = 1)
        self.enc2 = _get_res_block(features[0], features[1], stride = 2)
        self.enc3 = _get_res_block(features[1], features[2], stride = 2)
        self.enc4 = _get_res_block(features[2], features[3], stride = 2)

        # Bottleneck + FiLM conditioning 
        self.bottleneck = _get_res_block(features[3], features[4], stride = 2)
        self.film_bottle = FiLMconditioning(num_prior_stats, features[4], dropout_rate=dropout)

        # Decoder 4 + FiLM
        self.up4 = _get_upsample(features[4], features[3])
        self.dec4 = _get_res_block(features[3] * 2, features[3]) # *2 because the input is the concatenated image
        self.film4 = FiLMconditioning(num_prior_stats, features[3], dropout_rate=dropout)

        # Decoder 3 + FiLM
        self.up3 = _get_upsample(features[3], features[2])
        self.dec3 = _get_res_block(features[2] * 2, features[2])
        self.film3 = FiLMconditioning(num_prior_stats, features[2], dropout_rate=dropout)

        # Decoder 2
        self.up2 = _get_upsample(features[2], features[1])
        self.dec2 = _get_res_block(features[1] * 2, features[1])

        # Decoder 1 
        self.up1 = _get_upsample(features[1], features[0])
        self.dec1 = _get_res_block(features[0] * 2, features[0])

        # Out
        self.out_conv = Convolution(
            spatial_dims= spatial_dims,
            in_channels= features[0],
            out_channels= out_channels,
            kernel_size = 1,
            conv_only = True
        )

    def forward(self, x, prior_stats):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)

        # Bottleneck
        b = self.bottleneck(e4)
        b = self.film_bottle(b, prior_stats)

        # Decoder 4 
        d4 = self.up4(b)
        d4 = self.dec4(torch.cat((d4, e4), dim=1))
        d4 = self.film4(d4, prior_stats)

        # Decoder 3 
        d3 = self.up3(d4)
        d3 = self.dec3(torch.cat((d3, e3), dim=1))
        d3 = self.film3(d3, prior_stats)

        # Decoder 2 
        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat((d2, e2), dim=1))

        # Decoder 1
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat((d1, e1), dim=1))

        # Out
        out = self.out_conv(d1)

        return out
    
class StandardCustomUNet(nn.Module):
    def __init__(
            self, 
            spatial_dims = 3, 
            in_channels = 2, 
            out_channels = 1, 
            features = (16,32,64,128,256),
            dropout = 0.2
    ):
        super().__init__()
        norm = ("instance", {"affine": True})

        def _get_res_block(in_c, out_c, stride=1):
            return ResidualUnit(
                spatial_dims=spatial_dims,
                in_channels=in_c,
                out_channels=out_c,
                strides=stride,
                kernel_size=3,
                subunits=2,
                norm=norm,
                dropout=dropout,
            )
        
        def _get_upsample(in_c, out_c): 
            return Convolution(
                spatial_dims=spatial_dims,
                in_channels=in_c,
                out_channels=out_c,
                strides=2,
                kernel_size=3,
                is_transposed=True,
                norm=norm,
                dropout=dropout,
            )
        
        # Encoder 
        self.enc1 = _get_res_block(in_channels, features[0], stride = 1)
        self.enc2 = _get_res_block(features[0], features[1], stride = 2)
        self.enc3 = _get_res_block(features[1], features[2], stride = 2)
        self.enc4 = _get_res_block(features[2], features[3], stride = 2)

        # Bottleneck
        self.bottleneck = _get_res_block(features[3], features[4], stride = 2)

        # Decoder 4 (Senza FiLM)
        self.up4 = _get_upsample(features[4], features[3])
        self.dec4 = _get_res_block(features[3] * 2, features[3])

        # Decoder 3 (Senza FiLM)
        self.up3 = _get_upsample(features[3], features[2])
        self.dec3 = _get_res_block(features[2] * 2, features[2])

        # Decoder 2
        self.up2 = _get_upsample(features[2], features[1])
        self.dec2 = _get_res_block(features[1] * 2, features[1])

        # Decoder 1
        self.up1 = _get_upsample(features[1], features[0])
        self.dec1 = _get_res_block(features[0] * 2, features[0])

        self.out_conv = Convolution(
            spatial_dims= spatial_dims,
            in_channels= features[0],
            out_channels= out_channels,
            kernel_size = 1,
            conv_only = True
        )

    def forward(self, x, prior_stats=None):
        # Il parametro prior_stats viene accettato ma ignorato
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)

        b = self.bottleneck(e4)

        d4 = self.up4(b)
        d4 = self.dec4(torch.cat((d4, e4), dim=1))

        d3 = self.up3(d4)
        d3 = self.dec3(torch.cat((d3, e3), dim=1))

        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat((d2, e2), dim=1))

        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat((d1, e1), dim=1))

        out = self.out_conv(d1)

        return out

        










    