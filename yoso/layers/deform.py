import torch
from torch import nn

from torch import Tensor
from mmcv.cnn import build_norm_layer
from mmcv.ops import DeformConv2d, ModulatedDeformConv2d
from mmengine.model import BaseModule
from mmdet.utils import ConfigType, OptMultiConfig

import math


class DeformLayer(BaseModule):
    """ DeformLayer acts as the top-down bridge for the YOSO FPN (Feature Pyramid Aggregator).  
    This layer consist of Deformable Convolution (DCN) and upsampling.

    Args:
        in_channels (int): Number of channels in the input feature map.
        out_channels (int): Number of channels for output.
        use_modulate_deform (bool): Whether to use ModulatedDeformConv2d (True) or DeformConv2d (False) on DeformLayer.
            Defaults to True.
        num_groups (int): Same as nn.Conv2d.
            Defaults to 1.
        dilation (int): Same as nn.Conv2d.
            Defaults to 1.
        deform_num_groups (int): Number of groups used in deformable convolution.
            Defaults to 1.
        norm_cfg (:obj:`ConfigDict` or dict): Config for normalization.
            Defaults to None.
        init_cfg (:obj:`ConfigDict` or dict or list[:obj:`ConfigDict` or \
            dict], optional): Initialization config dict.
            Defaults to [ dict(type='Kaiming', layer=['ModulatedDeformConv2d', 'DeformConv2d']),
                          dict(type='Constant', val=0, override=dict(name='dcn_offset'))].     
    """
    def __init__(self, 
                 in_channels: int,
                 out_channels: int,
                 use_modulate_deform: bool = True,
                 num_groups: int = 1,
                 dilation: int = 1,
                 deform_num_groups: int = 1, 
                 norm_cfg: ConfigType = None,
                 init_cfg: OptMultiConfig = [
                     dict(type='Kaiming', layer=['ModulatedDeformConv2d', 'DeformConv2d']),
                     dict(type='Constant', val=0, override=dict(name='dcn_offset'))
                 ]) -> None:
        super().__init__(init_cfg=init_cfg)
        self.use_modulate_deform = use_modulate_deform

        if self.use_modulate_deform:
            dcn_op = ModulatedDeformConv2d
            # offset channels are 2 or 3 (if with modulated) * kernel_size * kernel_size
            offset_channels = 27
        else:
            dcn_op = DeformConv2d
            offset_channels = 18
        
        self.dcn_offset = nn.Conv2d(in_channels,
                                    offset_channels * deform_num_groups,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1*dilation,
                                    dilation=dilation)

        self.dcn = dcn_op(in_channels,
                          out_channels,
                          kernel_size=3,
                          stride=1,
                          padding=1*dilation,
                          bias=False,
                          groups=num_groups,
                          dilation=dilation,
                          deform_groups=deform_num_groups)

        self.dcn_bn = build_norm_layer(norm_cfg, out_channels)[1]
        self.relu = nn.ReLU()
        
        # Deconvolution
        self.upsample = nn.ConvTranspose2d(in_channels=out_channels,
                                            out_channels=out_channels,
                                            kernel_size=4,
                                            stride=2, padding=1,
                                            output_padding=0,
                                            bias=False)

        self.upsample_bn = build_norm_layer(norm_cfg, out_channels)[1]

    def forward(self, x: Tensor) -> Tensor:
        """Forward function.
        Args:
            x (Tensor): Feature map that is going to be upsampled,
                with shape (batch_size, c, h, w).

        Returns:
            Tensor: Upsampled feature, 
                with shape (batch_size, c, h*2, w*2).
        """
        out = self.dcn_offset(x)

        if self.use_modulate_deform:
            offset_x, offset_y, mask = torch.chunk(out, 3, dim=1)
            offset = torch.cat((offset_x, offset_y), dim=1)
            mask = mask.sigmoid()

            out = self.dcn(x, offset, mask)
        else: 
            out = self.dcn(x, out)
        
        out = self.dcn_bn(out)
        out = self.relu(out)
        out = self.upsample(out)

        out = self.upsample_bn(out)
        out = self.relu(out)

        return out

    def init_weights(self) -> None:
        """Initialize weights."""
        super().init_weights()
        self._deconv_init()

    def _deconv_init(self) -> None:
        """Weight initialization for Deconvolution."""
        w = self.upsample.weight.data
        f = math.ceil(w.size(2) / 2)
        c = (2 * f - 1 - f % 2) / (2. * f)
        for i in range(w.size(2)):
            for j in range(w.size(3)):
                w[0, 0, i, j] = \
                    (1 - math.fabs(i / f - c)) * (1 - math.fabs(j / f - c))
        for c in range(1, w.size(0)):
            w[c, 0, :, :] = w[0, 0, :, :]
