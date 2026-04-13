from typing import List, Tuple, Union

import torch
from torch import nn
import torch.nn.functional as F

from torch import Tensor
from mmengine.model import BaseModule
from mmdet.registry import MODELS
from mmdet.utils import ConfigType, OptMultiConfig
from ..layers import DeformLayer
from ..utils.misc import generate_coordinate


@MODELS.register_module()
class YOSONeck(BaseModule):
    """Implementation of YOSO neck (Feature Pyramid Aggregator).

    Args:
        in_channels (list[int]): Number of channels in the input feature maps.
        strides (list[int] | tuple[int]): Output strides of feature from backbone.
        aggregate_channels (int): Number of channels for aggregated feature map.
        out_channels (int): Number of channels for output.
        agg_method (str): Whether to use Convolution-First ('cfa') or Interpolation-First Aggregation ('ifa')\
            method to merge multi-level feature maps.
            Defaults to 'cfa'.
        deform_layer (:obj:`ConfigDict` or dict): Config for deformable layer. 
            Defaults to None.
        norm_cfg (:obj:`ConfigDict` or dict): Config for normalization.
            Defaults to dict(type='BN').
        init_cfg (:obj:`ConfigDict` or dict or list[:obj:`ConfigDict` or \
            dict], optional): Initialization config dict.
            Defaults to dict(type='Xavier', layer='Conv2d', distribution='uniform').
    """
    def __init__(self,
                 in_channels: Union[List[int],
                                    Tuple[int]] = [256, 512, 1024, 2048],
                 strides: Union[List[int], Tuple[int]] = [4, 8, 16, 32],
                 aggregate_channels: int = 128,
                 out_channels: int = 256,
                 agg_method: str = 'cfa',
                 deform_layer: ConfigType = None,
                 norm_cfg: ConfigType = dict(type='BN'),
                 init_cfg: OptMultiConfig = dict(type='Xavier', layer='Conv2d', distribution='uniform')
                 ) -> None:
        super().__init__(init_cfg=init_cfg)
        self.strides = strides
        self.agg_method = agg_method

        assert agg_method in ['cfa', 'ifa']

        # Lateral conv: feature maps from backbone is compressed with with 1x1 conv
        self.lateral_conv = nn.ModuleList([
            nn.Conv2d(in_channels=ch, out_channels=ch//2, kernel_size=1, stride=1, padding=0) 
            for ch in in_channels[::-1]])
        # Deform conv: adaptive receptive field
        self.deform_conv = nn.ModuleList([
            DeformLayer(in_channels=in_channels[i]//2, out_channels=in_channels[i-1]//2, 
                        norm_cfg=norm_cfg, **deform_layer) 
            for i in range(len(in_channels)-1, 0,-1)]) #Pn: Pn+1 + Cn

        # Aggregation
        if self.agg_method == 'cfa':
            self.bias = nn.Parameter(torch.zeros(1, aggregate_channels, 1, 1))

            self.conv_a = nn.ModuleList([
                nn.Conv2d(in_channels=ch//2, out_channels=aggregate_channels, kernel_size=1, stride=1, padding=0, bias=False) 
                for ch in in_channels[::-1]])
        else:
            self.fuse_conv = nn.Conv2d(in_channels=sum([ch//2 for ch in in_channels]), out_channels=aggregate_channels, kernel_size=1, stride=1, padding=0)
        
        self.output_conv = nn.Conv2d(in_channels=aggregate_channels, out_channels=aggregate_channels, kernel_size=3, stride=1, padding=1) # Outputs one single aggregated feature map
        
        # Fuse feature with spatial coordinate (x,y) from CoordConv 
        self.loc_conv = nn.Conv2d(in_channels=aggregate_channels + 2, 
                                  out_channels=out_channels, 
                                  kernel_size=1) 

    def forward(self, feats: List[Tensor]) -> Tensor:
        """Forward function.
        Args:
            feats (list[Tensor]): : Feature maps of each level. Each has
                shape of (batch_size, c, h, w).

        Returns:
            Tensor: Single-level aggregated feature map, 
                with shape (batch_size, c, h, w).
        """
        x = self.lateral_conv[0](feats[-1])
        out_feats = [x]

        for i in range(len(self.deform_conv)):
            x_up = self.deform_conv[i](x)
            x_lat = self.lateral_conv[i+1](feats[-(i+2)])
            x = x_lat + x_up
            out_feats.append(x)

        aligned_feats = []
        for i in range(len(out_feats)):
            feat = out_feats[i]
            scale = self.strides[-(i+1)] // 4

            if self.agg_method == 'cfa':
                feat = self.conv_a[i](feat)

            if scale > 1:
                feat = F.interpolate(feat, scale_factor=scale, align_corners=False, mode='bilinear')

            aligned_feats.append(feat)

        if self.agg_method == 'cfa':
            x = sum(aligned_feats) + self.bias
        else:
            x_fuse = torch.concat(aligned_feats, dim=1)
            x = self.fuse_conv(x_fuse)

        agg_feat = self.output_conv(x)   
        coord_feat = generate_coordinate(agg_feat)
        out_feat = torch.cat([agg_feat, coord_feat], 1)
        out_feat = self.loc_conv(out_feat)

        return out_feat