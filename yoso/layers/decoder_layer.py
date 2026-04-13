from typing import List, Tuple, Union

import torch
from torch import nn
import torch.nn.functional as F

from torch import Tensor
from mmcv.cnn import build_norm_layer
from mmcv.cnn.bricks.transformer import FFN, MultiheadAttention
from mmengine.model import BaseModule
from mmdet.utils import ConfigType, OptMultiConfig


class YOSODecoderLayer(BaseModule):
    """Implements decoder layer in YOSO head (Separable Dynamic Decoder).

    Args:
        in_channels (int): Number of channels in the input feature map.
        num_classes (int): Number of total things and stuff. 
        num_proposals (int): Number of proposal kernels.
        num_cls_fcs (int): Number of fully connected layers for classification prediction.
            Defaults to 1.
        num_mask_fcs (int): Number of fully connected layers for mask prediction.
            Defaults to 1.
        num_c_attn_blocks (int): Number of cross attention blocks.
            Defaults to 2.
        dy_conv1d_kernel_size (int): Kernel size for dynamic convoltion attention (DySepConvAtten).
            Defaults to 3.
        self_attn_cfg (:obj:`ConfigDict` or dict): Config for MultiHeadAttention. 
            Defaults to None.
        ffn_cfg (:obj:`ConfigDict` or dict): Config for FFN. 
            Defaults to None.
        norm_cfg (:obj:`ConfigDict` or dict): Config for normalization.
            Defaults to dict(type='LN').
        init_cfg (:obj:`ConfigDict` or dict or list[:obj:`ConfigDict` or \
            dict], optional): Initialization config dict. 
            Defaults to [ dict(type='TruncNormal', layer='Linear', std=0.02, bias=0.0),
                          dict(type='Constant', layer='LayerNorm', val=1.0, bias=0.0)].
    """
    def __init__(self,
                 in_channels: int,
                 num_classes: int,
                 num_proposals: int,
                 num_cls_fcs: int =  1,
                 num_mask_fcs: int =  1,
                 num_c_attn_blocks: int = 2, 
                 dy_conv1d_kernel_size: int = 3,
                 self_attn_cfg: ConfigType = None,
                 ffn_cfg: ConfigType = None,
                 norm_cfg: ConfigType = dict(type='LN'),
                 init_cfg: OptMultiConfig = [
                    dict(type='TruncNormal', layer='Linear', std=0.02, bias=0.0),
                    dict(type='Constant', layer='LayerNorm', val=1.0, bias=0.0)
                ]) -> None:
        super().__init__(init_cfg=init_cfg)
       
        self.in_channels = in_channels
        self.num_proposals = num_proposals
        self.num_c_attn_blocks = num_c_attn_blocks
        self.hard_mask_thr = 0.5

        self.c_atten = nn.ModuleList()
        self.c_dropout = nn.ModuleList()
        self.c_atten_norm = nn.ModuleList()

        for _ in range(self.num_c_attn_blocks):
            self.c_atten.append(DySepConvAtten(in_channels=self.in_channels, 
                                               num_proposals=self.num_proposals, 
                                               kernel_size=dy_conv1d_kernel_size,
                                               norm_cfg=norm_cfg))
            self.c_dropout.append(nn.Dropout(0.0))
            self.c_atten_norm.append(build_norm_layer(norm_cfg,self.in_channels)[1])

        self.s_atten = MultiheadAttention(**self_attn_cfg)
        self.s_dropout = nn.Dropout(0.0)
        self.s_atten_norm = build_norm_layer(norm_cfg,self.in_channels)[1]

        self.ffn = FFN(**ffn_cfg)
        self.ffn_norm = build_norm_layer(norm_cfg,self.in_channels)[1]
        
        cls_layers = []
        for _ in range(num_cls_fcs):
            cls_layers.append(nn.Linear(self.in_channels, self.in_channels, bias=False))
            cls_layers.append(build_norm_layer(norm_cfg, self.in_channels)[1])
            cls_layers.append(nn.ReLU(True))
        self.cls_fcs = nn.Sequential(*cls_layers)
        self.fc_cls = nn.Linear(self.in_channels, num_classes + 1)
        
        mask_layers = []
        for _ in range(num_mask_fcs):
            mask_layers.append(nn.Linear(self.in_channels, self.in_channels, bias=False))
            mask_layers.append(build_norm_layer(norm_cfg, self.in_channels)[1])
            mask_layers.append(nn.ReLU(True))
        self.mask_fcs = nn.Sequential(*mask_layers)
        self.fc_mask = nn.Linear(self.in_channels, self.in_channels)

    def forward(self, feat: Tensor, 
                proposal_kernels: Tensor, 
                mask_preds: Tensor, 
                calc_cls_score: bool) -> Tuple[Tensor]:
        """Forward function.

        Args:
            feat (Tensor): Single-level aggregated feature map from neck,
                with shape (batch_size, c, h, w).
            proposal_kernels (Tensor): Learnable proposal kernels for masks generation,\
                the kernels are updated each decoder stages,
                with shape (batch_size, num_proposals, c, k, k).
            mask_preds (Tensor): Mask prediction from last decoder stage,\
                obtained through 2D Convolution between aggregated feature map\
                and learnable proposal kernels,\
                with shape (batch_size, num_proposals, h, w).
            calc_cls_score (bool): Whether to return the classification score.

        Returns:
            tuple[Tensor]: a tuple containing three elements.

                - cls_score (Tensor): Classification score for\
                    current decoder decoder stage. The shape is\
                    (batch_size, num_proposals, cls_out_channels).\
                    Note `cls_out_channels` should includes background.
                - new_mask_preds (Tensor): Mask prediction for current\
                    decoder stage. The shape is (batch_size, num_proposals, h, w).
                - new_panoptic_kernels (Tensor): Updated learnable proposal kernels\
                    The shape is (batch_size, num_proposals, c, k, k).
        """
        B, C, H, W = feat.shape

        # == Pre-Attention ==
        soft_sigmoid_masks = mask_preds.sigmoid()
        nonzero_inds = soft_sigmoid_masks > self.hard_mask_thr
        hard_sigmoid_masks = nonzero_inds.float()

        # Masked Feature (V) = r(A)r(S).T
        # [B, N, H, W] @ [B, C, H, W] -> [B, N, C]
        V = torch.einsum('bnhw,bchw->bnc', hard_sigmoid_masks, feat)

        # Learnable Proposal Kernel (Q)
        # [B, N, C, K, K] -> [B, N, C * K * K] with K=1 -> [B,N,C]
        Q = proposal_kernels.view(B, self.num_proposals, -1)

        # == SDCA (Cross-Attention) ==
        for i in range(self.num_c_attn_blocks):
            c_out = self.c_atten[i](Q,V)
            V = V + self.c_dropout[i](c_out)
            V = self.c_atten_norm[i](V)
        
        # == Post-Attention ==
        k = V.permute(1,0,2) # [N, B, C]
        # MHSA (Self-attention)
        k_tmp = self.s_atten(query = k, key = k, value = k )[0]
        k = k + self.s_dropout(k_tmp)
        k = self.s_atten_norm(k.permute(1, 0, 2))

        # FFN Masks & Class 
        # [B, N, C * K * K] -> [B, N, C, K * K] -> [B, N, K * K, C]
        obj_feat = k.reshape(B, self.num_proposals, self.in_channels, -1).permute(0, 1, 3, 2)
        obj_feat = self.ffn_norm(self.ffn(obj_feat))

        cls_feat = obj_feat.sum(-2)
        mask_feat = obj_feat 

        if calc_cls_score:
            cls_feat = self.cls_fcs(cls_feat)
            # [B, N, num_classes + 1]
            cls_score = self.fc_cls(cls_feat).view(B, self.num_proposals, -1)
        else:
            cls_score = None

        mask_feat = self.mask_fcs(mask_feat)

        # [B, N, K * K, C] -> [B, N, C]
        mask_kernels = self.fc_mask(mask_feat).squeeze(2)
        new_mask_preds = torch.einsum("bqc,bchw->bqhw", mask_kernels, feat)
        #torch.bmm(mask_kernels, feat.view(B, C, H * W)).view(B, self.num_proposals, H, W)

        #  [B, N, K * K, C] -> [B, N, C, K, K]
        new_panoptic_kernels = obj_feat.permute(0, 1, 3, 2).reshape(B, self.num_proposals, self.in_channels, 1, 1)
        
        return cls_score, new_mask_preds, new_panoptic_kernels


class DySepConvAtten(BaseModule):
    """Cross Attention with 1D Convolution in a weight sharing manner.

    DySepConvAtten(Q,V) = (V*r(QWd)) * r(QWp)

    > Q (Proposal kernels) query is projected to Wd (Depthwise projection matrix)\
        and Wp (Pointwise projection matrix) to create kernels for convolution.\
    > V (Masked feature) value is convoluted with QWd\
        (depthwise convolution) for cross-dimension interaction.\
        After that, the output is convoluted with QWp\
        (pointwise convolution) for cross-token interaction.\

    Args:
        in_channels (int): Number of channels in the input features.
        num_proposals (int): Number of proposal kernels.
        kernel_size (int): Kernel size for 1D Convolution.
            Defaults to 3.
        norm_cfg (:obj:`ConfigDict` or dict): Config for normalization.
            Defaults to dict(type='LN').
    """
    def __init__(self,
                 in_channels: int,
                 num_proposals: int,
                 kernel_size: int = 3,
                 norm_cfg: ConfigType = dict(type='LN')):
        super().__init__()
        self.in_channels = in_channels
        self.num_proposals = num_proposals
        self.kernel_size = kernel_size

        self.padding = self.kernel_size // 2

        self.weight_linear = nn.Linear(self.in_channels, self.num_proposals + self.kernel_size)
        self.norm = build_norm_layer(norm_cfg,self.in_channels)[1]
    
    def forward(self, query: Tensor, 
                value: Tensor) -> Tensor:
        """Forward function.

        Args:
            query (Tensor): Learnable proposal kernel (Q) projected that will be
                linear projected to create dynamic convolution weight,\
                with shape (batch_size, num_proposals, c).
            value (Tensor): Masked feature (V)  to be convoluted,\
                with shape (batch_size, num_proposals, c).

        Returns:
            - out (Tensor): Cross-Attention output\
                with shape (batch_size, num_proposals, c).
        """
        assert query.shape == value.shape
        B, N, C = query.shape
        
        dy_conv_weight = self.weight_linear(query)
        
        # Split projected weights into Depthwise and Pointwise kernels
        dy_depth_conv_weight = dy_conv_weight[:, :, :self.kernel_size].view(B,self.num_proposals,1,self.kernel_size)
        dy_point_conv_weight = dy_conv_weight[:, :, self.kernel_size:].view(B,self.num_proposals,self.num_proposals,1)

        res = []
        value = value.unsqueeze(1)
        for i in range(B):
            # input: [1, N, C]
            # weight: [N, 1, K]
            # output: [1, N, C]
            out = F.relu(F.conv1d(input=value[i], weight=dy_depth_conv_weight[i], groups=N, padding="same"))
            # input: [1, N, C]
            # weight: [N, N, 1]
            # output: [1, N, C]
            out = F.conv1d(input=out, weight=dy_point_conv_weight[i], padding='same')

            res.append(out)
        out = torch.cat(res, dim=0)
        out = self.norm(out)

        return out