_base_ = ['./yoso_r50_50e_coco-panoptic.py']

custom_imports = dict(
    imports=['yoso.modeling', 
             'yoso.layers',
             'yoso.modeling.backbone'],
    allow_failed_imports=False)

model = dict(
    backbone=dict(
        _delete_=True,
        type='repvit_m1_5',
        init_cfg=dict(
            type='Pretrained',
            checkpoint='pretrain/repvit_m1_5_distill_450e.pth',
        ),
        out_indices=[4, 10, 36, 42]
    ),
    neck=dict(in_channels=[64, 128, 256, 512]))

# set all layers in backbone to lr_mult=0.1
# set all backbone norm layers to decay_multi=0.0
backbone_norm_multi = dict(lr_mult=0.1, decay_mult=0.0)
custom_keys = {
    'backbone': dict(lr_mult=0.1, decay_mult=1.0),
       
    # RepViT Stem Embedding BN
    'features.0.0.bn': backbone_norm_multi,
    'features.0.2.bn': backbone_norm_multi,
    
    # RepViT Token Mixer BN
    'token_mixer.0.conv.bn': backbone_norm_multi, 
    'token_mixer.0.bn': backbone_norm_multi,      
    'token_mixer.2.bn': backbone_norm_multi,      
    
    # RepViT Channel Mixer BN
    'channel_mixer.m.0.bn': backbone_norm_multi,
    'channel_mixer.m.2.bn': backbone_norm_multi,
}
# optimizer
optim_wrapper = dict(
    paramwise_cfg=dict(custom_keys=custom_keys, norm_decay_mult=0.0))