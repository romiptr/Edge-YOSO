_base_ = ['./yoso_r50_50e_coco-panoptic.py']

custom_imports = dict(
    imports=['yoso.modeling', 
             'yoso.layers',
             'yoso.modeling.backbone'],
    allow_failed_imports=False)

model = dict(
    backbone=dict(
        _delete_=True,
        type='hgnetv2_b2',
        freeze_at=-1,
        freeze_norm=True,
        use_lab=True,
        init_cfg=dict(
            type='Pretrained',
            checkpoint='pretrain/PPHGNetV2_B2_stage1.pth',
        )
    ),
    neck=dict(in_channels=[96, 384, 768, 1536]))
