_base_ = ['./yoso_r50_50e_coco-panoptic.py']

train_cfg = dict(
    _delete_=True,
    type='EpochBasedTrainLoop', 
    max_epochs=12, 
    val_interval=1)

param_scheduler = dict(
    _delete_=True,
    type='MultiStepLR',
    begin=0,
    end=12,
    by_epoch=True,
    milestones=[8, 11],
    gamma=0.1)

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        _delete_=True,
        by_epoch=True,
        save_last=True,
        max_keep_ckpts=3,
        interval=1))
log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)
