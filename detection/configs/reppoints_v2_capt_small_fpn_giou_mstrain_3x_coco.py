_base_ = './_base_/models/reppoints_v2_r50_fpn_giou_1x_coco.py'

#lr_config = dict(step=[16, 22])
#total_epochs = 36

model = dict(
    # pretrained='pretrained/pvt_v2_b0.pth',
    # pretrained='https://github.com/whai362/PVT/releases/download/v2/pvt_v2_b0.pth',
    pretrained='/pub/data/ligang/projects/CAPT/checkpoints/capt_small/capt_small_checkpoint.pth',
    backbone=dict(
        type='capt_small',
        style='pytorch'),
    neck=dict(
        type='FPN',
        in_channels=[64, 128, 320, 512],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_input',
        num_outs=5))

# multi-scale training
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='AutoAugment',
         policies=[[
             dict(type='Resize',
                  img_scale=[(480, 1333), (512, 1333), (544, 1333), (576, 1333),
                             (608, 1333), (640, 1333), (672, 1333), (704, 1333),
                             (736, 1333), (768, 1333), (800, 1333)],
                  multiscale_mode='value',
                  keep_ratio=True)],
             [
                 dict(type='Resize',
                      img_scale=[(400, 1333), (500, 1333), (600, 1333)],
                      multiscale_mode='value',
                      keep_ratio=True),
                 dict(type='RandomCrop',
                      crop_type='absolute_range',
                      crop_size=(384, 600),
                      allow_negative_crop=True),
                 dict(type='Resize',
                      img_scale=[(480, 1333), (512, 1333), (544, 1333),
                                 (576, 1333), (608, 1333), (640, 1333),
                                 (672, 1333), (704, 1333), (736, 1333),
                                 (768, 1333), (800, 1333)],
                      multiscale_mode='value',
                      override=True,
                      keep_ratio=True)]
         ]),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]

data = dict(train=dict(pipeline=train_pipeline))

optimizer = dict(_delete_=True, type='AdamW', lr=0.0001, betas=(0.9, 0.999), weight_decay=0.05,
                 paramwise_cfg=dict(custom_keys={'absolute_pos_embed': dict(decay_mult=0.),
                                                 'relative_position_bias_table': dict(decay_mult=0.),
                                                 'norm': dict(decay_mult=0.)}))
lr_config = dict(step=[27, 33])

# do not use apex fp16
# runner = dict(type='EpochBasedRunner', max_epochs=36)

# use apex fp16
runner = dict(type='EpochBasedRunnerAmp', max_epochs=36)
fp16 = None
optimizer_config = dict(
    type="DistOptimizerHook",
    update_interval=1,
    grad_clip=None,
    coalesce=True,
    bucket_size_mb=-1,
    use_fp16=True,
)

find_unused_parameters = True