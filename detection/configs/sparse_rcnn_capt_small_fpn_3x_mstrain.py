_base_ = [
    '_base_/models/sparse_rcnn_r50_fpn_1x_coco.py'
]

model = dict(
    pretrained='/pub/data/ligang/projects/PVT/checkpoints/capt_small/capt_small_checkpoint.pth',
    backbone=dict(
        type='capt_small',
        style='pytorch'),
    neck=dict(
        type='FPN',
        in_channels=[64, 128, 320, 512],
        out_channels=256,
        start_level=0,
        add_extra_convs='on_input',
        num_outs=4))

# model = dict(
#     # pretrained='pretrained/pvt_v2_b1.pth',
#     pretrained='https://github.com/whai362/PVT/releases/download/v2/pvt_v2_b1.pth',
#     backbone=dict(
#         type='capt_tiny',
#         style='pytorch'),
#     neck=dict(
#         type='FPN',
#         in_channels=[64, 128, 320, 512],
#         out_channels=256,
#         start_level=1,
#         add_extra_convs='on_input',
#         num_outs=5))

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
min_values = (480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='Resize',
        img_scale=[(1333, value) for value in min_values],
        multiscale_mode='value',
        keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
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
runner = dict(type='EpochBasedRunner', max_epochs=36)

# use apex fp16
# runner = dict(type='EpochBasedRunnerAmp', max_epochs=36)
# fp16 = None
# optimizer_config = dict(
#     type="DistOptimizerHook",
#     update_interval=1,
#     grad_clip=None,
#     coalesce=True,
#     bucket_size_mb=-1,
#     use_fp16=True,
# )
#
find_unused_parameters = True