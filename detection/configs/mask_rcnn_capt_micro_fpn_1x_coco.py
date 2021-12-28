_base_ = [
    '_base_/models/mask_rcnn_r50_fpn.py',
    '_base_/datasets/coco_instance.py',
    '_base_/schedules/schedule_1x.py',
    '_base_/default_runtime.py'
]
# optimizer
model = dict(
    # pretrained='pretrained/pvt_v2_b0.pth',
    # pretrained='https://github.com/whai362/PVT/releases/download/v2/pvt_v2_b0.pth',
    pretrained='/pub/data/ligang/projects/PVT/checkpoints/capt_micro/capt_micro_checkpoint.pth',
    backbone=dict(
        type='capt_micro',
        style='pytorch'),
    neck=dict(
        type='FPN',
        in_channels=[32, 64, 160, 256],
        out_channels=256,
        num_outs=5))
# optimizer
optimizer = dict(_delete_=True, type='AdamW', lr=0.0002, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
