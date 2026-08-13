_base_ = [
    './_base_/default_runtime.py',
    './_base_/bevformerv2_swin.py',
    './_base_/schedule_24ep.py',
]

model = dict(
    img_backbone=dict(
        arch='tiny',
        drop_path_rate=0.2,
        init_cfg=dict(
            type='Pretrained',
            checkpoint='../../checkpoints/swin_t.pth',
            prefix='backbone.',
        ),
    ),
    img_neck=dict(in_channels=[192, 384, 768]),
)
