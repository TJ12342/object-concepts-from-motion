_base_ = [
    './_base_/default_runtime.py',
    './_base_/bevformerv2_swin.py',
    './_base_/schedule_24ep.py',
]

model = dict(
    img_backbone=dict(
        arch='huge',
        drop_path_rate=0.5,
        with_cp=True,
        init_cfg=dict(
            type='Pretrained',
            checkpoint='../../checkpoints/swin_h.pth',
            prefix='backbone.',
        ),
    ),
    img_neck=dict(in_channels=[768, 1536, 3072]),
)
