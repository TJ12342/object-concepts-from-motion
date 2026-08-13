# optimizer
optimizer = dict(
    type='AdamW',
    lr=4e-4,
    paramwise_cfg=dict(
        custom_keys=dict(
            img_backbone=dict(lr_mult=0.25),
            absolute_pos_embed=dict(decay_mult=0.),
            relative_position_bias_table=dict(decay_mult=0.),
            norm=dict(decay_mult=0.),
        )),
    weight_decay=0.01)
optimizer_config = dict(grad_clip=dict(max_norm=3, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=2000,
    warmup_ratio=1.0 / 3,
    step=[20, ])
total_epochs = 24
runner = dict(type='EpochBasedRunner', max_epochs=total_epochs)
