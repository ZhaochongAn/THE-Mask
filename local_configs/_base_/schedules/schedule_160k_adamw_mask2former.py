# optimizer
backbone_multiplier = 0.1 # cfg.SOLVER.BACKBONE_MULTIPLIER

optimizer = dict(
    type='AdamW', lr=0.0001, weight_decay=0.05, 
    paramwise_cfg = dict(
        custom_keys={
            'backbone': dict(lr_mult=backbone_multiplier),
            'norm': dict(decay_mult=0.)
            }
        )
    )

optimizer_config = dict(grad_clip=dict(max_norm=0.01, norm_type=2))

# learning policy
lr_config = dict(policy='poly', power=0.9, min_lr=0.0, by_epoch=False, warmup_iters=0)


# runtime settings
runner = dict(type='IterBasedRunner', max_iters=160000)
checkpoint_config = dict(by_epoch=False, interval=4000)
# evaluation = dict(interval=4000, metric='mIoU', efficient_test=True, save_best='mIoU')
evaluation = dict(interval=4000, metric='mIoU')
