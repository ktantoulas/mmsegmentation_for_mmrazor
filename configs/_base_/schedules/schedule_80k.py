# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)
optim_wrapper = dict(type='OptimWrapper', optimizer=optimizer, clip_grad=None)
# learning policy
# param_scheduler = [
#     dict(
#         type='ReduceLROnPlateau',
#         mode='max',  # or 'max' depending on the metric (min for loss, max for accuracy/mIoU)
#         factor=0.1,  # Factor by which the learning rate will be reduced
#         patience=10,  # Number of epochs with no improvement after which learning rate will be reduced
#         threshold=0.01,  # Threshold for measuring improvement
#         cooldown=0,  # Number of epochs to wait before resuming normal operation
#         min_lr=1e-6,  # Minimum learning rate
#         verbose=False
#     )
# ]
param_scheduler = [
    dict(
        type='PolyLR',
        eta_min=1e-4,
        power=0.9,
        begin=0,
        end=80000,
        by_epoch=True)
]
# training schedule for 80k
# train_cfg = dict(type='IterBasedTrainLoop', max_iters=80000, val_interval=1000) 
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs = 150, val_interval = 15)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=True),
    param_scheduler=dict(type='ParamSchedulerHook'),
    # checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=8000),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(draw=True, interval=100, type='SegVisualizationHook'),
    # visualization=dict(type='SegVisualizationHook'),
    checkpoint=dict(
        type='CheckpointHook',
        by_epoch=True,
        interval=10,  
        save_best='mIoU',  
        # rule='greater',  
        max_keep_ckpts=3  
    )
    )

# custom_hooks = [
#     dict(
#         type='EarlyStoppingHook',
#         patience=6,             
#         monitor='mIoU',          
#         rule='greater',          
#         min_delta=0.01           
#     )
# ]
