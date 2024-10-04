default_scope = 'mmseg'
env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0) ,
    dist_cfg=dict(backend='nccl'),
)
# vis_backends = [dict(type='LocalVisBackend')]
vis_backends=[
    dict(
        type='WandbVisBackend',
        init_kwargs={
            'project': 'mmsegmentation',
            'entity': 'iulia-e-teodorescu-university-jonkoping',
            'name' : 'stdc1_seg'
        },
    ),dict(type='LocalVisBackend')
]

visualizer = dict(
    type='SegLocalVisualizer', vis_backends=vis_backends, name='visualizer')

log_processor = dict(by_epoch=True)
log_level = 'INFO'
load_from = None
resume = False

tta_model = dict(type='SegTTAModel')
