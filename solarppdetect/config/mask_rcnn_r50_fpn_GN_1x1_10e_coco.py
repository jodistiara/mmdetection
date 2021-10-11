_base_ = [
    '../../configs/_base_/models/mask_rcnn_r50_fpn.py',
    '../../configs/_base_/datasets/coco_instance.py',
    '../../configs/_base_/schedules/schedule_1x.py', 
    '../../configs/_base_/default_runtime.py'
]
norm_cfg = dict(type='GN', num_groups=32, requires_grad=True)
model = dict(
    backbone=dict(
        frozen_stages=-1,
        zero_init_residual=False,
        norm_cfg=norm_cfg,
        init_cfg=None),
    neck=dict(norm_cfg=norm_cfg),
    roi_head=dict(
        bbox_head=dict(
            type='Shared4Conv1FCBBoxHead',
            conv_out_channels=256,
            norm_cfg=norm_cfg,
            num_classes=1),
        mask_head=dict(norm_cfg=norm_cfg,
                       num_classes=1)))
# evaluation
evaluation = dict(interval=10)
# optimizer
optimizer = dict(paramwise_cfg=dict(norm_decay_mult=0),
                 lr = 0.02 / 8)
optimizer_config = dict(_delete_=True, grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(warmup_ratio=0.1, step=20)
runner = dict(type='EpochBasedRunner', max_epochs=100)

checkpoint_config = dict(interval=20)

# runtime log
log_config = dict(
    interval=58
)
dataset_type = 'CocoDataset'
data_root = '../data/'
classes = ('module',)


data = dict(
    samples_per_gpu=1,
    workers_per_gpu=1,
    train=dict(
        type='CocoDataset',
        classes=('module',),
        ann_file='solarppdetect/data/train2/via_project_25Mar2021_20h43m_coco.json',
        img_prefix='solarppdetect/data/train2/'),
    val=dict(
        type='CocoDataset',
        # explicitly add your class names to the field `classes`
        classes=('module',),
        ann_file='solarppdetect/data/test/via_project_10Mar2021_11h10m_coco.json',
        img_prefix='solarppdetect/data/test/'),
    test=dict(
        type='CocoDataset',
        # explicitly add your class names to the field `classes`
        classes=('module',),
        ann_file='solarppdetect/data/test/via_project_16Mar2021_10h16m_coco.json',
        img_prefix='solarppdetect/data/test/')
)