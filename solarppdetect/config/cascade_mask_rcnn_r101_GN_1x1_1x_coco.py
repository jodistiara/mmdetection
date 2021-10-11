_base_ = '../../configs/cascade_rcnn/cascade_mask_rcnn_r50_fpn_1x_coco.py'
model = dict(
    backbone=dict(
        depth=101,
        frozen_stages = -1,
        zero_init_residual = False,
        init_cfg = None),
    roi_head=dict(
        bbox_head=[
            dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=1,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.1, 0.1, 0.2, 0.2]),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0,
                               loss_weight=1.0)),
            dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=1,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.05, 0.05, 0.1, 0.1]),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0,
                               loss_weight=1.0)),
            dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=1,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.033, 0.033, 0.067, 0.067]),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0))
        ],
        mask_head=dict(num_classes=1)
    ))

dataset_type = 'CocoDataset'
data_root = '../data/'
classes = ('module',)


data = dict(
    samples_per_gpu=1,
    workers_per_gpu=1,
    train=dict(
        type='CocoDataset',
        classes=('module',),
        ann_file='../data/test/via_project_25Mar2021_20h43m_coco.json',
        img_prefix='../data/train2/'),
    val=dict(
        type='CocoDataset',
        # explicitly add your class names to the field `classes`
        classes=('module',),
        ann_file='../data/test/via_project_16Mar2021_10h16m_coco.json',
        img_prefix='../data/test/'),
    test=dict(
        type='CocoDataset',
        # explicitly add your class names to the field `classes`
        classes=('module',),
        ann_file='solarppdetect/data/test/via_project_16Mar2021_10h16m_coco.json',
        img_prefix='solarppdetect/data/test/')
)