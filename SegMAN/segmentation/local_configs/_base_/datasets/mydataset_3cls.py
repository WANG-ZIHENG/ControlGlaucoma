dataset_type = 'CustomDataset'
data_root = 'data/segman'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True)

crop_size = (512, 512)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(2048, 512), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(2048, 512),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),  
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='train/images',
        ann_dir='train/masks_fixed',
        pipeline=train_pipeline,
        img_suffix='.png',
        seg_map_suffix='.png',
        classes=('black', 'gray', 'white'),
        palette=[[0, 0, 0], [128, 128, 128], [255, 255, 255]]
    ),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='val/images',
        ann_dir='val/masks_fixed',
        pipeline=test_pipeline,
        img_suffix='.png',
        seg_map_suffix='.png',
        classes=('black', 'gray', 'white'),
        palette=[[0, 0, 0], [128, 128, 128], [255, 255, 255]]
    ),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        # img_dir=<DATA_ROOT>/test_slo_fundus,
        #只需要改这里的路径，任何图片都可以
       # ann_dir='test/masks_fixed',
        # img_dir=<DATA_ROOT>/test_slo_fundus,
        # ann_dir='test_race/asian/masks_fixed',
        img_dir='test/images',
        ann_dir='test/masks_fixed',
        pipeline=test_pipeline,
        img_suffix='.png',
        seg_map_suffix='.png',
        classes=('black', 'gray', 'white'),
        palette=[[0, 0, 0], [128, 128, 128], [255, 255, 255]]
    )
)
