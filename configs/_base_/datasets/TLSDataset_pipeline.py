 # 数据处理 pipeline
# 同济子豪兄 2023-6-28

# 数据集路径
dataset_type = 'TLSDataset' # 数据集类名
data_root = '/lustre1/g/path_dwhho/new_LMH/TLS_segmentation/188_slides_tiles/merge_TLS_and_Immunecluster_2048' # 数据集路径（相对于mmsegmentation主目录）#****

# 输入模型的图像裁剪尺寸，一般是 128 的倍数，越小显存开销越少
crop_size = (512, 512)

# 训练预处理(主要是数据增强data augmentation)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(
        type='RandomResize',
        scale=(2048, 1024),
        ratio_range=(0.8, 1.5),
        keep_ratio=True), #数据缩放
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.5), #数据裁剪
    dict(type='RandomFlip', prob=0.5), #数据翻转
    dict(type='PhotoMetricDistortion'),#数据增强
    dict(type='PackSegInputs')
]

# 测试预处理
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(2048, 1024), keep_ratio=True),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs')
]

# TTA(Test time Augmentation)后处理
img_ratios = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
tta_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=dict(backend='disk')),
    dict(
        type='TestTimeAug',
        transforms=[
            [
                dict(type='Resize', scale_factor=r, keep_ratio=True)
                for r in img_ratios
            ],
            [
                dict(type='RandomFlip', prob=0., direction='horizontal'),
                dict(type='RandomFlip', prob=1., direction='horizontal')
            ], [dict(type='LoadAnnotations')], [dict(type='PackSegInputs')]
        ])
]



train_dataloader = dict(
    batch_size=2,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type='ConcatDataset',
        datasets=[
            # 1) 全体训练集（原来的 train）
            dict(
                type=dataset_type,
                data_root=data_root,
                data_prefix=dict(
                    img_path='train/images',
                    seg_map_path='train/masks'),
                pipeline=train_pipeline
            ),

            # 2) 只含 GC 的子集，重复若干次做过采样
            dict(
                type=dataset_type,
                data_root=data_root,
                data_prefix=dict(
                    img_path='train_gc/images',
                    seg_map_path='train_gc/masks'),
                pipeline=train_pipeline
            ),
            dict(
                type=dataset_type,
                data_root=data_root,
                data_prefix=dict(
                    img_path='train_gc/images',
                    seg_map_path='train_gc/masks'),
                pipeline=train_pipeline
            ),
            dict(
                type=dataset_type,
                data_root=data_root,
                data_prefix=dict(
                    img_path='train_gc/images',
                    seg_map_path='train_gc/masks'),
                pipeline=train_pipeline
            ),
            # ↑ 想再加强 GC 就再复制几段 dict
        ]
    )
)


# 测试 Dataloader
test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='test/images', seg_map_path='test/masks'),#****
        pipeline=test_pipeline))

# 验证 Dataloader
val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='val/images', seg_map_path='val/masks'),#****
        pipeline=test_pipeline))

# 测试 Evaluator
test_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU', 'mDice', 'mFscore'])

# 验证 Evaluator
val_evaluator = test_evaluator
