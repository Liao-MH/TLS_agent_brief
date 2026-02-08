import os

DATA_ROOT = os.getenv(
    "DATA_ROOT",
    "/lustre1/g/path_dwhho/new_LMH/TLS_segmentation/188_slides_tiles/merge_TLS_and_Immunecluster_2048",
)
OUTPUT_ROOT = os.getenv("OUTPUT_ROOT", "/lustre1/g/path_dwhho/new_LMH/mmsegmentation")
CHECKPOINT_ROOT = os.getenv(
    "CHECKPOINT_ROOT",
    "/lustre1/g/path_dwhho/new_LMH/mmsegmentation/work_dirs/morph_ablation_12/C5_All_w010",
)
WORK_DIR = os.getenv(
    "WORK_DIR",
    os.path.join(OUTPUT_ROOT, "work_dirs/morph_ablation_12/C3_H_GradMag_LBP_w050"),
)
PRETRAINED = os.getenv("PRETRAINED", os.path.join(CHECKPOINT_ROOT, "best_mFscore_iter_19000.pth"))

# ============================================================
# C3_H_GradMag_LBP_w050  (EPOCH-BASED, CLEAN & EXPLICIT)
# - Keep critical blocks visible in dataloaders (pipelines expanded)
# - Remove iter-based loop and switch to epoch-based training
# - Keep morph fusion + your original model/loss/settings
# ============================================================

default_scope = 'mmseg'

custom_imports = dict(
    allow_failed_imports=False,
    imports=['projects.morph_inputfusion'],
)

env_cfg = dict(
    cudnn_benchmark=True,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
)

log_level = 'INFO'
randomness = dict(seed=0)
resume = False
load_from = None

# --------------------
# Paths
# --------------------
data_root = DATA_ROOT
work_dir  = WORK_DIR
dataset_type = 'TLSDataset'

pretrained = PRETRAINED

# --------------------
# Epoch-based schedule
# --------------------
max_epochs   = 20   # <<< 你可以改
val_interval = 1    # 每多少个 epoch 做一次 val

log_processor = dict(by_epoch=True)

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        by_epoch=True,
        interval=1,
        max_keep_ckpts=2,
        rule='greater',
        save_best='mFscore',
    ),
    logger=dict(
        type='LoggerHook',
        interval=200,              # 仍按 iter 打印日志
        log_metric_by_epoch=True,  # 按 epoch 汇总 metric
    ),
    morph_weight_warmup=dict(
        type='MorphWeightWarmupHook',
        hold_ratio=0.3,
        ramp_ratio=0.3,
        start_weight=0.0,
        target_weight=0.5,
        log_interval=200,
        verbose=True,
        mode='auto',
    ),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(type='SegVisualizationHook'),
)

train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=max_epochs,
    val_interval=val_interval,
)
val_cfg  = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# poly lr: by_epoch=True
param_scheduler = [
    dict(
        type='PolyLR',
        begin=0,
        end=max_epochs,
        by_epoch=True,
        eta_min=0,
        power=0.9,
    ),
]

# --------------------
# Optimizer
# --------------------
custom_keys = dict({
    'absolute_pos_embed': dict(decay_mult=0.0, lr_mult=0.1),
    'backbone': dict(decay_mult=1.0, lr_mult=0.1),
    'backbone.norm': dict(decay_mult=0.0, lr_mult=0.1),
    'backbone.patch_embed.norm': dict(decay_mult=0.0, lr_mult=0.1),

    'backbone.stages.0.blocks.0.norm': dict(decay_mult=0.0, lr_mult=0.1),
    'backbone.stages.0.blocks.1.norm': dict(decay_mult=0.0, lr_mult=0.1),
    'backbone.stages.0.downsample.norm': dict(decay_mult=0.0, lr_mult=0.1),

    'backbone.stages.1.blocks.0.norm': dict(decay_mult=0.0, lr_mult=0.1),
    'backbone.stages.1.blocks.1.norm': dict(decay_mult=0.0, lr_mult=0.1),
    'backbone.stages.1.downsample.norm': dict(decay_mult=0.0, lr_mult=0.1),

    'backbone.stages.2.blocks.0.norm': dict(decay_mult=0.0, lr_mult=0.1),
    'backbone.stages.2.blocks.1.norm': dict(decay_mult=0.0, lr_mult=0.1),
    'backbone.stages.2.blocks.2.norm': dict(decay_mult=0.0, lr_mult=0.1),
    'backbone.stages.2.blocks.3.norm': dict(decay_mult=0.0, lr_mult=0.1),
    'backbone.stages.2.blocks.4.norm': dict(decay_mult=0.0, lr_mult=0.1),
    'backbone.stages.2.blocks.5.norm': dict(decay_mult=0.0, lr_mult=0.1),
    'backbone.stages.2.blocks.6.norm': dict(decay_mult=0.0, lr_mult=0.1),
    'backbone.stages.2.blocks.7.norm': dict(decay_mult=0.0, lr_mult=0.1),
    'backbone.stages.2.blocks.8.norm': dict(decay_mult=0.0, lr_mult=0.1),
    'backbone.stages.2.blocks.9.norm': dict(decay_mult=0.0, lr_mult=0.1),
    'backbone.stages.2.blocks.10.norm': dict(decay_mult=0.0, lr_mult=0.1),
    'backbone.stages.2.blocks.11.norm': dict(decay_mult=0.0, lr_mult=0.1),
    'backbone.stages.2.blocks.12.norm': dict(decay_mult=0.0, lr_mult=0.1),
    'backbone.stages.2.blocks.13.norm': dict(decay_mult=0.0, lr_mult=0.1),
    'backbone.stages.2.blocks.14.norm': dict(decay_mult=0.0, lr_mult=0.1),
    'backbone.stages.2.blocks.15.norm': dict(decay_mult=0.0, lr_mult=0.1),
    'backbone.stages.2.blocks.16.norm': dict(decay_mult=0.0, lr_mult=0.1),
    'backbone.stages.2.blocks.17.norm': dict(decay_mult=0.0, lr_mult=0.1),
    'backbone.stages.2.downsample.norm': dict(decay_mult=0.0, lr_mult=0.1),

    'backbone.stages.3.blocks.0.norm': dict(decay_mult=0.0, lr_mult=0.1),
    'backbone.stages.3.blocks.1.norm': dict(decay_mult=0.0, lr_mult=0.1),

    'level_embed': dict(decay_mult=0.0, lr_mult=1.0),
    'query_embed': dict(decay_mult=0.0, lr_mult=1.0),
    'query_feat': dict(decay_mult=0.0, lr_mult=1.0),
    'relative_position_bias_table': dict(decay_mult=0.0, lr_mult=0.1),
})

optim_wrapper = dict(
    type='OptimWrapper',
    clip_grad=dict(max_norm=0.01, norm_type=2),
    optimizer=dict(
        type='AdamW',
        lr=1e-06,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=0.05,
    ),
    paramwise_cfg=dict(
        custom_keys=dict(custom_keys),
        norm_decay_mult=0.0,
    ),
)

# --------------------
# Morph preprocessor
# --------------------
data_preprocessor = dict(
    type='MorphSegDataPreProcessor',
    bgr_to_rgb=True,
    morph_enabled=True,
    morph_mean=0.5,
    morph_std=0.5,
    morph_weight=0.5,
    pad_val=0,
    seg_pad_val=255,
    size=(1200, 1200),
    rgb_mean=[123.675, 116.28, 103.53],
    rgb_std=[58.395, 57.12, 57.375],
)

# Morph transform args (shared)
morph_transform = dict(
    type='LoadMorphologyAndConcat',
    enabled=True,
    source='compute',
    channel_names=['H','GradMag','LBP','Gabor_S1','Gabor_S2'],
    active_channel_names=['H','GradMag','LBP'],
    expected_num_channels=5,
    strict=True,
    h_lo=0.0, h_hi=1.5,
    sobel_ksize=3,
    lbp_enabled=True,
    gabor_gamma=0.5,
    gabor_sigma_1=2.0, gabor_lambda_1=6.0,
    gabor_sigma_2=4.0, gabor_lambda_2=12.0,
    clip=(0.0, 1.0),
)

# --------------------
# Model (keep your original)
# --------------------
model = dict(
    type='EncoderDecoder',
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type='MorphSwinTransformer',
        embed_dims=192,
        depths=[2, 2, 18, 2],
        num_heads=[6, 12, 24, 48],
        window_size=12,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.3,
        patch_norm=True,
        out_indices=(0, 1, 2, 3),
        frozen_stages=-1,
        pretrain_img_size=384,
        with_cp=False,
        morph=dict(enabled=True, num_channels=5, project_to_rgb=True),
        init_cfg=dict(type='Pretrained', checkpoint=pretrained),
    ),
    decode_head=dict(
        type='Mask2FormerHead',
        in_channels=[192, 384, 768, 1536],
        feat_channels=256,
        out_channels=256,
        num_classes=3,
        num_queries=100,
        num_transformer_feat_level=3,
        enforce_decoder_input_project=False,
        align_corners=False,
        loss_cls=dict(
            type='mmdet.CrossEntropyLoss',
            use_sigmoid=False,
            reduction='mean',
            loss_weight=3.0,
            class_weight=[0.08, 1.2, 4.5, 0.1],
        ),
        loss_mask=dict(
            type='mmdet.CrossEntropyLoss',
            use_sigmoid=True,
            reduction='mean',
            loss_weight=5.0,
        ),
        loss_dice=dict(
            type='mmdet.DiceLoss',
            use_sigmoid=True,
            activate=True,
            naive_dice=True,
            eps=1.0,
            reduction='mean',
            loss_weight=5.0,
        ),
        pixel_decoder=dict(
            type='mmdet.MSDeformAttnPixelDecoder',
            num_outs=3,
            norm_cfg=dict(type='GN', num_groups=32),
            act_cfg=dict(type='ReLU'),
            positional_encoding=dict(normalize=True, num_feats=128),
            encoder=dict(
                num_layers=6,
                layer_cfg=dict(
                    self_attn_cfg=dict(
                        embed_dims=256,
                        num_heads=8,
                        num_levels=3,
                        num_points=4,
                        dropout=0.0,
                        batch_first=True,
                        im2col_step=64,
                        norm_cfg=None,
                        init_cfg=None,
                    ),
                    ffn_cfg=dict(
                        embed_dims=256,
                        feedforward_channels=1024,
                        num_fcs=2,
                        ffn_drop=0.0,
                        act_cfg=dict(type='ReLU', inplace=True),
                    ),
                ),
                init_cfg=None,
            ),
            init_cfg=None,
        ),
        positional_encoding=dict(normalize=True, num_feats=128),
        transformer_decoder=dict(
            num_layers=9,
            return_intermediate=True,
            layer_cfg=dict(
                self_attn_cfg=dict(
                    embed_dims=256,
                    num_heads=8,
                    attn_drop=0.0,
                    proj_drop=0.0,
                    dropout_layer=None,
                    batch_first=True,
                ),
                cross_attn_cfg=dict(
                    embed_dims=256,
                    num_heads=8,
                    attn_drop=0.0,
                    proj_drop=0.0,
                    dropout_layer=None,
                    batch_first=True,
                ),
                ffn_cfg=dict(
                    embed_dims=256,
                    feedforward_channels=2048,
                    num_fcs=2,
                    ffn_drop=0.0,
                    add_identity=True,
                    dropout_layer=None,
                    act_cfg=dict(type='ReLU', inplace=True),
                ),
            ),
            init_cfg=None,
        ),
        train_cfg=dict(
            num_points=12544,
            oversample_ratio=3.0,
            importance_sample_ratio=0.75,
            assigner=dict(
                type='mmdet.HungarianAssigner',
                match_costs=[
                    dict(type='mmdet.ClassificationCost', weight=2.0),
                    dict(type='mmdet.CrossEntropyLossCost', use_sigmoid=True, weight=5.0),
                    dict(type='mmdet.DiceCost', pred_act=True, eps=1.0, weight=5.0),
                ],
            ),
            sampler=dict(type='mmdet.MaskPseudoSampler'),
        ),
        strides=[4, 8, 16, 32],
    ),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'),
)

# --------------------
# Train / Val / Test pipelines (explicit & visible)
# --------------------
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='PhotoMetricDistortionRGB'),   # <<< 必须已注册
    dict(**morph_transform),

    dict(type='LoadAnnotations'),

    dict(
        type='RandomResize',
        scale=(2048, 1024),
        ratio_range=(0.8, 1.5),
        keep_ratio=True,
    ),
    dict(type='RandomCrop', crop_size=(512, 512), cat_max_ratio=0.5),
    dict(type='RandomFlip', prob=0.5),

    dict(type='PackSegInputs'),
]

val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(**morph_transform),
    dict(type='Resize', scale=(2048, 1024), keep_ratio=True),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs'),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(**morph_transform),
    dict(type='Resize', scale=(2048, 1024), keep_ratio=True),
    dict(type='PackSegInputs'),
]

# Inference-only pipeline (ndarray feed)
infer_pipeline = [
    dict(type='LoadImageFromNDArrayTLS'),
    dict(**morph_transform),
    dict(type='Resize', scale=(2048, 1024), keep_ratio=True),
    dict(type='PackSegInputs'),
]

# --------------------
# Dataloaders (pipelines are present here through dataset.pipeline)
# --------------------
train_dataloader = dict(
    batch_size=2,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='ConcatDataset',
        datasets=[
            dict(
                type=dataset_type,
                data_root=data_root,
                data_prefix=dict(img_path='train/images', seg_map_path='train/masks'),
                pipeline=train_pipeline,
            ),
            dict(
                type=dataset_type,
                data_root=data_root,
                data_prefix=dict(img_path='train_gc/images', seg_map_path='train_gc/masks'),
                pipeline=train_pipeline,
            ),
            dict(
                type=dataset_type,
                data_root=data_root,
                data_prefix=dict(img_path='train_gc/images', seg_map_path='train_gc/masks'),
                pipeline=train_pipeline,
            ),
            dict(
                type=dataset_type,
                data_root=data_root,
                data_prefix=dict(img_path='train_gc/images', seg_map_path='train_gc/masks'),
                pipeline=train_pipeline,
            ),
        ],
    ),
)

val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img_path='val/images', seg_map_path='val/masks'),
        pipeline=val_pipeline,
    ),
)

test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img_path='test/images', seg_map_path='test/masks'),
        pipeline=val_pipeline,  # 有 GT 时用 val_pipeline；无 GT 请换 test_pipeline
    ),
)

# --------------------
# Evaluators
# --------------------
val_evaluator  = dict(type='IoUMetric', iou_metrics=['mIoU', 'mDice', 'mFscore'])
test_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU', 'mDice', 'mFscore'])

# --------------------
# Visualizer
# --------------------
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='SegLocalVisualizer',
    name='visualizer',
    vis_backends=vis_backends,
)

# --------------------
# Keep (optional) original extra vars to avoid surprises
# --------------------
auto_scale_lr = dict(base_batch_size=16, enable=False)
img_ratios = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
