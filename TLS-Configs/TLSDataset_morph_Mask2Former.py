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
    os.path.join(OUTPUT_ROOT, "work_dirs/TLSDataset-Mask2Former_morphIF"),
)
PRETRAINED = os.getenv("PRETRAINED", os.path.join(CHECKPOINT_ROOT, "best_mFscore_iter_19000.pth"))

custom_imports = dict(imports=['projects.morph_inputfusion'], allow_failed_imports=False)

# -----------------------------
# Morphology feature settings (input-fusion)
# -----------------------------
morph_cfg = dict(
    enabled=False,  # set False for RGB baseline
    img_dirname='images',
    morph_dirname='morph',      # e.g., .../train_gc/morph/<same_stem>.npy
    morph_ext='.npy',           # supports .npy or .npz
    morph_npz_key=None,         # set key name if using .npz; otherwise None
    channel_names=['H', 'GradMag', 'LBP', 'Gabor_S1', 'Gabor_S2'],
    active_channel_names=['H', 'GradMag', 'LBP', 'Gabor_S1', 'Gabor_S2'],  # 只改这里
    expected_num_channels=5,               # 强烈建议
    clip=(0.0, 1.0),
    strict=True,

)

auto_scale_lr = dict(base_batch_size=16, enable=False)
backbone_embed_multi = dict(decay_mult=0.0, lr_mult=0.1)
backbone_norm_multi = dict(decay_mult=0.0, lr_mult=0.1)
crop_size = (1200, 1200)

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
    'backbone.stages.2.blocks.10.norm': dict(decay_mult=0.0, lr_mult=0.1),
    'backbone.stages.2.blocks.11.norm': dict(decay_mult=0.0, lr_mult=0.1),
    'backbone.stages.2.blocks.12.norm': dict(decay_mult=0.0, lr_mult=0.1),
    'backbone.stages.2.blocks.13.norm': dict(decay_mult=0.0, lr_mult=0.1),
    'backbone.stages.2.blocks.14.norm': dict(decay_mult=0.0, lr_mult=0.1),
    'backbone.stages.2.blocks.15.norm': dict(decay_mult=0.0, lr_mult=0.1),
    'backbone.stages.2.blocks.16.norm': dict(decay_mult=0.0, lr_mult=0.1),
    'backbone.stages.2.blocks.17.norm': dict(decay_mult=0.0, lr_mult=0.1),
    'backbone.stages.2.blocks.2.norm': dict(decay_mult=0.0, lr_mult=0.1),
    'backbone.stages.2.blocks.3.norm': dict(decay_mult=0.0, lr_mult=0.1),
    'backbone.stages.2.blocks.4.norm': dict(decay_mult=0.0, lr_mult=0.1),
    'backbone.stages.2.blocks.5.norm': dict(decay_mult=0.0, lr_mult=0.1),
    'backbone.stages.2.blocks.6.norm': dict(decay_mult=0.0, lr_mult=0.1),
    'backbone.stages.2.blocks.7.norm': dict(decay_mult=0.0, lr_mult=0.1),
    'backbone.stages.2.blocks.8.norm': dict(decay_mult=0.0, lr_mult=0.1),
    'backbone.stages.2.blocks.9.norm': dict(decay_mult=0.0, lr_mult=0.1),
    'backbone.stages.2.downsample.norm': dict(decay_mult=0.0, lr_mult=0.1),
    'backbone.stages.3.blocks.0.norm': dict(decay_mult=0.0, lr_mult=0.1),
    'backbone.stages.3.blocks.1.norm': dict(decay_mult=0.0, lr_mult=0.1),
    'level_embed': dict(decay_mult=0.0, lr_mult=1.0),
    'query_embed': dict(decay_mult=0.0, lr_mult=1.0),
    'query_feat': dict(decay_mult=0.0, lr_mult=1.0),
    'relative_position_bias_table': dict(decay_mult=0.0, lr_mult=0.1)
})

# NOTE: 这一段你原来就有（全局 data_preprocessor），我按“尽量不动”的要求保留
data_preprocessor = dict(
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255,
    size=(1200, 1200),
    type='MorphSegDataPreProcessor',
    rgb_mean=[123.675, 116.28, 103.53],
    rgb_std=[58.395, 57.12, 57.375],
    
    morph_mean=0.5,
    morph_std=0.5,
    morph_weight=1.0,
    morph_enabled=morph_cfg['enabled'],
)

data_root = DATA_ROOT
dataset_type = 'TLSDataset'

default_hooks = dict(
    checkpoint=dict(
        by_epoch=False,
        interval=50000,
        max_keep_ckpts=2,
        save_best='mFscore',#mIoU
        rule='greater',   # 关键：必须指定
        type='CheckpointHook'),
    logger=dict(interval=200, log_metric_by_epoch=False, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(type='SegVisualizationHook'),
    
    # ---- ADD: percentage-based morph weight warmup ----
    morph_weight_warmup=dict(
    type='MorphWeightWarmupHook',
    hold_ratio=0.30,      # 前 30% 训练：morph_weight = start_weight
    ramp_ratio=0.30,      # 接下来 30%：线性升到 target_weight
    start_weight=0.0,
    target_weight=data_preprocessor['morph_weight'],
    log_interval=200,
    verbose=True,
    ),
)

default_scope = 'mmseg'
depths = [2, 2, 18, 2]
embed_multi = dict(decay_mult=0.0, lr_mult=1.0)

env_cfg = dict(
    cudnn_benchmark=True,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0)
)

img_ratios = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
load_from = None
log_level = 'INFO'
log_processor = dict(by_epoch=False)

model = dict(
    backbone=dict(
        attn_drop_rate=0.0,
        depths=[2, 2, 18, 2],
        drop_path_rate=0.3,
        drop_rate=0.0,
        embed_dims=192,
        frozen_stages=-1,
        init_cfg=dict(
            checkpoint=PRETRAINED,
            type='Pretrained'),
        mlp_ratio=4,
        num_heads=[6, 12, 24, 48],
        out_indices=(0, 1, 2, 3),
        patch_norm=True,
        pretrain_img_size=384,
        qk_scale=None,
        qkv_bias=True,
        type='MorphSwinTransformer',
        morph=dict(
            enabled=morph_cfg['enabled'],
            num_channels=len(morph_cfg['channel_names']),
            project_to_rgb=True,
        ),
        window_size=12,
        with_cp=False
    ),

    # -----------------------------
    # CHANGE #1 (必须改)：用 MorphSegDataPreProcessor
    # -----------------------------
    data_preprocessor=data_preprocessor,

    decode_head=dict(
        align_corners=False,
        enforce_decoder_input_project=False,
        feat_channels=256,
        in_channels=[192, 384, 768, 1536],
        loss_cls=dict(
            class_weight=[0.08, 1.2, 4.5, 0.1],
            loss_weight=3.0,
            reduction='mean',
            type='mmdet.CrossEntropyLoss',
            use_sigmoid=False),
        loss_dice=dict(
            activate=True,
            eps=1.0,
            loss_weight=5.0,
            naive_dice=True,
            reduction='mean',
            type='mmdet.DiceLoss',
            use_sigmoid=True),
        loss_mask=dict(
            loss_weight=5.0,
            reduction='mean',
            type='mmdet.CrossEntropyLoss',
            use_sigmoid=True),
        num_classes=3,
        num_queries=100,
        num_transformer_feat_level=3,
        out_channels=256,
        pixel_decoder=dict(
            act_cfg=dict(type='ReLU'),
            encoder=dict(
                init_cfg=None,
                layer_cfg=dict(
                    ffn_cfg=dict(
                        act_cfg=dict(inplace=True, type='ReLU'),
                        embed_dims=256,
                        feedforward_channels=1024,
                        ffn_drop=0.0,
                        num_fcs=2),
                    self_attn_cfg=dict(
                        batch_first=True,
                        dropout=0.0,
                        embed_dims=256,
                        im2col_step=64,
                        init_cfg=None,
                        norm_cfg=None,
                        num_heads=8,
                        num_levels=3,
                        num_points=4)),
                num_layers=6),
            init_cfg=None,
            norm_cfg=dict(num_groups=32, type='GN'),
            num_outs=3,
            positional_encoding=dict(normalize=True, num_feats=128),
            type='mmdet.MSDeformAttnPixelDecoder'),
        positional_encoding=dict(normalize=True, num_feats=128),
        strides=[4, 8, 16, 32],
        train_cfg=dict(
            assigner=dict(
                match_costs=[
                    dict(type='mmdet.ClassificationCost', weight=2.0),
                    dict(type='mmdet.CrossEntropyLossCost', use_sigmoid=True, weight=5.0),
                    dict(eps=1.0, pred_act=True, type='mmdet.DiceCost', weight=5.0),
                ],
                type='mmdet.HungarianAssigner'),
            importance_sample_ratio=0.75,
            num_points=12544,
            oversample_ratio=3.0,
            sampler=dict(type='mmdet.MaskPseudoSampler')),
        transformer_decoder=dict(
            init_cfg=None,
            layer_cfg=dict(
                cross_attn_cfg=dict(
                    attn_drop=0.0,
                    batch_first=True,
                    dropout_layer=None,
                    embed_dims=256,
                    num_heads=8,
                    proj_drop=0.0),
                ffn_cfg=dict(
                    act_cfg=dict(inplace=True, type='ReLU'),
                    add_identity=True,
                    dropout_layer=None,
                    embed_dims=256,
                    feedforward_channels=2048,
                    ffn_drop=0.0,
                    num_fcs=2),
                self_attn_cfg=dict(
                    attn_drop=0.0,
                    batch_first=True,
                    dropout_layer=None,
                    embed_dims=256,
                    num_heads=8,
                    proj_drop=0.0)),
            num_layers=9,
            return_intermediate=True),
        type='Mask2FormerHead'),
    test_cfg=dict(mode='whole'),
    train_cfg=dict(),
    type='EncoderDecoder'
)

norm_cfg = dict(requires_grad=True, type='BN')
num_classes = 150

optim_wrapper = dict(
    clip_grad=dict(max_norm=0.01, norm_type=2),
    optimizer=dict(
        betas=(0.9, 0.999),
        eps=1e-08,
        lr=1e-06,
        type='AdamW',
        weight_decay=0.05),
    paramwise_cfg=dict(
        custom_keys=custom_keys,
        norm_decay_mult=0.0),
    type='OptimWrapper'
)

optimizer = dict(betas=(0.9, 0.999), eps=1e-08, lr=1e-06, type='AdamW', weight_decay=0.05)

param_scheduler = [
    dict(begin=0, by_epoch=False, end=160000, eta_min=0, power=0.9, type='PolyLR'),
]

pretrained = PRETRAINED
randomness = dict(seed=0)
resume = False

test_cfg = dict(type='TestLoop')

test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        data_prefix=dict(img_path='test/images', seg_map_path='test/masks'),
        data_root=DATA_ROOT,
        pipeline=[
            dict(type='LoadImageFromFile'),
            # -----------------------------
            # CHANGE #2 (必须改)：test 加载 morph
            # -----------------------------
            dict(type='LoadMorphologyAndConcat', **morph_cfg),
            dict(keep_ratio=True, scale=(2048, 1024), type='Resize'),
            dict(type='LoadAnnotations'),
            dict(type='PackSegInputs'),
        ],
        type='TLSDataset'),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler')
)

test_evaluator = dict(iou_metrics=['mIoU', 'mDice', 'mFscore'], type='IoUMetric')

test_pipeline = [
    dict(type='LoadImageFromFile'),
    # -----------------------------
    # CHANGE #3 (必须改)：test_pipeline 同步加 morph（便于 tools/test.py 直接引用）
    # -----------------------------
    dict(type='LoadMorphologyAndConcat', **morph_cfg),
    dict(keep_ratio=True, scale=(2048, 1024), type='Resize'),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs'),
]

# -----------------------------
# NEW: val_pipeline（和 val_dataloader 内保持一致）
# -----------------------------
val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadMorphologyAndConcat', **morph_cfg),
    dict(type='Resize', scale=(2048, 1024), keep_ratio=True),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs'),
]


train_cfg = dict(max_iters=15000, type='IterBasedTrainLoop', val_interval=default_hooks['morph_weight_warmup']['log_interval'])

# 你的 train_dataloader 我按要求“尽量不动”，仅保留你原本内容
train_dataloader = dict(
    batch_size=2,
    dataset=dict(
        datasets=[
            dict(
                data_prefix=dict(img_path='train/images', seg_map_path='train/masks'),
                data_root=DATA_ROOT,
                pipeline=[
                    dict(type='LoadImageFromFile'),
                    dict(type='LoadMorphologyAndConcat', **morph_cfg),
                    dict(type='LoadAnnotations'),
                    dict(keep_ratio=True, ratio_range=(0.8, 1.5), scale=(2048, 1024), type='RandomResize'),
                    dict(cat_max_ratio=0.5, crop_size=(512, 512), type='RandomCrop'),
                    dict(prob=0.5, type='RandomFlip'),
                    dict(type='PhotoMetricDistortionRGB'),
                    dict(type='PackSegInputs'),
                ],
                type='TLSDataset'),
            dict(
                data_prefix=dict(img_path='train_gc/images', seg_map_path='train_gc/masks'),
                data_root=DATA_ROOT,
                pipeline=[
                    dict(type='LoadImageFromFile'),
                    dict(type='LoadMorphologyAndConcat', **morph_cfg),
                    dict(type='LoadAnnotations'),
                    dict(keep_ratio=True, ratio_range=(0.8, 1.5), scale=(2048, 1024), type='RandomResize'),
                    dict(cat_max_ratio=0.5, crop_size=(512, 512), type='RandomCrop'),
                    dict(prob=0.5, type='RandomFlip'),
                    dict(type='PhotoMetricDistortionRGB'),
                    dict(type='PackSegInputs'),
                ],
                type='TLSDataset'),
            dict(
                data_prefix=dict(img_path='train_gc/images', seg_map_path='train_gc/masks'),
                data_root=DATA_ROOT,
                pipeline=[
                    dict(type='LoadImageFromFile'),
                    dict(type='LoadMorphologyAndConcat', **morph_cfg),
                    dict(type='LoadAnnotations'),
                    dict(keep_ratio=True, ratio_range=(0.8, 1.5), scale=(2048, 1024), type='RandomResize'),
                    dict(cat_max_ratio=0.5, crop_size=(512, 512), type='RandomCrop'),
                    dict(prob=0.5, type='RandomFlip'),
                    dict(type='PhotoMetricDistortionRGB'),
                    dict(type='PackSegInputs'),
                ],
                type='TLSDataset'),
            dict(
                data_prefix=dict(img_path='train_gc/images', seg_map_path='train_gc/masks'),
                data_root=DATA_ROOT,
                pipeline=[
                    dict(type='LoadImageFromFile'),
                    dict(type='LoadMorphologyAndConcat', **morph_cfg),
                    dict(type='LoadAnnotations'),
                    dict(keep_ratio=True, ratio_range=(0.8, 1.5), scale=(2048, 1024), type='RandomResize'),
                    dict(cat_max_ratio=0.5, crop_size=(512, 512), type='RandomCrop'),
                    dict(prob=0.5, type='RandomFlip'),
                    dict(type='PhotoMetricDistortionRGB'),
                    dict(type='PackSegInputs'),
                ],
                type='TLSDataset'),
        ],
        type='ConcatDataset'),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='InfiniteSampler')
)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadMorphologyAndConcat', **morph_cfg),
    dict(type='LoadAnnotations'),
    dict(keep_ratio=True, ratio_range=(0.8, 1.5), scale=(2048, 1024), type='RandomResize'),
    dict(cat_max_ratio=0.5, crop_size=(512, 512), type='RandomCrop'),
    dict(prob=0.5, type='RandomFlip'),
    dict(type='PhotoMetricDistortionRGB'),
    dict(type='PackSegInputs'),
]

tta_model = dict(type='SegTTAModel')

tta_pipeline = [
    dict(file_client_args=dict(backend='disk'), type='LoadImageFromFile'),
    # -----------------------------
    # CHANGE #4 (必须改)：TTA 进入 TestTimeAug 前加载 morph
    # -----------------------------
    dict(type='LoadMorphologyAndConcat', **morph_cfg),
    dict(
        transforms=[
            [
                dict(keep_ratio=True, scale_factor=0.5, type='Resize'),
                dict(keep_ratio=True, scale_factor=0.75, type='Resize'),
                dict(keep_ratio=True, scale_factor=1.0, type='Resize'),
                dict(keep_ratio=True, scale_factor=1.25, type='Resize'),
                dict(keep_ratio=True, scale_factor=1.5, type='Resize'),
                dict(keep_ratio=True, scale_factor=1.75, type='Resize'),
            ],
            [
                dict(direction='horizontal', prob=0.0, type='RandomFlip'),
                dict(direction='horizontal', prob=1.0, type='RandomFlip'),
            ],
            [
                dict(type='LoadAnnotations'),
            ],
            [
                dict(type='PackSegInputs'),
            ],
        ],
        type='TestTimeAug'),
]

val_cfg = dict(type='ValLoop')

val_dataloader = dict(
    batch_size=1,
    dataset=dict(
        data_prefix=dict(img_path='val/images', seg_map_path='val/masks'),
        data_root=DATA_ROOT,
        pipeline=val_pipeline,
        type='TLSDataset'),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler')
)

val_evaluator = dict(iou_metrics=['mIoU', 'mDice', 'mFscore'], type='IoUMetric')

vis_backends = [dict(type='LocalVisBackend')]

visualizer = dict(
    name='visualizer',
    type='SegLocalVisualizer',
    vis_backends=[dict(type='LocalVisBackend')]
)

# -----------------------------
# CHANGE #6 (强烈建议)：换一个 work_dir，避免覆盖 baseline
# -----------------------------
work_dir = WORK_DIR
