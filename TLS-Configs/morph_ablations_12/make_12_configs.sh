#!/usr/bin/env bash
set -euo pipefail
source activate openmmlab_new

OUT_DIR="$(cd "$(dirname "$0")" && pwd)"
BASE_CFG="/lustre1/g/path_dwhho/new_LMH/mmsegmentation/TLS-Configs/TLSDataset_morph_Mask2Former.py"
WORK_ROOT="/lustre1/g/path_dwhho/new_LMH/mmsegmentation/work_dirs/morph_ablation_12"

mkdir -p "${OUT_DIR}"
mkdir -p "${WORK_ROOT}"

# 实验参数
declare -a COMBO_NAMES=("C1_H" "C2_H_GradMag" "C3_H_GradMag_LBP" "C5_All")
declare -a COMBO_ACTIVE_CHANNELS=(
  "['H']"
  "['H','GradMag']"
  "['H','GradMag','LBP']"
  "['H','GradMag','LBP','Gabor_S1','Gabor_S2']"
)
FULL_CHANNELS="['H','GradMag','LBP','Gabor_S1','Gabor_S2']"
EXPECTED_NUM_CH=5

declare -a WVALS=(0.05 0.1 0.15 0.2 0.5 1.0)
declare -a WTAGS=("w005" "w010" "w015" "w020" "w050" "w100")

for i in "${!COMBO_NAMES[@]}"; do
  cname="${COMBO_NAMES[$i]}"
  active_chlist="${COMBO_ACTIVE_CHANNELS[$i]}"

  for j in "${!WVALS[@]}"; do
    w="${WVALS[$j]}"
    wtag="${WTAGS[$j]}"
    cfg="${OUT_DIR}/${cname}_${wtag}.py"
    wdir="${WORK_ROOT}/${cname}_${wtag}"

    # 1. 保留你要求的控制面板格式
    cat > "${cfg}" << EOF
_base_ = ['${BASE_CFG}']

morph_cfg = dict(
    enabled=True,
    channel_names=${FULL_CHANNELS},
    active_channel_names=${active_chlist},
    expected_num_channels=${EXPECTED_NUM_CH},
)

data_preprocessor = dict(
    morph_mean=0.5,
    morph_std=0.5,
    morph_weight=${w},
    morph_enabled=True,
)

default_hooks = dict(
    morph_weight_warmup=dict(
        type='MorphWeightWarmupHook',
        hold_ratio=0.30,
        ramp_ratio=0.30,
        start_weight=0.0,
        target_weight=${w},
        log_interval=200,
        verbose=True,
    ),
)

work_dir = '${wdir}'
EOF

    # 2. 增强版 Python 修复逻辑
    python3 -c "
from mmengine.config import Config

def sync_pipeline(obj, ref):
    '''专门递归修复所有 dataloader 中的 pipeline 节点'''
    if isinstance(obj, list):
        for item in obj: sync_pipeline(item, ref)
    elif isinstance(obj, dict):
        # 如果是加载 morph 的节点，直接同步所有参数
        if obj.get('type') == 'LoadMorphologyAndConcat':
            obj.update(ref)
        # 继续向下找
        for v in obj.values():
            sync_pipeline(v, ref)

# 加载配置
cfg = Config.fromfile('${cfg}')

# --- A. 显式路径修复 (解决 model 内部引用失效问题) ---
if hasattr(cfg, 'model'):
    # 1. 修复 backbone 里的 morph 设置
    if 'backbone' in cfg.model and 'morph' in cfg.model.backbone:
        cfg.model.backbone.morph.enabled = cfg.morph_cfg.enabled
        cfg.model.backbone.morph.num_channels = cfg.morph_cfg.expected_num_channels
    
    # 2. 修复 model 里的 data_preprocessor
    if 'data_preprocessor' in cfg.model:
        cfg.model.data_preprocessor.morph_enabled = cfg.data_preprocessor.morph_enabled
        cfg.model.data_preprocessor.morph_weight = cfg.data_preprocessor.morph_weight

# --- B. 递归修复 Pipeline (解决 dataloaders 里的解包问题) ---
sync_pipeline(cfg._cfg_dict, cfg.morph_cfg)

# --- C. 同步 Warmup Hook (确保目标权重一致) ---
if 'morph_weight_warmup' in cfg.default_hooks:
    cfg.default_hooks.morph_weight_warmup.target_weight = cfg.data_preprocessor.morph_weight

# 导出平铺后的完整文件
cfg.dump('${cfg}')
"
    echo "[FIXED] Standalone config with explicit model sync: ${cfg}"
  done
done