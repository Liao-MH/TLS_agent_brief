#!/bin/bash
#SBATCH --job-name=TLS_agent       # 任务名称（便于识别）
#SBATCH --mail-type=BEGIN,END,FAIL    # 2. Send email upon events (Options: NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=u3013381@connect.hku.hk     #    Email address to receive notification
#SBATCH --ntasks=1                  # 单节点任务
#SBATCH --cpus-per-task=4           # 每个任务分配的CPU核心数
#SBATCH --nodes=1                   # 使用1个节点
#SBATCH --partition=l40s             # 指定GPU分区
#SBATCH --qos=gpu                   # GPU服务质量等级
#SBATCH --gres=gpu:1                # 申请2块GPU
#SBATCH --mem=50G                  # 内存总量
#SBATCH --output=logs/TLS_agent.out  # 输出日志路径
#SBATCH --error=logs/TLS_agent.err   # 错误日志路径（补充缺失项）

# -------- Path setup (portable) --------
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MMSEG_ROOT_DEFAULT="/lustre1/g/path_dwhho/new_LMH/mmsegmentation"
DATA_ROOT_DEFAULT="/lustre1/g/path_dwhho/new_LMH/svs-slides"
OUTPUT_ROOT_DEFAULT="/home/u3013381/new_LMH/TLS_segmentation/tls_toolkit_project/output"
CHECKPOINT_ROOT_DEFAULT="/lustre1/g/path_dwhho/new_LMH/mmsegmentation/work_dirs"

TLS_TOOLKIT_ROOT="${TLS_TOOLKIT_ROOT:-$MMSEG_ROOT_DEFAULT}"
MMSEG_ROOT="${MMSEG_ROOT:-$MMSEG_ROOT_DEFAULT}"
CONFIG_ROOT="${CONFIG_ROOT:-$MMSEG_ROOT/TLS-Configs}"
DATA_ROOT="${DATA_ROOT:-$DATA_ROOT_DEFAULT}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$OUTPUT_ROOT_DEFAULT}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-$CHECKPOINT_ROOT_DEFAULT}"

export PROJECT_ROOT DATA_ROOT OUTPUT_ROOT CHECKPOINT_ROOT

mkdir -p "$PROJECT_ROOT/logs" "$OUTPUT_ROOT"

# 加载CUDA环境模块（如可用）
if command -v module >/dev/null 2>&1; then
  module purge                        # 清除已有模块（避免冲突）
  module load cuda/11.8               # 明确CUDA版本（需与Python环境匹配）
fi

# 激活conda环境（如可用）
if command -v conda >/dev/null 2>&1; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate openmmlab_wsi || true
fi

# 执行训练脚本（建议使用绝对路径）
export PYTHONUNBUFFERED=1
# 1) 工程根目录（包含 tls_toolkit/ 的那一层）
export TLS_TOOLKIT_ROOT

# 2) 你的 mmsegmentation 源码根目录（如确实需要）
export MMSEG_ROOT

# 3) 组合 PYTHONPATH（注意顺序：先放工程根目录）
export PYTHONPATH="${TLS_TOOLKIT_ROOT}:${MMSEG_ROOT}:${PYTHONPATH}"

# 建议：从工程根目录执行，避免相对路径/日志目录问题
cd "${TLS_TOOLKIT_ROOT}"

#  --config_file /lustre1/g/path_dwhho/new_LMH/mmsegmentation/TLS-Configs/morph_ablations_12/C3_H_GradMag_LBP_w050.py \
#  --checkpoint /lustre1/g/path_dwhho/new_LMH/mmsegmentation/work_dirs/morph_ablation_12/C3_H_GradMag_LBP_w050/best_mFscore_iter_10000.pth \

#/lustre1/g/path_dwhho/new_LMH/mmsegmentation/TLS-Configs/Standard_config_with_morph.py
#/lustre1/g/path_dwhho/new_LMH/mmsegmentation/work_dirs/morph_ablation_12/C3_H_GradMag_LBP_w050/best_mFscore_iter_10000.pth

#/lustre1/g/path_dwhho/new_LMH/mmsegmentation/TLS-Configs/TLSDataset_Mask2Former_20251203.py
#/lustre1/g/path_dwhho/new_LMH/mmsegmentation/work_dirs/TLSDataset-Mask2Former/best_mIoU_iter_46400.pth

# fast_lowres low_memory default_balanced

WSI_PATH="${WSI_PATH:-$DATA_ROOT/CRC/279912.svs}"
OUT_ROOT="${OUT_ROOT:-$OUTPUT_ROOT}"
CONFIG_FILE="${CONFIG_FILE:-$CONFIG_ROOT/Standard_config_with_morph.py}"
CHECKPOINT_FILE="${CHECKPOINT_FILE:-$CHECKPOINT_ROOT/morph_ablation_12/C3_H_GradMag_LBP_w050/best_mFscore_iter_10000.pth}"

python -m tls_toolkit.cli agent \
  --preset low_memory \
  --allow_preset_fallback \
  --wsi "${WSI_PATH}" \
  --out_root "${OUT_ROOT}" \
  --device cuda \
  --config_file "${CONFIG_FILE}" \
  --checkpoint "${CHECKPOINT_FILE}" \
  --max_reruns 0
