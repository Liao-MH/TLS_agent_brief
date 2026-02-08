#!/bin/bash
#SBATCH --job-name=Morph_Mask2former       # 任务名称（便于识别）
#SBATCH --mail-type=BEGIN,END,FAIL    # 2. Send email upon events (Options: NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=u3013381@connect.hku.hk     #    Email address to receive notification
#SBATCH --ntasks=1                  # 单节点任务
#SBATCH --cpus-per-task=4           # 每个任务分配的CPU核心数
#SBATCH --nodes=1                   # 使用1个节点
#SBATCH --partition=l40s             # 指定GPU分区
#SBATCH --qos=gpu                   # GPU服务质量等级
#SBATCH --gres=gpu:4                # 申请2块GPU
#SBATCH --mem=30G                  # 内存总量
#SBATCH --output=logs/Morph_Mask2former.out  # 输出日志路径
#SBATCH --error=logs/Morph_Mask2former.err   # 错误日志路径（补充缺失项）

# -------- Path setup (portable) --------
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MMSEG_ROOT_DEFAULT="/lustre1/g/path_dwhho/new_LMH/mmsegmentation"
DATA_ROOT_DEFAULT="/lustre1/g/path_dwhho/new_LMH/TLS_segmentation/188_slides_tiles/merge_TLS_and_Immunecluster_2048"
OUTPUT_ROOT_DEFAULT="/lustre1/g/path_dwhho/new_LMH/mmsegmentation/work_dirs"
CHECKPOINT_ROOT_DEFAULT="/lustre1/g/path_dwhho/new_LMH/mmsegmentation/work_dirs"

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
  conda activate openmmlab_new || true
fi

# 执行训练脚本（建议使用绝对路径）
export PYTHONUNBUFFERED=1
export PYTHONPATH="${MMSEG_ROOT}:${PYTHONPATH}"
# 为每个作业生成稳定且基本不冲突的端口：29500~30499
export PORT=$((29500 + SLURM_JOB_ID % 1000))
export MASTER_PORT=$PORT
export MASTER_ADDR=127.0.0.1


bash "${MMSEG_ROOT}/tools/dist_train.sh" "${CONFIG_ROOT}/TLSDataset_morph_Mask2Former.py" 4
