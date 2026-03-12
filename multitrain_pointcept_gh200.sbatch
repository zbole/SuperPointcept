#!/bin/bash
#SBATCH --job-name=SuperPTV3_Sensat
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mem=240G
#SBATCH --array=1-2  # 1跑纯物理版，2跑物理+语义双流版
#SBATCH --output=logs/super_exp_%A_task%a.log
#SBATCH --error=logs/super_exp_%A_task%a.err
#SBATCH --exclusive

export ENV_DIR=/home/b6ae/bolezhang.b6ae/Pointcept
export CODE_DIR=$SLURM_SUBMIT_DIR
export SIF_FILE=$ENV_DIR/pytorch_24.08.sif
export DATA_DIR=/lus/lfs1aip2/projects/b6ae/datasets
export WANDB_API_KEY=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
export PYTHONUNBUFFERED=1
export WANDB_MODE=online
export WANDB_DIR=/workspace/wandb_$SLURM_ARRAY_TASK_ID
export WANDB_START_METHOD=thread

export PYTHONUSERBASE=$ENV_DIR/.pip
export PYTHONPATH=$CODE_DIR:$ENV_DIR/cumm:$ENV_DIR/spconv:$PYTHONPATH
export PATH=$ENV_DIR/.pip/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export TORCH_CUDA_ARCH_LIST="9.0a"

export TMPDIR=$CODE_DIR/tmp_$SLURM_ARRAY_TASK_ID
mkdir -p $TMPDIR

# ✅ 改回你原本实际存在的 config 名字
CONFIG_NAME="semseg-pt-v3m1-0-base" 
EXP_NAME="SuperPTV3_Sensat_Task_GEO_DINO${SLURM_ARRAY_TASK_ID}"

echo "🚀 Starting Trial $SLURM_ARRAY_TASK_ID"
echo "📂 Using Config: $CONFIG_NAME"
echo "🏷️  Experiment Name: $EXP_NAME"

apptainer exec --nv \
  --cleanenv \
  --containall \
  -B $CODE_DIR:/workspace \
  -B $ENV_DIR:$ENV_DIR \
  -B $DATA_DIR:/datasets \
  -B /lus:/lus \
  $SIF_FILE \
  bash -c "
    cd /workspace
    export PYTHONPATH=/workspace:$ENV_DIR/cumm:$ENV_DIR/spconv:\$PYTHONPATH
    export PYTHONUSERBASE=$ENV_DIR/.pip
    export PATH=$ENV_DIR/.pip/bin:\$PATH
    export WANDB_API_KEY=$WANDB_API_KEY
    
    # ✅ 这里把写死的 config 换成变量 $CONFIG_NAME
    sh scripts/train.sh -g 1 -d sensaturban -c $CONFIG_NAME -n $EXP_NAME -r true
  "