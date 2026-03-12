#!/bin/bash
#SBATCH --job-name=SuperPTV3_V2
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=logs/super_exp_v2_%j.log
#SBATCH --error=logs/super_exp_v2_%j.err
#SBATCH --exclusive

# 1. 路径定义：明确区分“环境目录”和“代码目录”
export ENV_DIR=/home/b6ae/bolezhang.b6ae/Pointcept
export CODE_DIR=$SLURM_SUBMIT_DIR  # 假设你在 SuperPointcept 目录下提交
export SIF_FILE=$ENV_DIR/pytorch_24.08.sif
export DATA_DIR=/lus/lfs1aip2/projects/b6ae/datasets

# 2. WandB 配置
export WANDB_API_KEY=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
export WANDB_MODE=online
export WANDB_DIR=/workspace/wandb
export WANDB_START_METHOD=thread

# 3. 针对 GH200 的环境变量（指向 Pointcept 的编译产物）
# 我们依然需要 Pointcept 下的 .pip 文件夹，因为那里有编译好的 spconv
export PYTHONUSERBASE=$ENV_DIR/.pip
export PYTHONPATH=$CODE_DIR:$ENV_DIR/cumm:$ENV_DIR/spconv:$PYTHONPATH
export PATH=$ENV_DIR/.pip/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export TORCH_CUDA_ARCH_LIST="9.0a"
export TMPDIR=$CODE_DIR/tmp
mkdir -p $TMPDIR

echo "🚀 Training started with SuperPointcept GCDM code"
echo "📍 Code Directory: $CODE_DIR"
echo "🛠️ Environment Directory: $ENV_DIR"

# 4. 使用 Apptainer 启动容器
# 注意：我们需要同时挂载 CODE_DIR 到 /workspace 和 挂载 ENV_DIR 供容器读取环境
apptainer exec --nv \
  --cleanenv \
  --containall \
  -B $CODE_DIR:/workspace \
  -B $ENV_DIR:$ENV_DIR \
  -B $DATA_DIR:/datasets \
  $SIF_FILE \
  bash -c "
    cd /workspace
    # 容器内路径设置：优先级最高的是当前代码目录 /workspace
    export PYTHONPATH=/workspace:$ENV_DIR/cumm:$ENV_DIR/spconv:\$PYTHONPATH
    export PYTHONUSERBASE=$ENV_DIR/.pip
    export PATH=$ENV_DIR/.pip/bin:\$PATH
    export WANDB_API_KEY=$WANDB_API_KEY
    
    # 预检：验证 pointcept 包的路径是否指向了 /workspace (你的新代码)
    python3 -c 'import pointcept; print(\"✅ Current Code Path:\", pointcept.__file__)'
    python3 -c 'import torch; import cumm; print(\"✅ Device:\", torch.cuda.get_device_name(0)); print(\"✅ CUMM Loaded!\")'
    
    # 正式训练
    sh scripts/train.sh -g 1 -d sensaturban -c semseg-pt-v3m1-0-base -n SuperPTV3_GCDM_Sensat_grid0.1_GH200 -r true
  "