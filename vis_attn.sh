#!/bin/bash
#SBATCH --job-name=PTV3_Vis_Attn
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00     # 跑单块数据前向传播，2小时绰绰有余
#SBATCH --mem=120G          # 内存 120G 足够，降低资源申请能让你排队更快
#SBATCH --output=logs/vis_attention_%j.log
#SBATCH --error=logs/vis_attention_%j.err

export ENV_DIR=/home/b6ae/bolezhang.b6ae/Pointcept
export CODE_DIR=$SLURM_SUBMIT_DIR
export SIF_FILE=$ENV_DIR/pytorch_24.08.sif

# 确保 python 输出实时打印，不被缓存
export PYTHONUNBUFFERED=1

export PYTHONUSERBASE=$ENV_DIR/.pip
export PYTHONPATH=$CODE_DIR:$ENV_DIR/cumm:$ENV_DIR/spconv:$PYTHONPATH
export PATH=$ENV_DIR/.pip/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# 适配 GH200 架构
export TORCH_CUDA_ARCH_LIST="9.0a"

export MALLOC_ARENA_MAX=1
export OMP_NUM_THREADS=4

export TMPDIR=/dev/shm/tmp_${SLURM_JOB_ID}
mkdir -p $TMPDIR
mkdir -p logs  # 确保 logs 文件夹存在，防止 slurm 报错

echo "=========================================================="
echo "🚀 Starting Visualization Job $SLURM_JOB_ID on $(hostname)"
echo "=========================================================="

echo "📊 Checking GPU Status..."
nvidia-smi
echo "=========================================================="

# 🚀 容器化执行你的 Python 可视化脚本
apptainer exec --nv \
  --cleanenv \
  --containall \
  -B $CODE_DIR:/workspace \
  -B $ENV_DIR:$ENV_DIR \
  -B /lus:/lus \
  -B $TMPDIR:/tmp \
  $SIF_FILE \
  bash -c "
    cd /workspace
    export PYTHONPATH=/workspace:$ENV_DIR/cumm:$ENV_DIR/spconv:\$PYTHONPATH
    export PYTHONUSERBASE=$ENV_DIR/.pip
    export PATH=$ENV_DIR/.pip/bin:\$PATH
    
    echo '🎨 Running Attention Visualization Script...'
    python visualize_attention.py
  "

rm -rf $TMPDIR
echo "🎉 Job Completed!"