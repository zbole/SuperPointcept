#!/bin/bash
#SBATCH --job-name=Profile_DPT
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00  # 🚀 测算任务很快，申请 2 小时可以更快排进队列
#SBATCH --mem=64G        # 🚀 推理不需要占用太多 CPU 内存，64G 足够了
#SBATCH --output=logs/profile_efficiency_%j.log
#SBATCH --error=logs/profile_efficiency_%j.err

export ENV_DIR=/home/b6ae/bolezhang.b6ae/Pointcept
export CODE_DIR=$SLURM_SUBMIT_DIR
export SIF_FILE=$ENV_DIR/pytorch_24.08.sif

export PYTHONUNBUFFERED=1

export PYTHONUSERBASE=$ENV_DIR/.pip
export PYTHONPATH=$CODE_DIR:$ENV_DIR/cumm:$ENV_DIR/spconv:$PYTHONPATH
export PATH=$ENV_DIR/.pip/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export TORCH_CUDA_ARCH_LIST="9.0a"

# 🛡️ 防 OOM 底层护盾
export MALLOC_ARENA_MAX=1
export OMP_NUM_THREADS=4

export TMPDIR=/tmp/slurm_tmp_${SLURM_JOB_ID}
mkdir -p $TMPDIR

# 🚀 指向 Lustre 上的数据源
export DATA_DIR=/lus/lfs1aip2/projects/b6ae/datasets/sensaturban/processed_1025D_SP-PT

echo "=========================================================="
echo "🚀 Running Efficiency Profiler Job $SLURM_JOB_ID on $(hostname)"
echo "=========================================================="

# 🚀 预检：打印分配到的 GPU 信息
echo "📊 Checking GPU Status..."
nvidia-smi
echo "=========================================================="

apptainer exec --nv \
  --cleanenv \
  --containall \
  -B $CODE_DIR:/workspace \
  -B $ENV_DIR:$ENV_DIR \
  -B $DATA_DIR:/datasets/sensaturban/processed_1025D_SP-PT \
  -B /lus:/lus \
  -B $TMPDIR:/tmp \
  $SIF_FILE \
  bash -c "
    cd /workspace
    export PYTHONPATH=/workspace:$ENV_DIR/cumm:$ENV_DIR/spconv:\$PYTHONPATH
    export PYTHONUSERBASE=$ENV_DIR/.pip
    export PATH=$ENV_DIR/.pip/bin:\$PATH
    
    # 🚀 核心命令：直接运行效率测试脚本
    python tools/pro_benchmark.py
  "

rm -rf $TMPDIR