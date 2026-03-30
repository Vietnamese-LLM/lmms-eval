#!/bin/bash
#SBATCH --job-name=qwen3-a17b-eval
#SBATCH --partition=main          # adjust to your cluster's GPU partition
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --exclusive
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# ── Environment ──
cd /home/mingzhed/Projects/lmms-eval
source .venv/bin/activate

export NCCL_TIMEOUT=18000000
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=18000  # 5h – match NCCL_TIMEOUT
export NCCL_ASYNC_ERROR_HANDLING=1              # detect & report hung NCCL ops
set -a; source .env; set +a
export WANDB__SERVICE_WAIT=300                     # wandb init timeout (seconds)
export WANDB_INIT_TIMEOUT=300

# ── Multi-node rendezvous ──
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=$(( RANDOM % 1000 + 29500 ))

# ── Run ──
# 2 nodes × 8 GPUs = 16 GPUs total → TP=8, DP=2
# TP=8 (full node) gives the 397B MoE model more memory headroom
srun python -m torch.distributed.run \
    --nnodes=$SLURM_NNODES \
    --nproc_per_node=8 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    -m lmms_eval \
    --model vllm \
    --model_args model=Qwen/Qwen3.5-397B-A17B,tensor_parallel_size=8,data_parallel_size=2,gpu_memory_utilization=0.85,max_new_tokens=8192 \
    --tasks docvqa_val,infovqa_val,ocrbench_v2,vmmu,realworldqa,mmstar,blink,hrbench4k,hrbench8k,safety_vn,laion_mcq_en,laion_mcq_vi \
    --batch_size 16 \
    --output_path ./results/qwen3_5_a17b \
    --wandb_args "project=lmms-eval-image,job_type=eval" \
    --log_samples
