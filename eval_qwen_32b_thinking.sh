#!/bin/bash
#SBATCH --job-name=qwen3-vl-32b-thinking-eval
#SBATCH --partition=main          # adjust to your cluster's GPU partition
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=8                 # or: --gres=gpu:8
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
export MASTER_PORT=$(( RANDOM % 1000 + 29500 ))   # avoid port collisions

# ── Run ──
python -m torch.distributed.run \
    --standalone \
    --nproc_per_node=8 \
    -m lmms_eval \
    --model vllm \
    --model_args model=Qwen/Qwen3-VL-32B-Thinking,tensor_parallel_size=4,data_parallel_size=2,gpu_memory_utilization=0.85,max_new_tokens=8192 \
    --tasks docvqa_val,infovqa_val,ocrbench_v2,vmmu,realworldqa,mmstar,blink,hrbench4k,hrbench8k,safety_vn,laion_mcq_en,laion_mcq_vi \
    --batch_size 16 \
    --output_path ./results/qwen3_vl_32b_thinking \
    --wandb_args "project=lmms-eval-image,job_type=eval" \
    --log_samples
