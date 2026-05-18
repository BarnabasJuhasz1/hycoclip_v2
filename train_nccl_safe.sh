#!/bin/bash
# Wrapper script to run training with NCCL timeout and optimization settings

# Set aggressive NCCL timeouts and optimizations
export NCCL_TIMEOUT=1800  # 30 minutes instead of 10 minutes
export TORCH_NCCL_BLOCKING_WAIT=1  # Block instead of timeout (modern PyTorch API)
export NCCL_DEBUG=WARN  # Reduced verbosity (was INFO, too verbose)
export NCCL_IB_TIMEOUT=60  # InfiniBand timeout 60 seconds
export NCCL_ALGO=Ring  # Use Ring algorithm (more stable than Tree)
# REMOVED: NCCL_SOCKET_IFNAME=eth0 (Leonardo uses InfiniBand, this breaks connection)
# REMOVED: NCCL_TREE_THRESHOLD (not needed, let NCCL auto-select)

# Keep async GPU ops (blocking=1 is too slow for training)
export CUDA_LAUNCH_BLOCKING=0

# Run the training script
python scripts/train.py \
  --config configs/train_hycoclip_repulsion_vit_b.py \
  --num-gpus 4 \
  --output-dir /leonardo/home/userexternal/astanic0/IscrC_TBSP/fast/anja/output/training/hycoclip_repulsion_nccl_safe \
  --checkpoint-period 10000
