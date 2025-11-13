#!/bin/bash

echo "======================================"
echo "Starting ROLL Workflow Optimizer Training"
echo "======================================"

# Set environment variables for CUDA
export CUDA_HOME=/usr/local/cuda-12.5
export LD_LIBRARY_PATH=/usr/lib64-nvidia:/usr/local/cuda-12.5/lib64:$LD_LIBRARY_PATH
export PATH=/usr/local/cuda-12.5/bin:$PATH
export PYTHONPATH=$(pwd):$PYTHONPATH

# Set CUDA device
export CUDA_VISIBLE_DEVICES=0

# Verify GPU
echo "Verifying GPU..."
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Verify PyTorch CUDA
echo "Verifying PyTorch CUDA..."
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Training config
CONFIG_PATH="qwen3-8B-workflow-optimizer"
CONFIG_NAME="workflow_optimizer_full_training"

echo ""
echo "Configuration:"
echo "  Path: $CONFIG_PATH"
echo "  Name: $CONFIG_NAME"
echo "  Dataset: data/rl_training_data_full/"
echo ""

# Launch training
python3 examples/start_rlvr_pipeline.py \
    --config_path "$CONFIG_PATH" \
    --config_name "$CONFIG_NAME"

