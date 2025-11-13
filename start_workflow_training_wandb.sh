#!/bin/bash

echo "======================================"
echo "Starting ROLL Workflow Optimizer Training with WandB"
echo "======================================"

# Set environment variables
export CUDA_HOME=/usr/local/cuda-12.5
export LD_LIBRARY_PATH=/usr/lib64-nvidia:/usr/local/cuda-12.5/lib64:$LD_LIBRARY_PATH
export PATH=/usr/local/cuda-12.5/bin:$PATH
export PYTHONPATH=$(pwd):$PYTHONPATH

# Load API keys from ~/.bashrc_apis
# Create this file with: echo 'export OPENAI_API_KEY="your-key"' >> ~/.bashrc_apis
# and: echo 'export WANDB_API_KEY="your-key"' >> ~/.bashrc_apis
if [ -f ~/.bashrc_apis ]; then
    source ~/.bashrc_apis
else
    echo "Warning: ~/.bashrc_apis not found. Please create it with your API keys."
    echo "Example: echo 'export OPENAI_API_KEY=\"your-key\"' >> ~/.bashrc_apis"
    echo "         echo 'export WANDB_API_KEY=\"your-key\"' >> ~/.bashrc_apis"
fi

# Set CUDA device
export CUDA_VISIBLE_DEVICES=0

# Verify GPU
echo "Verifying GPU..."
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Verify API keys
echo "Verifying API keys..."
echo "OpenAI API key: ${OPENAI_API_KEY:0:20}...${OPENAI_API_KEY: -10}"
echo "WandB API key: ${WANDB_API_KEY:0:10}...${WANDB_API_KEY: -10}"

# Training config
CONFIG_PATH="qwen3-8B-workflow-optimizer"
CONFIG_NAME="workflow_optimizer_full_training_wandb"

echo ""
echo "Configuration:"
echo "  Path: $CONFIG_PATH"
echo "  Name: $CONFIG_NAME"
echo "  Dataset: data/rl_training_data_full/"
echo "  Tracking: WandB (project: roll-workflow-optimizer)"
echo ""

# Launch training
python3 examples/start_rlvr_pipeline.py \
    --config_path "$CONFIG_PATH" \
    --config_name "$CONFIG_NAME"

