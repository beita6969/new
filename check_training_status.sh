#!/bin/bash
echo "=== Training Process Status ==="
ps aux | grep start_rlvr_pipeline | grep -v grep || echo "No training process found"
echo ""
echo "=== GPU Usage ==="
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader
echo ""
echo "=== Download Progress ==="
tail -20 training_initial_output.log | grep -E "Downloading.*safetensors|%|Processing"
echo ""
echo "=== Latest Log Entries ==="
tail -5 training_initial_output.log
