#!/bin/bash
echo "=== 当前下载进度 ==="
echo ""
echo "model-00001: ✅ 已完成 (3.67GB)"

# 检查缓存目录中的文件
cache_dir="$HOME/.cache/modelscope/hub/models/Qwen/Qwen2.5-7B-Instruct"
if [ -d "$cache_dir" ]; then
    for file in model-00002-of-00004.safetensors model-00003-of-00004.safetensors model-00004-of-00004.safetensors; do
        if [ -f "$cache_dir/$file" ]; then
            size=$(du -h "$cache_dir/$file" | cut -f1)
            echo "$file: $size 已下载"
        else
            echo "$file: 未找到"
        fi
    done
fi

echo ""
echo "=== 预计剩余时间 ==="
echo "以当前速度 (400-500 KB/s):"
echo "- 剩余约 9.5GB 需要下载"
echo "- 预计需要 5-7 小时完成"
echo ""
echo "建议："
echo "1. 继续等待 ModelScope 下载 (慢但稳定)"
echo "2. 或者手动下载模型后继续训练"
