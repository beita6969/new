#!/bin/bash
echo "=== 模型下载状态检查 ==="
echo ""

cache_dir="$HOME/.cache/modelscope/hub/models/Qwen/Qwen2.5-7B-Instruct"

if [ ! -d "$cache_dir" ]; then
    echo "❌ 模型缓存目录不存在"
    exit 1
fi

total_size=0
downloaded_size=0
files_complete=0
files_total=4

for i in {1..4}; do
    file="model-0000${i}-of-00004.safetensors"
    
    if [ "$i" -eq 1 ]; then
        expected_size=3942825968  # 3.67GB
    elif [ "$i" -eq 2 ] || [ "$i" -eq 3 ]; then
        expected_size=3865470704  # 3.60GB
    else
        expected_size=3552258032  # 3.31GB
    fi
    
    total_size=$((total_size + expected_size))
    
    if [ -f "$cache_dir/$file" ]; then
        current_size=$(stat -f%z "$cache_dir/$file" 2>/dev/null || stat -c%s "$cache_dir/$file" 2>/dev/null)
        downloaded_size=$((downloaded_size + current_size))
        
        percent=$((current_size * 100 / expected_size))
        human_current=$(numfmt --to=iec-i --suffix=B $current_size 2>/dev/null || echo "$((current_size / 1024 / 1024))M")
        human_expected=$(numfmt --to=iec-i --suffix=B $expected_size 2>/dev/null || echo "$((expected_size / 1024 / 1024))M")
        
        if [ $percent -ge 99 ]; then
            echo "✅ $file: $human_current / $human_expected ($percent%)"
            files_complete=$((files_complete + 1))
        else
            echo "⏬ $file: $human_current / $human_expected ($percent%)"
        fi
    else
        echo "❌ $file: 未开始下载"
    fi
done

echo ""
echo "=== 总体进度 ==="
overall_percent=$((downloaded_size * 100 / total_size))
echo "完成文件: $files_complete / $files_total"
echo "总体进度: $overall_percent%"
echo ""

if [ $files_complete -eq $files_total ]; then
    echo "🎉 所有模型文件下载完成！可以开始训练了！"
    exit 0
else
    echo "⏳ 还有 $((files_total - files_complete)) 个文件正在下载..."
    exit 1
fi
