#!/bin/bash
# Box automation audio analysis - startup script

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
cd "$SCRIPT_DIR"

echo "=========================================="
echo "Box 自动化录音分析 Skill"
echo "=========================================="

# 检查依赖（使用国内源）
echo "检查依赖..."
if ! python3 -c "import funasr" 2>/dev/null; then
    echo "📦 正在安装依赖（使用清华源）..."
    pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
    if [ $? -eq 0 ]; then
        echo "✅ 依赖安装完成"
    else
        echo "⚠️ 依赖安装遇到问题，请参考 INSTALL.md"
    fi
fi

# 检查 ffmpeg
if ! command -v ffmpeg &> /dev/null; then
    echo "⚠️ ffmpeg 未安装，请参考 INSTALL.md 安装"
fi

# 运行 handler
echo ""
echo "Starting handler..."
python3 handler.py