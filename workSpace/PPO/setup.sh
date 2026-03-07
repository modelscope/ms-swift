#!/bin/bash

# PPO远程Rollout环境设置脚本

echo "正在设置PPO远程Rollout环境..."

# 设置脚本权限
chmod +x PPO_14B.sh
chmod +x start_remote_vllm_server.sh
chmod +x monitor_weight_sync.sh

echo "✅ 脚本权限设置完成"

# 检查必要的包
echo "📦 检查依赖包..."

python -c "import torch; print(f'PyTorch版本: {torch.__version__}')" || echo "❌ PyTorch未安装"
python -c "import swift; print('✅ ms-swift已安装')" || echo "❌ ms-swift未安装，请运行: pip install ms-swift[llm] -U"
python -c "import transformers; print(f'✅ transformers版本: {transformers.__version__}')" || echo "❌ transformers未安装"

# 检查网络连通性（如果已设置远程IP）
if grep -q "192.168.1.100" PPO_14B.sh; then
    echo "⚠️  请修改PPO_14B.sh中的REMOTE_VLLM_HOST为实际IP地址"
fi

# 创建输出目录
mkdir -p output_ppo/logs
echo "📁 创建输出目录: output_ppo/"

# 显示使用说明
echo ""
echo "🚀 设置完成！使用步骤："
echo ""
echo "1. 在远程机器上运行:"
echo "   scp start_remote_vllm_server.sh user@remote_host:~/"
echo "   ssh user@remote_host './start_remote_vllm_server.sh'"
echo ""
echo "2. 修改PPO_14B.sh中的REMOTE_VLLM_HOST为实际IP"
echo ""
echo "3. 在本机运行:"
echo "   ./PPO_14B.sh"
echo ""
echo "4. 监控权重同步状态 (可选):"
echo "   ./monitor_weight_sync.sh"
echo ""
echo "📖 详细说明请查看: PPO_Remote_Rollout_README.md"
echo ""
echo "⚠️  重要提醒："
echo "   - MODEL_NAME是基线模型地址，训练中权重会自动更新"
echo "   - 确保网络带宽充足，建议1Gbps以上"
echo "   - 使用监控工具实时观察权重同步状态" 