#!/bin/bash

# PPO权重同步监控脚本
# 用于实时监控训练过程中的权重同步状态

REMOTE_VLLM_HOST="192.168.1.100"  # 修改为实际IP
REMOTE_VLLM_PORT=8000
OUTPUT_DIR="./output_ppo"

echo "🔍 PPO权重同步监控工具"
echo "================================"
echo "远程vLLM地址: $REMOTE_VLLM_HOST:$REMOTE_VLLM_PORT"
echo "训练输出目录: $OUTPUT_DIR"
echo ""

# 函数：检查vLLM服务器状态
check_vllm_status() {
    echo "📡 检查vLLM服务器状态..."
    if curl -s "http://$REMOTE_VLLM_HOST:$REMOTE_VLLM_PORT/health" > /dev/null; then
        echo "✅ vLLM服务器正常运行"
        
        # 获取服务器信息
        MODEL_INFO=$(curl -s "http://$REMOTE_VLLM_HOST:$REMOTE_VLLM_PORT/v1/models" | jq -r '.data[0].id' 2>/dev/null || echo "未知模型")
        echo "📚 当前加载模型: $MODEL_INFO"
    else
        echo "❌ vLLM服务器无法访问"
        return 1
    fi
}

# 函数：监控网络流量
monitor_network() {
    echo "🌐 监控网络流量 (Ctrl+C退出)..."
    echo "寻找指向 $REMOTE_VLLM_HOST 的权重传输..."
    
    # 检查是否安装了iftop
    if command -v iftop > /dev/null; then
        iftop -t -s 10 -i eth0 -f "host $REMOTE_VLLM_HOST"
    else
        echo "⚠️  iftop未安装，使用netstat监控连接..."
        while true; do
            CONNECTIONS=$(netstat -tn | grep "$REMOTE_VLLM_HOST:$REMOTE_VLLM_PORT" | wc -l)
            echo "$(date): 活跃连接数: $CONNECTIONS"
            sleep 5
        done
    fi
}

# 函数：监控训练日志
monitor_training_logs() {
    echo "📋 监控训练日志中的权重同步..."
    
    if [ -f "$OUTPUT_DIR/logs/train.log" ]; then
        echo "实时日志 (Ctrl+C退出):"
        tail -f "$OUTPUT_DIR/logs/train.log" | grep -E "(move_model_to_vllm|update_named_param|weight_sync|epoch)"
    else
        echo "⚠️  训练日志文件不存在: $OUTPUT_DIR/logs/train.log"
        echo "   请确保训练已开始或检查输出目录路径"
    fi
}

# 函数：分析rollout完成数据
analyze_completions() {
    echo "📊 分析rollout完成数据..."
    
    if [ -f "$OUTPUT_DIR/completions.jsonl" ]; then
        echo "最近的完成记录:"
        tail -5 "$OUTPUT_DIR/completions.jsonl" | jq -r '"\(.step) | 奖励: \(.reward // "N/A") | 长度: \(.completion | length)"' 2>/dev/null || \
        tail -5 "$OUTPUT_DIR/completions.jsonl"
        
        echo ""
        echo "统计信息:"
        TOTAL_LINES=$(wc -l < "$OUTPUT_DIR/completions.jsonl")
        echo "总完成数: $TOTAL_LINES"
        
        # 计算平均奖励趋势
        if command -v jq > /dev/null; then
            RECENT_REWARD=$(tail -10 "$OUTPUT_DIR/completions.jsonl" | jq -s 'map(.reward // 0) | add / length' 2>/dev/null)
            echo "近期平均奖励: ${RECENT_REWARD:-N/A}"
        fi
    else
        echo "⚠️  completions.jsonl文件不存在"
        echo "   请确保训练配置了 --log_completions true"
    fi
}

# 主菜单
while true; do
    echo ""
    echo "请选择监控选项:"
    echo "1) 检查vLLM服务器状态"
    echo "2) 监控网络流量"
    echo "3) 监控训练日志"
    echo "4) 分析rollout完成数据"
    echo "5) 全面监控 (推荐)"
    echo "6) 退出"
    echo ""
    read -p "输入选项 (1-6): " choice

    case $choice in
        1)
            check_vllm_status
            ;;
        2)
            monitor_network
            ;;
        3)
            monitor_training_logs
            ;;
        4)
            analyze_completions
            ;;
        5)
            echo "🚀 启动全面监控..."
            check_vllm_status
            echo ""
            analyze_completions
            echo ""
            echo "开始监控训练日志 (Ctrl+C切换到网络监控)..."
            monitor_training_logs
            ;;
        6)
            echo "👋 退出监控"
            exit 0
            ;;
        *)
            echo "❌ 无效选项，请输入1-6"
            ;;
    esac
done 