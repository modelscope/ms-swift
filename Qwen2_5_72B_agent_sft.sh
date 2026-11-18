#!/bin/bash
# ============================================================
# 多节点分布式训练启动脚本（交互式密码输入）
# 
# 用法:
#   bash launch_train.sh \
#       --ips "10.0.0.8,10.0.0.9,10.0.0.10,10.0.0.11" \
#       --save /path/to/save \
#       --dataset /path/to/data.jsonl
#
# ============================================================

# set -e

# ================== 全局变量（用于 cleanup）==================

declare -a NODE_IPS
declare -a PIDS
SSH_PASSWORD=""
REMOTE_USER=""
SSH_OPTS=""
TAIL_PID=""
CLEANUP_DONE=0

# ================== 清理函数（提前定义）==================

cleanup() {
    # 防止重复执行
    if [ "$CLEANUP_DONE" -eq 1 ]; then
        return
    fi
    CLEANUP_DONE=1
    
    echo ""
    echo "=============================================="
    echo "收到中断信号，正在停止所有节点..."
    echo "=============================================="
    
    # 停止 tail 进程
    if [ -n "$TAIL_PID" ]; then
        kill $TAIL_PID 2>/dev/null
    fi
    
    # 杀死本地后台 SSH 进程
    for pid in "${PIDS[@]}"; do
        if [ -n "$pid" ] && kill -0 $pid 2>/dev/null; then
            kill -9 $pid 2>/dev/null
        fi
    done
    
    # 停止所有远程节点的训练进程
    if [ ${#NODE_IPS[@]} -gt 0 ] && [ -n "$SSH_PASSWORD" ]; then
        echo ""
        echo "正在清理远程节点进程（包括 GPU 进程）..."
        
        for ip in "${NODE_IPS[@]}"; do
            echo "  清理节点: ${ip}"
            
            # 使用更强力的清理命令
            sshpass -p "${SSH_PASSWORD}" ssh ${SSH_OPTS} ${REMOTE_USER}@${ip} bash << 'REMOTE_CLEAN' 2>/dev/null &
                # 杀死所有相关进程
                pkill -9 -f torchrun 2>/dev/null
                pkill -9 -f megatron 2>/dev/null
                pkill -9 -f "python.*train" 2>/dev/null
                pkill -9 -f "python.*sft" 2>/dev/null
                
                # 杀死所有占用 GPU 的进程
                gpu_pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | sort -u)
                for pid in $gpu_pids; do
                    kill -9 $pid 2>/dev/null
                done
                
                sleep 1
REMOTE_CLEAN
        done
        
        # 等待所有清理命令完成
        echo "  等待清理完成..."
        sleep 5
        wait 2>/dev/null
        
        # 二次确认清理
        echo ""
        echo "二次确认清理..."
        for ip in "${NODE_IPS[@]}"; do
            sshpass -p "${SSH_PASSWORD}" ssh ${SSH_OPTS} ${REMOTE_USER}@${ip} \
                "pkill -9 -f torchrun; pkill -9 -f megatron; \
                 for pid in \$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null); do kill -9 \$pid 2>/dev/null; done" 2>/dev/null &
        done
        wait 2>/dev/null
        sleep 2
    fi
    
    echo ""
    echo "所有节点已停止 ✓"
    echo ""
    echo "提示: 如果显存仍未释放，请手动执行:"
    echo "  bash kill_all_nodes.sh --ips \"${NODE_IPS_STR}\" --ssh_password \"xxx\""
    echo "=============================================="
    exit 1
}

# 设置信号捕获（尽早设置）
trap cleanup SIGINT SIGTERM SIGHUP

# ================== 默认参数 ==================

NODE_IPS_STR=""
IP_FILE=""
SSH_PASSWORD=""
SAVE_PATH=""
DATASET_PATH=""
MODEL_PATH="/data_large_v2/liangxiaoyun/model_output/Qwen2.5-72B-Instruct"
LOAD_PATH="/data_large_v2/liangxiaoyun/model_output/Qwen2.5-72B-Instruct-megatron"
MASTER_PORT=29900
MICRO_BATCH_SIZE=4
GLOBAL_BATCH_SIZE=32
TP=8
PP=1
CP=8
MAX_LENGTH=64000
MAX_EPOCHS=1
SAVE_INTERVAL=50
REMOTE_USER="${USER}"

# 固定配置
NPROC_PER_NODE=8

# ================== 参数解析 ==================

print_usage() {
    echo "用法: $0 --ips <ip1,ip2,...> --save <path> --dataset <path> [选项]"
    echo ""
    echo "必需参数:"
    echo "  --ips <string>         节点IP列表，逗号分隔 (与 --ip_file 二选一)"
    echo "  --ip_file <file>       节点IP列表文件，每行一个IP (与 --ips 二选一)"
    echo "  --ssh_password <string>         节点账户密码"
    echo "  --save <path>          模型保存路径"
    echo "  --dataset <path>       训练数据路径"
    echo ""
    echo "可选参数:"
    echo "  --model <path>         模型路径 (默认: ${MODEL_PATH})"
    echo "  --load <path>          checkpoint路径 (默认: ${LOAD_PATH})"
    echo "  --port <int>           Master端口 (默认: ${MASTER_PORT})"
    echo "  --micro_bs <int>       micro batch size (默认: ${MICRO_BATCH_SIZE})"
    echo "  --global_bs <int>      global batch size (默认: ${GLOBAL_BATCH_SIZE})"
    echo "  --tensor_model_parallel_size <int>         张量并行，将模型层内参数切分到多GPU (默认: ${TP})"
    echo "  --pipeline_model_parallel_size <int>         训练轮数 (默认: ${PP})"
    echo "  --context_parallel_size <int>         训练轮数 (默认: ${CP})"
    echo "  --max_length <int>     最大序列长度 (默认: ${MAX_LENGTH})"
    echo "  --epochs <int>         训练轮数 (默认: ${MAX_EPOCHS})"
    echo "  --save_interval <int>  保存间隔步数 (默认: ${SAVE_INTERVAL})"
    echo "  --user <string>        SSH用户名 (默认: 当前用户)"
    echo "  -h, --help             显示帮助信息"
    echo ""
    echo "注意: 脚本运行时会提示输入SSH密码（密码不会显示在屏幕上）"
    echo ""
    echo "示例:"
    echo "  $0 --ips '10.0.0.8,10.0.0.9,10.0.0.10,10.0.0.11' \\"
    echo "     --save /data/output/model \\"
    echo "     --dataset /data/train.jsonl"
}

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --ips)
            NODE_IPS_STR="$2"
            shift 2
            ;;
        --ip_file)
            IP_FILE="$2"
            shift 2
            ;;
        --ssh_password)
            SSH_PASSWORD="$2"
            shift 2
            ;;
        --save)
            SAVE_PATH="$2"
            shift 2
            ;;
        --dataset)
            DATASET_PATH="$2"
            shift 2
            ;;
        --model)
            MODEL_PATH="$2"
            shift 2
            ;;
        --load)
            LOAD_PATH="$2"
            shift 2
            ;;
        --port)
            MASTER_PORT="$2"
            shift 2
            ;;
        --micro_bs)
            MICRO_BATCH_SIZE="$2"
            shift 2
            ;;
        --global_bs)
            GLOBAL_BATCH_SIZE="$2"
            shift 2
            ;;
        --tensor_model_parallel_size)
            TP="$2"
            shift 2
            ;;
        --pipeline_model_parallel_size)
            PP="$2"
            shift 2
            ;;
        --context_parallel_size)
            CP="$2"
            shift 2
            ;;
        --max_length)
            MAX_LENGTH="$2"
            shift 2
            ;;
        --epochs)
            MAX_EPOCHS="$2"
            shift 2
            ;;
        --save_interval)
            SAVE_INTERVAL="$2"
            shift 2
            ;;
        --user)
            REMOTE_USER="$2"
            shift 2
            ;;
        -h|--help)
            print_usage
            exit 0
            ;;
        *)
            echo "错误: 未知参数 $1"
            print_usage
            exit 1
            ;;
    esac
done

# ================== 检查 sshpass ==================

if ! command -v sshpass &> /dev/null; then
    echo "=============================================="
    echo "错误: 未安装 sshpass"
    echo "=============================================="
    echo ""
    echo "请先安装 sshpass:"
    echo "  Ubuntu/Debian: sudo apt-get install sshpass"
    echo "  CentOS/RHEL:   sudo yum install sshpass"
    echo "  MacOS:         brew install hudochenkov/sshpass/sshpass"
    echo ""
    exit 1
fi

# ================== 参数验证 ==================

# 从文件读取IP
if [ -n "$IP_FILE" ]; then
    if [ ! -f "$IP_FILE" ]; then
        echo "错误: IP文件不存在: $IP_FILE"
        exit 1
    fi
    NODE_IPS_STR=$(cat "$IP_FILE" | grep -v '^#' | grep -v '^$' | tr '\n' ',' | sed 's/,$//')
fi

# 检查必需参数
if [ -z "$NODE_IPS_STR" ]; then
    echo "错误: 必须指定 --ips 或 --ip_file"
    print_usage
    exit 1
fi

if [ -z "$SAVE_PATH" ]; then
    echo "错误: 必须指定 --save 参数"
    print_usage
    exit 1
fi

if [ -z "$DATASET_PATH" ]; then
    echo "错误: 必须指定 --dataset 参数"
    print_usage
    exit 1
fi

# 转换IP为数组
IFS=',' read -ra NODE_IPS <<< "${NODE_IPS_STR}"

if [ ${#NODE_IPS[@]} -eq 0 ]; then
    echo "错误: IP列表为空"
    exit 1
fi

# ================== SSH 配置 ==================

SSH_OPTS="-o StrictHostKeyChecking=no -o ConnectTimeout=10 -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR"

# ================== 计算分布式配置 ==================

NNODES=${#NODE_IPS[@]}
MASTER_ADDR=${NODE_IPS[0]}
WORLD_SIZE=$((NNODES * NPROC_PER_NODE))

# ================== 打印配置信息 ==================

echo ""
echo "【集群配置】"
echo "  节点数量:     ${NNODES}"
echo "  Master地址:   ${MASTER_ADDR}:${MASTER_PORT}"
echo "  每节点GPU:    ${NPROC_PER_NODE}"
echo "  总进程数:     ${WORLD_SIZE}"
echo ""
echo "【节点列表】"
for i in "${!NODE_IPS[@]}"; do
    role="Worker"
    [ $i -eq 0 ] && role="Master"
    echo "  [${i}] ${NODE_IPS[$i]} (${role})"
done
echo ""
echo "【训练配置】"
echo "  数据路径:     ${DATASET_PATH}"
echo "  保存路径:     ${SAVE_PATH}"
echo "  模型路径:     ${MODEL_PATH}"
echo "  Checkpoint:   ${LOAD_PATH}"
echo "  Micro BS:     ${MICRO_BATCH_SIZE}"
echo "  Global BS:    ${GLOBAL_BATCH_SIZE}"
echo "  TP:    ${TP}"
echo "  PP:    ${PP}"
echo "  CP:    ${CP}"
echo "  Max Length:   ${MAX_LENGTH}"
echo "  训练轮数:     ${MAX_EPOCHS}"
echo "  保存间隔:     ${SAVE_INTERVAL}"
echo "=============================================="

# ================== 检查节点连通性 ==================

echo ""
echo "检查节点连通性..."
FAILED_NODES=()
for ip in "${NODE_IPS[@]}"; do
    if sshpass -p "${SSH_PASSWORD}" ssh ${SSH_OPTS} ${REMOTE_USER}@${ip} "hostname" &>/dev/null; then
        HOSTNAME=$(sshpass -p "${SSH_PASSWORD}" ssh ${SSH_OPTS} ${REMOTE_USER}@${ip} "hostname" 2>/dev/null)
        echo "  ✓ ${ip} (${HOSTNAME})"
    else
        echo "  ✗ ${ip} 连接失败！"
        FAILED_NODES+=("$ip")
    fi
done

if [ ${#FAILED_NODES[@]} -gt 0 ]; then
    echo ""
    echo "错误: 以下节点连接失败: ${FAILED_NODES[*]}"
    echo "请检查:"
    echo "  1. 密码是否正确"
    echo "  2. 用户名是否正确 (当前: ${REMOTE_USER})"
    echo "  3. 网络是否通畅"
    echo "  4. SSH服务是否运行"
    exit 1
fi

echo ""
echo "所有节点连接成功 ✓"

# ================== 先清理所有节点的旧进程 ==================

echo ""
echo "清理所有节点的旧进程..."
for ip in "${NODE_IPS[@]}"; do
    sshpass -p "${SSH_PASSWORD}" ssh ${SSH_OPTS} ${REMOTE_USER}@${ip} \
        "pkill -9 -f torchrun 2>/dev/null; pkill -9 -f 'megatron sft' 2>/dev/null" 2>/dev/null &
done
wait
sleep 2
echo "清理完成 ✓"

# ================== 生成训练命令函数 ==================

generate_train_cmd() {
    local node_rank=$1
    
    cat << EOF
#!/bin/bash

# ========== 环境配置 ==========
export PYTHONPATH=\$PYTHONPATH:/data_large_v2/liangxiaoyun/projects/Megatron-LM
export http_proxy='http://124.223.104.128:8081'
export https_proxy='http://124.223.104.128:8081'
export MEGATRON_LM_PATH='/data_large_v2/liangxiaoyun/projects/Megatron-LM'
export MODELSCOPE_CACHE="/data_large_v2/liangxiaoyun/modelscope_cache"

# NCCL配置
export GLOO_SOCKET_IFNAME=eth0
export NCCL_SOCKET_IFNAME=eth0
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=1
export NCCL_ALGO=Ring
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# 其他配置
export TF_CPP_MIN_LOG_LEVEL=3
export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'
export NVTE_DEBUG=0

# ========== 分布式配置 ==========
export MASTER_ADDR=${MASTER_ADDR}
export MASTER_PORT=${MASTER_PORT}
export NNODES=${NNODES}
export NODE_RANK=${node_rank}
export NPROC_PER_NODE=${NPROC_PER_NODE}

# ========== 清理旧进程 ==========
pkill -9 -f torchrun 2>/dev/null || true
sleep 2

# 其他配置
export TF_CPP_MIN_LOG_LEVEL=3

echo "================ 分布式配置信息 ================"
echo "节点 ${node_rank} 启动"
echo "Master: ${MASTER_ADDR}:${MASTER_PORT}"
echo "NNODES: ${NNODES}, NODE_RANK: ${node_rank}"
echo "Hostname: \$(hostname)"
echo "=========================================="

# ========== 启动训练 ==========
export NCCL_DEBUG=WARN && export NCCL_ALGO=Ring && export GLOO_SOCKET_IFNAME=eth0 && export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' && export MODELSCOPE_CACHE='/data_large_v2/liangxiaoyun/modelscope_cache' && NVTE_DEBUG=0 /data_train/liangxiaoyun/miniconda3/envs/swift/bin/megatron sft \\
    --load ${LOAD_PATH} \\
    --model ${MODEL_PATH} \\
    --dataset ${DATASET_PATH} \\
    --save ${SAVE_PATH} \\
    --agent_template hermes \\
    --load_from_cache_file true \\
    --split_dataset_ratio 0.01 \\
    --train_type full \\
    --tensor_model_parallel_size ${TP} \\
    --pipeline_model_parallel_size ${PP} \\
    --context_parallel_size ${CP} \\
    --micro_batch_size ${MICRO_BATCH_SIZE} \\
    --global_batch_size ${GLOBAL_BATCH_SIZE} \\
    --recompute_granularity full \\
    --recompute_method uniform \\
    --recompute_num_layers 1 \\
    --max_epochs ${MAX_EPOCHS} \\
    --finetune true \\
    --cross_entropy_loss_fusion false \\
    --lr 1e-5 \\
    --lr_warmup_fraction 0.05 \\
    --min_lr 1e-6 \\
    --eval_interval ${SAVE_INTERVAL} \\
    --save_interval ${SAVE_INTERVAL} \\
    --max_length ${MAX_LENGTH} \\
    --max_position_embeddings ${MAX_LENGTH} \\
    --num_workers 8 \\
    --dataset_num_proc 8 \\
    --no_save_optim false \\
    --no_save_rng false \\
    --sequence_parallel true \\
    --use_flash_attn true \\
    --attention_backend flash \\
    --use_precision_aware_optimizer false \\
    --optimizer_cpu_offload false \\
    --log_throughput true \\
    --log_interval 1 \\
    --packing true \\
    --loss_scale ignore_empty_think
EOF
}

# ================== 创建日志目录 ==================

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="${SAVE_PATH}/logs_${TIMESTAMP}"
mkdir -p "${LOG_DIR}"

echo ""
echo "日志目录: ${LOG_DIR}"

# ================== 启动各节点 ==================

echo ""
echo "启动训练任务..."

declare -a PIDS

for i in "${!NODE_IPS[@]}"; do
    NODE_IP=${NODE_IPS[$i]}
    LOG_FILE="${LOG_DIR}/node_${i}_${NODE_IP}.log"
    
    echo "  启动节点 ${i} (${NODE_IP})..."
    
    # 生成训练命令
    TRAIN_CMD=$(generate_train_cmd $i)
    
    # 使用 sshpass 进行 SSH 连接
    sshpass -p "${SSH_PASSWORD}" ssh ${SSH_OPTS} ${REMOTE_USER}@${NODE_IP} "bash -s" <<< "${TRAIN_CMD}" > "${LOG_FILE}" 2>&1 &
    PIDS+=($!)
    
    # Master节点先启动，等待端口就绪
    if [ $i -eq 0 ]; then
        echo "  等待Master节点初始化..."
        sleep 8
    else
        sleep 1
    fi
done

# ================== 监控训练 ==================

echo ""
echo "=============================================="
echo "所有节点已启动！"
echo "=============================================="
echo ""
echo "查看各节点日志:"
for i in "${!NODE_IPS[@]}"; do
    echo "  节点${i}: tail -f ${LOG_DIR}/node_${i}_${NODE_IPS[$i]}.log"
done
echo ""
echo "╔══════════════════════════════════════════════╗"
echo "║  按 Ctrl+C 停止所有节点                       ║"
echo "╚══════════════════════════════════════════════╝"
echo ""
echo "========== Master节点实时日志 =========="

# 实时显示Master节点日志
tail -f "${LOG_DIR}/node_0_${MASTER_ADDR}.log" &
TAIL_PID=$!

# 等待所有训练进程完成
wait_result=0
for pid in "${PIDS[@]}"; do
    wait $pid 2>/dev/null || wait_result=1
done

# 停止 tail
if [ -n "$TAIL_PID" ]; then
    kill $TAIL_PID 2>/dev/null
fi

echo ""
echo "=============================================="
if [ $wait_result -eq 0 ]; then
    echo "训练完成！"
else
    echo "训练结束（部分节点可能有错误，请检查日志）"
fi
echo "日志目录: ${LOG_DIR}"
echo "模型保存: ${SAVE_PATH}"
echo "=============================================="