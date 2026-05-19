#!/bin/bash

nvidia-smi --query-compute-apps=pid --format=csv,noheader | sort -u | while read -r pid; do
    cmd=$(ps -p "$pid" -o args= 2>/dev/null)

    if [[ "$cmd" == *"examples/llava_ov_1_5/pretrain.py"* ]]; then
        echo "Killing PID: $pid"
        echo "$cmd"
        sudo kill -9 "$pid"
    fi
done
