ACCELERATE_LOG_LEVEL=info accelerate launch --config_file ./zero3.yaml --num_processes=1 --main_process_port 29600 ./grpo_vllm.py
