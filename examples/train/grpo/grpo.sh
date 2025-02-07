ACCELERATE_LOG_LEVEL=info accelerate launch --config_file ./ddp.yaml --num_processes=2 --main_process_port 29700 ./grpo_vllm.py
