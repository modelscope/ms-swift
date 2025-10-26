# TODO: Script to run VLA-0 server
conda activate msswift
CUDA_VISIBLE_DEVICES=0 python vla-0/infer/server_vla0_policy.py