# gptq int4 quantization, 20min with A100 card and an occupation of 7GiB
# you can use multiple datasets and your local datasets
# OMP_NUM_THREADS=14 please Check issue:https://github.com/AutoGPTQ/AutoGPTQ/issues/439
OMP_NUM_THREADS=14 \
swift export \
  --ckpt_dir /mnt/workspace/yzhao/modelscope/swift/output/Qwen2-7B/v0-20241129-171625/checkpoint-100-merged \
  --dataset iic/ms_bench#256 \
  --quant_n_samples 16 \
  --quant_method gptq \
  --quant_bits 4
