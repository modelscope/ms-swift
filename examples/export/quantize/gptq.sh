OMP_NUM_THREADS=14 \
swift export \
  --ckpt_dir /mnt/workspace/yzhao/modelscope/swift/output/Qwen2-7B/v0-20241129-171625/checkpoint-100-merged \
  --dataset iic/ms_bench#256 \
  --quant_n_samples 16 \
  --quant_method gptq \
  --quant_bits 4
