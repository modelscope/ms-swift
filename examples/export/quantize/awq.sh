# awq int4 quantization, about 18min with A100 card and an occupation of 13GiB
# If OOM when quantizing, reduce quant_n_samples(default 256) and quant_seqlen(default 2048)
# you can use multiple datasets and your local datasets
CUDA_VISIBLE_DEVICES=0 \
swift export \
  --ckpt_dir /mnt/workspace/yzhao/modelscope/swift/output/Qwen2-7B/v0-20241129-171625/checkpoint-100-merged \
  --dataset iic/ms_bench#256 \
  --quant_n_samples 16 \
  --quant_method awq \
  --quant_bits 4
