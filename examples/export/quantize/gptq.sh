OMP_NUM_THREADS=14 \
swift export \
  --ckpt_dir output/xxx/checkpoint-xx \
  --merge_lora true \
  --dataset alpaca-en \
  --quant_method gptq
