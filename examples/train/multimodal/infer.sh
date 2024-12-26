# Perform inference using the validation set from the training phase.
# CUDA_VISIBLE_DEVICES=0,1 \
swift infer \
    --adapters output/vx-xxx/checkpoint-xxx \
    --stream true \
    --load_data_args true \
    --max_new_tokens 2048
