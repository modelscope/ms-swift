# Custom dataset format
# {"messages": [{"role": "user", "content": "Extract all information from the main body of the document image and represent it in markdown format, ignoring headers and footers. Tables should be expressed in HTML format, formulas in the document should be represented using LaTeX format, and the parsing should be organized according to the reading order."}, {"role": "assistant", "content": "<response1>"}], "images": ["ocr.png"]}

pip install git+https://github.com/huggingface/transformers@82a06db03535c49aa987719ed0746a76093b1ec4


CUDA_VISIBLE_DEVICES=0 \
swift sft \
    --model Tencent-Hunyuan/HunyuanOCR \
    --dataset 'AI-ModelScope/LaTeX_OCR:human_handwrite#20000' \
    --load_from_cache_file true \
    --split_dataset_ratio 0.01 \
    --train_type lora \
    --torch_dtype bfloat16 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --learning_rate 1e-4 \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --freeze_vit true \
    --freeze_aligner true \
    --gradient_accumulation_steps 4 \
    --eval_steps 50 \
    --save_steps 50 \
    --save_total_limit 2 \
    --logging_steps 5 \
    --max_length 4096 \
    --output_dir output \
    --warmup_ratio 0.05 \
    --dataset_num_proc 4 \
    --dataloader_num_workers 4


# CUDA_VISIBLE_DEVICES=0 \
# swift infer \
#     --adapters output/vx-xxx/checkpoint-xxx \
#     --stream true \
#     --load_data_args true
