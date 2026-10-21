# v5 custom model + template: register a model family and a chat template from a plugin file.
#
# --external_plugins imports model.py before anything is built, so its registrations are live: a dev
# ModelLoader subclass (@register_model) for the model_type build_model resolves, a legacy
# register_model(ModelMeta) for the model_type build_template's get_model_processor validates (the two
# registries are independent), and register_template for the chat format. --model_type custom_im then
# selects that family and --template custom_im its chat format. This is the v5 replacement for legacy's
# custom register_model/register_template scripts: the file is imported through the same
# --external_plugins entry point as every other plugin (no separate custom_register_path flag any more).
CUDA_VISIBLE_DEVICES=0 \
USE_SWIFT_V5=1 \
swift sft \
    --external_plugins examples/train/custom/model_template/model.py \
    --model Qwen/Qwen2.5-0.5B-Instruct \
    --model_type custom_im \
    --template custom_im \
    --task_type causal_lm \
    --torch_dtype bfloat16 \
    --dataset 'swift/self-cognition#600' \
    --tuner lora \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --learning_rate 1e-4 \
    --lr_scheduler cosine \
    --warmup_ratio 0.05 \
    --max_length 2048 \
    --logging_steps 5 \
    --save_steps 100 \
    --save_total_limit 2 \
    --split_dataset_ratio 0.01 \
    --output_dir output \
    --report_to tensorboard
