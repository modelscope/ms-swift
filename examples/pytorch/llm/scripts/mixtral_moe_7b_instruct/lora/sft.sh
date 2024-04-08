# Experimental environment: 2 * A100
# 2 * 60GB GPU memory
PYTHONPATH=../../.. \
CUDA_VISIBLE_DEVICES=0,1 \
python llm_sft.py \
    --model_id_or_path AI-ModelScope/Mixtral-8x7B-Instruct-v0.1 \
    --model_revision master \
    --lora_alpha 32 \
    --lora_dropout_p 0.05 \
    --lora_target_modules q_proj k_proj v_proj o_proj \
    --gradient_checkpointing false \
    --batch_size 1 \
    --weight_decay 0.1 \
    --learning_rate 1e-4 \