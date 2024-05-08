pip install auto-gptq

checkpoint=/cephfs/group/teg-openrecom-openrc/starxhong/swift/output/deepseek-vl-7b-chat/v39-20240428-234401/checkpoint-3090
local_repo_path=/cephfs/group/teg-openrecom-openrc/starxhong/feature_eval/model/DeepSeek-VL
custom_train_dataset_path=/cephfs/group/teg-openrecom-openrc/starxhong/feature_eval/data/train_data/sharegpt/dataset_1231_fully_clean_0329_with_image_caption_by_deepseek_with_reason_pmt_v40_3imgs.jsonl

CUDA_VISIBLE_DEVICES=0 swift export \
    --ckpt_dir ${checkpoint} \
    --merge_lora true \
    --local_repo_path ${local_repo_path} \
    --model_type deepseek-vl-7b-chat \
    --quant_bits 4 \
    --quant_method awq \
    --custom_train_dataset_path ${custom_train_dataset_path} 
