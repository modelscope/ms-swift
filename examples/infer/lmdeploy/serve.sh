CUDA_VISIBLE_DEVICES=0 \
swift deploy \
    --model_id_or_path OpenGVLab/InternVL2-8B \
    --infer_backend lmdeploy
