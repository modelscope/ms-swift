CUDA_VISIBLE_DEVICES=0 \
swift deploy \
    --model OpenGVLab/InternVL2-8B \
    --hub_token bed37917-9190-45cd-9018-0f67f1924051 \
    --infer_backend lmdeploy
