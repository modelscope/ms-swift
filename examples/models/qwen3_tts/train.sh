# v5 models / qwen3_tts: full-parameter SFT of a Qwen3-TTS voice model.
#
# model_type = `qwen3_tts` (swift/dev/model/loader/qwen.py:1161; Qwen3TTSForConditionalGeneration,
# template `qwen3_tts`, tags audio/tts, is_multimodal). swift resolves it from --model. The TTS tower is a
# `talker` + `speaker_encoder` generator, so this is a speech-synthesis task, not text causal_lm.
#
# EXECUTION ORDER: prepare_data.py -> train.sh -> infer.py. prepare_data.py builds the speaker-conditioned
# `tts_data.parquet`; SPEAKER_NAME selects the target voice; infer.py drives generation through the
# `qwen_tts` runtime (Qwen3TTSModel.generate_custom_voice) and writes a wav -- TTS synthesis does NOT go
# through `swift infer`, which is why the inference half of this dir is infer.py, not infer.sh.
# Alternatively use a ready hub corpus: --dataset qsdong/Qwen3-1.7-TTS-SFT-Furina (slower preprocessing).
#
# REFERENCE-ONLY: not smoke-run here. The family `requires` the `qwen-tts` package (Qwen3TTSConfig /
# Qwen3TTSTokenizer / the 12Hz codec), which is NOT installed on this box, and the checkpoint is not
# cached. Install qwen-tts, run prepare_data.py, then this script as-is; the 1.7B base fits ~2 GPUs.
SPEAKER_NAME='speaker_test' \
NPROC_PER_NODE=2 \
CUDA_VISIBLE_DEVICES=0,1 \
USE_SWIFT_V5=1 \
swift sft \
    --model Qwen/Qwen3-TTS-12Hz-1.7B-Base \
    --dataset 'tts_data.parquet' \
    --split_dataset_ratio 0.01 \
    --load_from_cache_file true \
    --tuner full \
    --torch_dtype bfloat16 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --learning_rate 1e-5 \
    --gradient_accumulation_steps 1 \
    --gradient_checkpointing true \
    --eval_steps 200 \
    --save_steps 200 \
    --save_total_limit 2 \
    --logging_steps 5 \
    --max_length 4096 \
    --warmup_ratio 0.05 \
    --ddp_find_unused_parameters true \
    --dataset_num_proc 1 \
    --dataloader_num_workers 4 \
    --output_dir output
