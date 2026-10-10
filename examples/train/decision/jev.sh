# Typed-decision (System-1) DISTILLATION fine-tuning of autotrust/JEV-27B-VL.
#
# JEV scores ONE question per forward (serve_decide.py::s1_pass), so JevPreprocessor fans a record out
# into one row per question. It is a DISTILLATION model: `--loss_type jev_distill` is
# KL(target || model) on the activated option distribution (+ 0.5*RPS on `score`), and `target_probs`
# may be a full soft distribution (the teacher's) rather than a one-hot. `--task_type decision` routes
# to ScoringTrainer; `--attn_impl eager` matches tests/test_align/test_decision_jev_align.py.
#
# CONTINUE-TRAINING, not a fresh LoRA: JEV's decision ability lives in the shipped adapter
# `adapter_vllm/` (a backbone LoRA + the 24-slot verbalizer head packaged as an lm_head LoRA). So
# `--model_type jev` loads the base Qwen3.5-VL and builds the fp32 calibration head from
# `adapter_vllm/decision_head.json` (+ calibration.json), and `--adapters .../adapter_vllm` continues
# training THAT LoRA (is_trainable=True). A fresh `--tuner_type lora --target_modules all-linear`
# would DISCARD the trained lm_head-LoRA the head reads, so it is wrong for JEV. adapter_vllm ships no
# args.json, which is fine: `swift sft` defaults `--load_args false`, so nothing tries to read it.
#
# The head's fp32 `bias` is re-activated for joint training AUTOMATICALLY by ScoringTrainer (the
# `--trainable_parameters scoring_head` flag is a no-op in the adapter branch, so it is omitted). On
# save, ScoringTrainer._save writes the trained head beside the adapter as scoring_head.safetensors +
# scoring_head.json; the next `--adapters <checkpoint>` (or `swift infer`) reloads it via the guarded
# maybe_load_trained_scoring_head hook.
#
# sh examples/train/decision/jev.sh
# The decision LoRA lives in the model repo's `adapter_vllm/` subfolder. Resolve its local snapshot path
# with the SAME interpreter that runs `swift` (read from the swift entry-point shebang), so this works
# no matter which `python` happens to be first on PATH. USE_HF=1 -> huggingface_hub.
SWIFT_PY="$(sed -n '1s/^#!//p' "$(command -v swift)")"
JEV_ADAPTER="$("$SWIFT_PY" -c "import os; from huggingface_hub import snapshot_download; print(os.path.join(snapshot_download('autotrust/JEV-27B-VL'), 'adapter_vllm'))")"

USE_HF=1 \
CUDA_VISIBLE_DEVICES=0 \
swift sft \
    --external_plugins examples/train/decision/dataset.py \
    --model autotrust/JEV-27B-VL \
    --model_type jev \
    --task_type decision \
    --adapters "$JEV_ADAPTER" \
    --loss_type jev_distill \
    --dataset jev_decision \
    --torch_dtype bfloat16 \
    --attn_impl eager \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-4 \
    --gradient_accumulation_steps 4 \
    --gradient_checkpointing true \
    --logging_steps 1 \
    --save_steps 50 \
    --save_total_limit 2 \
    --max_length 4096 \
    --split_dataset_ratio 0 \
    --output_dir output \
    --dataloader_num_workers 1
