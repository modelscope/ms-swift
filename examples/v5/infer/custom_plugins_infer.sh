# v5 infer wiring EVERY hand-writable plugin kind from examples/v5/infer/custom_plugins.py at once.
#
# --external_plugins imports custom_plugins.py before anything is built, so all five name-selected kinds
# registered there are available; --sampler points at a SECOND file (custom_sampler.py) because a custom
# sampler is resolved by source, not by name (its kind is declared lazily -- see that file's docstring).
#
#   --model_type demo_qwen2      model loader  (@register_model ModelLoader)     [custom_plugins.py]
#   --template   demo_chatml     template      (register_template TemplateMeta)  [custom_plugins.py]
#   --dataset    demo_synthetic  dataset loader(@register_dataset DatasetLoader) [custom_plugins.py]
#   --orm length_bonus async_length_bonus  reward: one SYNC + one ASYNC rule, mixed in one channel
#                                (async __call__ is gathered, not awaited one-by-one) [custom_plugins.py]
#   --tools word_count           tool          (ToolPlugin.build)                [custom_plugins.py]
#   --sampler .../custom_sampler.py  sampler    (a twinkle Sampler subclass)      [custom_sampler.py]
#
# --model still names the real checkpoint the demo_qwen2 loader covers; --model_type only selects WHICH
# registered family loads it. Tools are inherently multi-turn (the model calls word_count, reads the
# observation, then answers), so --max_turns is required alongside --tools. --num_samples 4 draws a
# best-of-4 group per prompt and --output_format all stores every candidate plus each reward's score, so
# both the sync and the async reward land in the row's `scores`.
#
# A single card is enough here: both rewards are pure FUNCTIONS (no GPU-resident reward model), so no
# dedicated DeviceGroup and no --mode ray are needed. Swap `--sampler` back to `vllm` for throughput.
CUDA_VISIBLE_DEVICES=0 \
USE_SWIFT_V5=1 \
swift infer \
    --model Qwen/Qwen2.5-0.5B-Instruct \
    --model_type demo_qwen2 \
    --template demo_chatml \
    --external_plugins examples/v5/infer/custom_plugins.py \
    --dataset demo_synthetic \
    --sampler examples/v5/infer/custom_sampler.py \
    --orm length_bonus async_length_bonus \
    --tools word_count \
    --max_turns 4 \
    --sandbox_num_envs 1 \
    --num_samples 4 \
    --output_format all \
    --max_new_tokens 256 \
    --temperature 1.0 \
    --result_path ./output/custom_plugins_infer.jsonl
