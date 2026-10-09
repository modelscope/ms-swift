# v5 RERANKER inference: score query/passage relevance with a cross-encoder.
#
# USE_SWIFT_V5=1 routes `swift infer` to the v5 dev CLI. A reranker is a cross-encoder
# (AutoModelForSequenceClassification, num_labels=1): it reads a (query, passage) PAIR and emits one
# relevance score, not text. `--task_type reranker` selects that head (bge_reranker pins it, so the flag
# is optional here but kept explicit); `--sampler transformers` forwards each pair through HF
# (`_forward_via_transformers`). On a pooling-capable engine (`--sampler vllm`/`sglang`) the same command
# runs through the engine's own `encode` instead.
#
# DATA SHAPE (examples/infer/data/reranker.jsonl; run this script from the repo root): one row per
# (query, passage) pair, encoded as messages = [user=query, assistant=passage]. The bge_reranker template
# has NO system slot -- a `system` turn trips `check_system`'s assert -- so keep the pair in user/assistant
# only. Inference mode encodes the anchor (the whole conversation) directly; it does NOT need the
# train-mode positive_messages/negative_messages.
#
# OUTPUT: `--reranker_use_activation` defaults True, so each row's `response` is the raw score passed
# through a sigmoid and bounded to [0, 1] (higher = more relevant); `responses` wraps that single value in
# a list. Set `--reranker_use_activation false` to get the unbounded logit instead.
#
# A decoder-only reranker (Qwen3-Reranker) is a DIFFERENT task_type -- `generative_reranker` -- scored off
# the first generated token's yes/no logprob gap; it goes through the generation path, not this pooling one.
#
# Real inference scaffold, smoke-verified end to end. For a fast smoke keep --task_type/--sampler and shrink
# only the bounded dims (dataset slice, --max_batch_size, a smaller reranker variant).
CUDA_VISIBLE_DEVICES=0 \
USE_SWIFT_V5=1 \
swift infer \
    --model BAAI/bge-reranker-base \
    --task_type reranker \
    --sampler transformers \
    --dataset examples/infer/data/reranker.jsonl \
    --max_batch_size 8 \
    --output_format all \
    --result_path ./output/reranker_infer.jsonl
