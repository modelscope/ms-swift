# v5 EMBEDDING inference: turn each input into one pooled vector.
#
# USE_SWIFT_V5=1 routes `swift infer` to the v5 dev CLI. `--task_type embedding` selects the pooling head
# (embeddinggemma pins it, so the flag is optional here but kept explicit); `--sampler transformers`
# forwards each row through HF (`_forward_via_transformers`). On a pooling-capable engine
# (`--sampler vllm`/`sglang`) the same command runs through the engine's own `encode` instead. This is the
# inference-side counterpart of `examples/train/embedding/` -- there you FIT the vectors, here you READ them.
#
# DATA SHAPE (examples/infer/data/embedding.jsonl; run this script from the repo root): ANCHOR ONLY -- one
# row per text, messages = [user=text]. Unlike training, inference needs no positive_messages /
# negative_messages: the template encodes the single anchor. (A train-mode template would instead demand
# those pairs and IndexError -- run_infer builds the template in inference mode precisely to avoid that.)
#
# OUTPUT: each row's `response` is the pooled vector as a plain list of floats (width = the model's
# hidden_size; embeddinggemma-300m -> 768), jsonl-serialisable; `responses` wraps it in a list. Compare two
# rows' vectors (dot product / cosine) to get their similarity -- that is the whole point of an embedding.
#
# Real inference scaffold, smoke-verified end to end. For a fast smoke keep --task_type/--sampler and shrink
# only the bounded dims (dataset slice, --max_batch_size). Swap --model for any registered embedding model
# (e.g. a Qwen3-Embedding variant); a bigger hidden_size just widens the output vector.
CUDA_VISIBLE_DEVICES=0 \
USE_SWIFT_V5=1 \
swift infer \
    --model google/embeddinggemma-300m \
    --task_type embedding \
    --sampler transformers \
    --dataset examples/infer/data/embedding.jsonl \
    --max_batch_size 8 \
    --output_format all \
    --result_path ./output/embedding_infer.jsonl
