"""Standalone entry that runs ONE dev transformers-backend SFT shape under sequence parallelism.

Launched by feature/sft/test_e2e.py::test_run_sft_hf_sp_matches_single twice over the SAME samples:
once under ``torchrun --nproc_per_node=2`` with sequence_parallel_size=2 (--shape sp2, each rank
holds HALF of every sequence) and once as a plain single process (--shape single). SP is a pure
re-partition -- the gathered loss must equal the single-process loss on the same global batch.

Each rank writes ``{out}.rank{RANK}.json`` with its loss trajectory plus the SP plumbing facts
(mesh ulysses size, _enable_sp, sp_strategy constructed), read back off the model run_sft built.

Usage:
    python -m torch.distributed.run ... swift/dev/tests/_runners/hf_sp.py \
        --shape {sp2|single} --data DATA.jsonl --out RESULT.json --out_dir DIR [--padding_free]

Model/template default to the sibling dp runner's (Qwen2.5-0.5B + qwen2_5 via modelscope);
SP_TEST_MODEL / SP_TEST_TEMPLATE override them (e.g. a local path on an offline machine).
"""

import argparse
import os

import json

MODEL = os.environ.get("SP_TEST_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
TEMPLATE = os.environ.get("SP_TEST_TEMPLATE", "qwen2_5")


def _peak_mem_mb():
    """Peak accelerator memory (MB), NPU or CUDA; None where neither is queryable."""
    try:
        import torch

        if torch.cuda.is_available():
            return round(torch.cuda.max_memory_allocated() / 1e6, 1)
        import torch_npu  # noqa: F401

        return round(torch.npu.max_memory_allocated() / 1e6, 1)
    except Exception:
        return None


def _run(shape, data_path, out_dir, padding_free, max_length, sp_size, max_steps):
    """run_sft with sequence_parallel_size=2 (sp2) or 1 (single); see _hf_dp_runner for the
    capturing-build_model rationale (run_sft does not return the model it built)."""
    import swift.dev.builders as builders
    from swift.dev.config import (
        CheckpointConfig,
        DatasetConfig,
        DistributedConfig,
        ModelConfig,
        TemplateConfig,
        TrainConfig,
        TunerConfig,
    )
    from swift.dev.recipe import run_sft

    model_path = MODEL if os.path.isdir(MODEL) else __import__("modelscope").snapshot_download(MODEL)

    captured = {}
    orig_build_model = builders.build_model

    def capturing_build_model(*args, **kwargs):
        model = orig_build_model(*args, **kwargs)
        captured["model"] = model
        return model

    builders.build_model = capturing_build_model
    try:
        history = run_sft(
            # padding_free + SP requires a flash attention kernel (twinkle's SP strategy rejects the
            # variable-length layout otherwise); the padded path runs on the default (sdpa) kernel.
            ModelConfig(
                model=model_path, torch_dtype="bfloat16", attn_impl="flash_attention_2" if padding_free else None
            ),
            TemplateConfig(
                template=TEMPLATE,
                max_length=max_length,
                sequence_parallel_size=(sp_size if shape == "sp2" else 1),
                padding_free=padding_free,
            ),
            DatasetConfig(dataset=[data_path], dataset_shuffle=False),
            # One optimizer step on the untouched initial weights, so sp2 and single are comparable
            # without any update/scheduler compounding. Both shapes see the SAME global batch of 2:
            # under sp2 the loader's data_world_size is world/ulysses=1, so both SP ranks receive
            # identical samples and each computes half of every sequence.
            TrainConfig(
                learning_rate=1e-5,
                lr_scheduler_type="constant",
                warmup_ratio=0.0,
                per_device_train_batch_size=2,
                gradient_accumulation_steps=1,
                max_steps=max_steps,
            ),
            DistributedConfig(),
            CheckpointConfig(),
            # LoRA, mirroring the dp runner: exercises a second OptimizerGroup and keeps the run light.
            tuner_config=TunerConfig(tuner_type="lora"),
            output_dir=out_dir,
            _save_final=False,
        )
    finally:
        builders.build_model = orig_build_model

    model = captured["model"]
    mesh = model.device_mesh
    return {
        "shape": shape,
        "padding_free": padding_free,
        "rank": int(os.environ.get("RANK", "0")),
        "world_size": int(os.environ.get("WORLD_SIZE", "1")),
        "losses": [r["loss"] for r in history],
        "peak_mem_mb": _peak_mem_mb(),
        "ulysses_size": getattr(mesh, "ulysses_size", None) if mesh is not None else None,
        "data_world_size": mesh.data_world_size if mesh is not None else None,
        # mesh ranks come off a numpy-backed mesh tensor (np.int64) -- coerce or json.dump dies
        # mid-write, leaving a truncated rank file that reads back as JSONDecodeError.
        "data_rank": int(mesh.data_rank) if mesh is not None and mesh.data_rank is not None else None,
        "enable_sp": getattr(model, "_enable_sp", None),
        # sp_strategy is built lazily at the first forward; after training it must exist under sp2.
        "sp_strategy_present": getattr(model, "sp_strategy", None) is not None,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shape", choices=["sp2", "single"], required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--padding_free", action="store_true")
    parser.add_argument("--max_length", type=int, default=512)
    # SP degree when --shape sp2 (world may exceed it: torchrun x4 + --sp_size 2 is the hybrid
    # data+sequence-parallel layout, data_world_size=2).
    parser.add_argument("--sp_size", type=int, default=2)
    # >1 only for the loss-curve asset script (local/run_sp_loss_curve.sh); the pytest cases keep
    # the default single step so sp2/single compare on untouched initial weights.
    parser.add_argument("--max_steps", type=int, default=1)
    args = parser.parse_args()

    result = _run(args.shape, args.data, args.out_dir, args.padding_free, args.max_length, args.sp_size, args.max_steps)
    # EVERY rank writes: the sp2 ranks must be compared against each other (equal losses are what
    # proves the SP loss gather replicated the result), not just read off rank 0.
    with open(f"{args.out}.rank{result['rank']}.json", "w") as f:
        json.dump(result, f)
    print(f"RUNNER_DONE {result}", flush=True)


if __name__ == "__main__":
    main()
