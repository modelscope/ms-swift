"""Standalone entry that runs ONE dev transformers-backend SFT shape and reports what it measured.

Launched by test_run_sft_e2e.py::test_run_sft_hf_dp_loss_is_globally_token_weighted twice over the
SAME two samples: once under ``torchrun --nproc_per_node=2`` (--shape dp2, one sample per rank) and
once as a plain single process (--shape single, both samples in one batch). Only the split differs, so
a correctly aggregated dp2 run must report the single-process loss -- the global token-weighted mean.

Each rank writes ``{out}.rank{RANK}.json`` with its loss trajectory plus the three plumbing facts the
aggregation hangs on (mesh present, its dp world size, dp group built), read back off the model that
run_sft built.

Usage:
    python _hf_dp_runner.py --shape {dp2|single} --data DATA.jsonl --out RESULT.json --out_dir DIR
"""
import argparse
import json
import os

MODEL = 'Qwen/Qwen2.5-0.5B-Instruct'


def _run(shape, data_path, out_dir):
    """run_sft on the transformers backend, returning the trajectory + the model's dp plumbing.

    build_model is wrapped rather than reimplemented so this measures the real run_sft path: the
    model it built is the only place the mesh/dp-group facts can be read, and run_sft does not return
    it. The patch targets the module attribute because _run_sft_body imports build_model at call time.
    """
    from modelscope import snapshot_download

    import swift.dev.builders as builders
    from swift.dev.config import (CheckpointConfig, DatasetConfig, DistributedConfig, ModelConfig, TemplateConfig,
                                   TrainConfig, TunerConfig)
    from swift.dev.recipe import run_sft

    captured = {}
    orig_build_model = builders.build_model

    def capturing_build_model(*args, **kwargs):
        model = orig_build_model(*args, **kwargs)
        captured['model'] = model
        return model

    builders.build_model = capturing_build_model
    try:
        history = run_sft(
            ModelConfig(model=snapshot_download(MODEL), torch_dtype='bfloat16'),
            TemplateConfig(template='qwen2_5', max_length=512),
            DatasetConfig(dataset=[data_path], dataset_shuffle=False),
            # One optimizer step, so the reported loss is measured on the untouched initial weights
            # and the two shapes are comparable without any update/scheduler compounding. dp2 puts
            # one sample on each rank (bs=1 x 2 ranks), single puts both in one batch (bs=2) -- the
            # same two samples per step either way.
            TrainConfig(
                learning_rate=1e-5,
                lr_scheduler_type='constant',
                warmup_ratio=0.0,
                per_device_train_batch_size=(1 if shape == 'dp2' else 2),
                gradient_accumulation_steps=1,
                max_steps=1),
            DistributedConfig(),
            CheckpointConfig(),
            # LoRA on purpose: it is the DDP shape that used to crash outright without
            # ddp_find_unused_parameters, and it exercises the adapter's own OptimizerGroup (a second
            # group, built after __init__) rather than only the default one.
            tuner_config=TunerConfig(tuner_type='lora'),
            output_dir=out_dir,
            _save_final=False)
    finally:
        builders.build_model = orig_build_model

    model = captured['model']
    mesh = model.device_mesh
    group = model.optimizer_group[model.active_group]
    return {
        'shape': shape,
        'rank': int(os.environ.get('RANK', '0')),
        'losses': [r['loss'] for r in history],
        'device_mesh_present': mesh is not None,
        'dp_world_size': mesh._get_dp_fsdp_world_size() if mesh is not None else None,
        'dp_group_present': group._dp_group is not None,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--shape', choices=['dp2', 'single'], required=True)
    parser.add_argument('--data', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--out_dir', required=True)
    args = parser.parse_args()

    result = _run(args.shape, args.data, args.out_dir)
    # EVERY rank writes: the dp2 ranks must be compared against each other (equal values are what
    # proves the gather happened), not just read off rank 0.
    with open(f'{args.out}.rank{result["rank"]}.json', 'w') as f:
        json.dump(result, f)
    print(f'RUNNER_DONE {result}', flush=True)


if __name__ == '__main__':
    main()
