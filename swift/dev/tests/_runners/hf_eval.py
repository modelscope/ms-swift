"""Standalone entry that runs ONE dev transformers-backend SFT with a validation-loss eval under torchrun.

Launched by ``test_eval_sampler.py::test_val_loss_eval_under_torchrun`` under ``torchrun
--nproc_per_node=2``. Each rank drives its own copy of the loop over its shard of the eval dataloader
(the ``ShardedBatchSampler`` split), so a green run means the offline ``Evaluator.evaluate`` completed on
EVERY rank of a real distributed launch -- the launch mode the in-process single-card / Ray-driver eval
tests cannot reach (under Ray the loop runs once on the driver; under torchrun it runs per rank).

Each rank writes ``{out}.rank{RANK}.json`` with its training losses and the ``eval_loss`` values its
``evaluate()`` produced, read back off the Evaluator the run built (``run_sft`` returns only the
training history, so the Evaluator is captured the same way ``hf_dp.py`` captures the model).

Usage:
    python hf_eval.py --data DATA.jsonl --out RESULT.json --out_dir DIR
"""
import argparse
import json
import os

MODEL = 'Qwen/Qwen2.5-0.5B-Instruct'


def _run(data_path, out_dir):
    """run_sft with a val-loss eval on the transformers backend; return this rank's train + eval losses."""
    from modelscope import snapshot_download

    import importlib
    # `import ... as` would bind the FUNCTION the package __init__ re-exports under the
    # same name, not the module; importlib is explicit about wanting the module object.
    run_sft_mod = importlib.import_module('swift.dev.recipe.sft.run_sft')
    from swift.dev.config import (CheckpointConfig, DatasetConfig, DistributedConfig, ModelConfig, TemplateConfig,
                                  TrainConfig, TunerConfig)
    run_sft = run_sft_mod.run_sft

    # run_sft returns only the training history; the eval rows its evaluate() appends to a local
    # eval_history are observed by wrapping the module-level evaluate_loss it calls (the helper lives
    # in recipe.utils; run_sft binds it by name, so patching the module attribute is seen by the loop).
    captured = []
    orig_evaluate_loss = run_sft_mod.evaluate_loss

    def _capturing_evaluate_loss(*args, **kwargs):
        result = orig_evaluate_loss(*args, **kwargs)
        if result is not None:
            captured.append(result)
        return result

    run_sft_mod.evaluate_loss = _capturing_evaluate_loss
    try:
        history = run_sft(
            ModelConfig(model=snapshot_download(MODEL), torch_dtype='bfloat16'),
            TemplateConfig(template='qwen2_5', max_length=256),
            # split_dataset_ratio carves the validation set the val-loss path scores; 8 rows -> 2 val.
            DatasetConfig(dataset=[data_path], dataset_shuffle=False, split_dataset_ratio=0.25),
            TrainConfig(
                learning_rate=1e-4,
                lr_scheduler='constant',
                warmup_ratio=0.0,
                per_device_train_batch_size=1,
                per_device_eval_batch_size=1,
                gradient_accumulation_steps=1,
                eval_strategy='steps',
                eval_steps=1,
                max_steps=2,
                max_grad_norm=1.0),
            DistributedConfig(),
            CheckpointConfig(),
            tuner_config=TunerConfig(tuner='lora'),
            output_dir=out_dir,
            _save_final=False)
    finally:
        run_sft_mod.evaluate_loss = orig_evaluate_loss

    return {
        'rank': int(os.environ.get('RANK', '0')),
        'train_losses': [r['loss'] for r in history],
        'eval_losses': [e.get('eval_loss') for e in captured],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--out_dir', required=True)
    args = parser.parse_args()

    result = _run(args.data, args.out_dir)
    with open(f'{args.out}.rank{result["rank"]}.json', 'w') as f:
        json.dump(result, f)
    print(f'RUNNER_DONE {result}', flush=True)


if __name__ == '__main__':
    main()
