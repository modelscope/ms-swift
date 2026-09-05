# Copyright (c) ModelScope Contributors. All rights reserved.
"""One row per recipe entry point: does it train five steps from scratch and save a checkpoint?

Why a table rather than a file per task: the training recipes differ only in which Configs they
accept and what shape their data is. ``swift/dev/recipe`` exposes seventeen entry points and before
this file three of them had any coverage, so the point of the table is that the gap is visible and
closing it is one row, not one file.

Rows bind to entry points by *signature*, not by a hand-written argument list: ``RecipeHarness``
offers a pool of Configs keyed by parameter name and passes only what the callee declares. A recipe
that grows a new Config picks it up for free; one that grows a new REQUIRED parameter fails loudly
with its name instead of raising ``TypeError`` from inside.

Everything here is ``slow``: each row loads a model and runs an optimizer. The assertions are about
wiring, not numbers -- the one numeric claim, that a randomly-initialised model starts at
``ln(vocab_size)``, is exactly the claim a mis-shifted or double-normalised loss breaks.
"""
import inspect
import json
import os
from dataclasses import dataclass, field
from typing import Optional

import pytest

from swift.dev.tests.tiny import TinyData, TinyModel


@dataclass(frozen=True)
class Case:
    """A recipe plus the smallest inputs it accepts."""

    name: str
    entry: str
    data: str = 'sft'
    model: dict = field(default_factory=dict)
    template: dict = field(default_factory=dict)
    train: dict = field(default_factory=dict)
    rlhf: Optional[dict] = None
    #: A randomly-initialised causal LM starts at ln(vocab). Heads that pool per sequence
    #: (seq_cls / embedding / reranker) and pairwise objectives (the dpo family) do not.
    initial_loss: bool = False


class RecipeHarness:
    """Turn a ``Case`` into a real run, then check the run happened."""

    STEPS = 5
    #: ln(vocab) holds in expectation; the first logged value is already one step in, on random data.
    LOSS_TOLERANCE = 1.0

    @staticmethod
    def pool(case: Case, model_dir: str, data_path: str, out_dir: str) -> dict:
        from swift.dev.config import (CheckpointConfig, DatasetConfig, DistributedConfig, GenerationConfig,
                                      ModelConfig, RLHFConfig, RolloutConfig, SamplingConfig, TemplateConfig,
                                      TrainConfig)
        pool = {
            'model_config':
            ModelConfig(model=model_dir, model_type=TinyModel.MODEL_TYPE, torch_dtype='bfloat16', **case.model),
            'template_config':
            TemplateConfig(template=TinyModel.TEMPLATE, max_length=256, **case.template),
            'dataset_config':
            DatasetConfig(dataset=[data_path]),
            'train_config':
            TrainConfig(
                learning_rate=1e-4, per_device_train_batch_size=2, max_steps=RecipeHarness.STEPS, **case.train),
            'distributed_config':
            DistributedConfig(),
            'checkpoint_config':
            CheckpointConfig(),
            'generation_config':
            GenerationConfig(max_new_tokens=8),
            'rollout_config':
            RolloutConfig(),
            'sampling_config':
            SamplingConfig(),
            'tuner_config':
            None,
            'output_dir':
            out_dir,
        }
        if case.rlhf is not None:
            pool['rlhf_config'] = RLHFConfig(**case.rlhf)
        RecipeHarness.derive(pool)
        return pool

    @staticmethod
    def derive(pool: dict) -> None:
        """Run the same cross-Config derivation the CLI runs, in place.

        Without it a row only fails on whatever the recipe happens to check first -- dpo and kto stop
        at ``RLHFConfig.ref_model is None``, which ``_derive_rlhf_ref_model`` fills from the policy
        model. Constructing Configs by hand and skipping this step tests a path no user takes.
        """
        from swift.dev.config.process import process_configs

        params = inspect.signature(process_configs).parameters
        process_configs(**{name: pool[name] for name in params if name in pool})

    @staticmethod
    def call(entry: str, pool: dict):
        """Bind the pool to ``entry``'s signature, so a Config the recipe does not accept is not sent."""
        from swift.dev import recipe

        fn = getattr(recipe, entry)
        params = inspect.signature(fn).parameters
        missing = [
            name for name, p in params.items()
            if p.default is inspect.Parameter.empty and p.kind is not p.VAR_KEYWORD and name not in pool
        ]
        assert not missing, f'{entry} requires {missing}, which RecipeHarness.pool does not provide'
        return fn(**{name: pool[name] for name in params if name in pool})

    @staticmethod
    def assert_trained(case: Case, history, model_dir: str, out_dir: str) -> None:
        assert history, f'{case.name}: no optimizer steps were taken'
        losses = [row['loss'] for row in history if 'loss' in row]
        assert len(losses) == RecipeHarness.STEPS, \
            f'{case.name}: expected {RecipeHarness.STEPS} logged steps, got {len(losses)}: {losses}'
        assert all(loss == loss and abs(loss) != float('inf') for loss in losses), \
            f'{case.name}: non-finite loss: {losses}'
        if case.initial_loss:
            expected = TinyModel.initial_loss(model_dir)
            assert abs(losses[0] - expected) < RecipeHarness.LOSS_TOLERANCE, \
                f'{case.name}: first loss {losses[0]:.3f} is not ln(vocab)={expected:.3f} -- labels are misaligned'

        ckpt = os.path.join(out_dir, 'checkpoint-final')
        assert os.path.isdir(ckpt), f'{case.name}: no checkpoint-final in {sorted(os.listdir(out_dir))}'
        files = set(os.listdir(ckpt))
        assert any(f.endswith('.safetensors') for f in files), f'{case.name}: no weights saved: {sorted(files)}'
        assert 'args.json' in files, f'{case.name}: checkpoint is not self-describing: {sorted(files)}'
        with open(os.path.join(ckpt, 'args.json')) as f:
            args = json.load(f)
        for key in ('model_type', 'template', 'swift_version'):
            assert key in args, f'{case.name}: args.json missing {key!r}: {sorted(args)}'


#: The training recipes. The rlhf rows all land in ``run_dpo``, which dispatches on ``rlhf_type``.
TRAINING = [
    Case('sft', 'run_sft', initial_loss=True),
    Case('pt', 'run_sft', data='pretrain', template={'use_chat_template': False}, initial_loss=True),
    Case('dpo', 'run_dpo', data='preference', rlhf={'rlhf_type': 'dpo'}),
    # kto takes paired data too: run_dpo rejects a label-only row with "rlhf_type=kto needs paired
    # chosen/rejected data", so the KTO desirable/undesirable format is not what this path reads.
    Case('kto', 'run_dpo', data='preference', rlhf={'rlhf_type': 'kto'}),
    Case('cpo', 'run_dpo', data='preference', rlhf={'rlhf_type': 'cpo'}),
    Case('orpo', 'run_dpo', data='preference', rlhf={'rlhf_type': 'orpo'}),
    Case('simpo', 'run_dpo', data='preference', rlhf={'rlhf_type': 'simpo'}),
    Case('rm', 'run_dpo', data='preference', rlhf={'rlhf_type': 'rm'}),
    Case('seq_cls', 'run_seq_cls', data='seq_cls',
         model={'task_type': 'seq_cls', 'num_labels': 2, 'problem_type': 'single_label_classification'}),
    Case('embedding', 'run_embedding', data='embedding', model={'task_type': 'embedding'}),
]


@pytest.mark.slow
@pytest.mark.accel(1)
@pytest.mark.parametrize('case', TRAINING, ids=[c.name for c in TRAINING])
def test_trains_five_steps_from_scratch_and_saves(case, tmp_path):
    """Five optimizer steps on a 4-layer random model, then a checkpoint ``swift infer`` could read."""
    model_dir = TinyModel.build(tmp_path / 'model')
    data_path = getattr(TinyData, case.data)(tmp_path / f'{case.name}.jsonl')
    out_dir = str(tmp_path / 'out')

    history = RecipeHarness.call(case.entry, RecipeHarness.pool(case, model_dir, data_path, out_dir))
    RecipeHarness.assert_trained(case, history, model_dir, out_dir)
