# Copyright (c) ModelScope Contributors. All rights reserved.
import collections
import copy
import inspect
import os
import signal
import subprocess
import sys
import tempfile
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import unittest
from datetime import timedelta
from pathlib import Path
from tokenizers import Tokenizer, models
from torch.distributed import init_device_mesh
from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForSequenceClassification
from transformers import Trainer as HfTrainer
from transformers import TrainingArguments
from types import SimpleNamespace
from unittest.mock import patch

_CPU_TEST_SOURCE = 'SWIFT_TEST_SEQ_CLS_SP_CPU_SOURCE'
_cpu_source = os.environ.get(_CPU_TEST_SOURCE)
# NPU model patches are applied at the first Swift import and cannot be undone.
# Disable them only inside the isolated CPU test process (including spawn workers).
_import_argv = [sys.argv[0], '--enable_npu_model_patch', 'false', *sys.argv[1:]] if _cpu_source else sys.argv
with patch.object(sys, 'argv', _import_argv):
    from swift.metrics import MeanMetric
    from swift.model import get_llm_model
    from swift.model.model_meta import ModelInfo
    from swift.model.register import MODEL_MAPPING
    from swift.sequence_parallel import sequence_parallel
    from swift.template import get_template
    from swift.trainers import Trainer

if _cpu_source:
    assert Path(inspect.getfile(Trainer)).resolve().parents[2] == Path(_cpu_source)


def _make_template(config, parallel_size):
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(models.WordLevel({
            '[PAD]': 0,
            '[UNK]': 1
        }, unk_token='[UNK]')),
        pad_token='[PAD]',
        unk_token='[UNK]')
    tokenizer.model_info = ModelInfo(
        model_type='qwen2',
        model_dir='',
        torch_dtype=torch.float32,
        max_model_len=128,
        quant_method=None,
        quant_bits=0,
        config=config,
        task_type='seq_cls')
    tokenizer.model_meta = MODEL_MAPPING['qwen2']
    template = get_template(tokenizer, template_type='dummy', sequence_parallel_size=parallel_size)
    template.set_mode('train')
    return template


def _check_multilabel_targets(tmp_path, patch_attribute, parallel_size, num_labels):
    config = Qwen2Config(
        vocab_size=16,
        hidden_size=16,
        intermediate_size=24,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        num_labels=num_labels,
        pad_token_id=0,
        problem_type='multi_label_classification',
        attn_implementation='eager')
    template = _make_template(config, parallel_size)
    rows = [
        {
            'input_ids': [2, 3, 2, 3, 2, 3],
            'labels': [0]
        },
        {
            'input_ids': [3, 2, 3, 2],
            'labels': [num_labels - 1]
        },
    ]
    batch = template.data_collator(copy.deepcopy(rows))
    labels = batch['labels'].clone()
    torch.manual_seed(42)
    model = Qwen2ForSequenceClassification(config)
    reference = copy.deepcopy(model)
    # Use the real HF device preparation and Swift loss path without unrelated
    # checkpoint/tokenizer setup or requiring an attention kernel on a GPU.
    trainer = Trainer.__new__(Trainer)
    args = TrainingArguments(output_dir=str(tmp_path), use_cpu=True, report_to=[], disable_tqdm=True)
    HfTrainer.__init__(trainer, model=model, args=args)
    trainer.template = template
    trainer.task_type = 'seq_cls'
    trainer.problem_type = config.problem_type
    trainer.custom_metrics = {'train': collections.defaultdict(lambda: MeanMetric(nan_value=None, device='cpu'))}
    for name, value in {
            'world_size': parallel_size,
            'sp_world_size': parallel_size,
            'rp_world_size': 1,
            'tokenizer': template.tokenizer,
            'device_mesh': None,
            'extra_kwargs': {},
    }.items():
        patch_attribute(sequence_parallel, name, value)

    prepared = trainer._prepare_inputs(copy.deepcopy(batch))
    torch.testing.assert_close(prepared['labels'], labels)
    assert prepared['labels'].dtype == labels.dtype
    loss = trainer.compute_loss(model, prepared)
    reference_loss = reference(**batch).loss
    torch.testing.assert_close(loss, reference_loss)
    loss.backward()
    reference_loss.backward()
    for actual, expected in zip(model.parameters(), reference.parameters()):
        torch.testing.assert_close(actual.grad, expected.grad)


def _check_one_dimensional_targets(patch_attribute, task_type):
    trainer = Trainer.__new__(Trainer)
    trainer.args = SimpleNamespace(device=torch.device('cpu'))
    trainer.is_deepspeed_enabled = False
    trainer.template = SimpleNamespace(sequence_parallel_size=2)
    trainer.task_type = task_type
    patch_attribute(sequence_parallel, 'world_size', 2)
    patch_attribute(sequence_parallel, 'sp_world_size', 2)
    patch_attribute(sequence_parallel, 'rp_world_size', 1)
    patch_attribute(sequence_parallel, 'device_mesh', None)
    patch_attribute(sequence_parallel, 'extra_kwargs', {})
    inputs = {'input_ids': torch.ones(2, 6, dtype=torch.long), 'labels': torch.tensor([0, 1])}
    torch.testing.assert_close(trainer._prepare_inputs(inputs)['labels'], torch.tensor([0, 1]))


def _check_token_labels(patch_attribute):
    trainer = Trainer.__new__(Trainer)
    trainer.args = SimpleNamespace(device=torch.device('cpu'))
    trainer.is_deepspeed_enabled = False
    trainer.template = SimpleNamespace(sequence_parallel_size=2)
    trainer.task_type = 'causal_lm'
    for name, value in {
            'world_size': 2,
            'sp_world_size': 2,
            'rp_world_size': 1,
            'tokenizer': SimpleNamespace(pad_token_id=0),
            'device_mesh': None,
            'extra_kwargs': {},
    }.items():
        patch_attribute(sequence_parallel, name, value)
    inputs = {
        'input_ids': torch.tensor([[1, 2, 3, 4, 5]]),
        'labels': torch.tensor([[-100, 2, 3, 4, 5]]),
        'position_ids': torch.arange(5).unsqueeze(0),
    }
    prepared = trainer._prepare_inputs(inputs)
    torch.testing.assert_close(prepared['labels'], torch.tensor([[2, 3, 4]]))


def _check_distributed_multilabel(rank, rendezvous, output_dir, trainer_source):
    assert inspect.getfile(Trainer) == trainer_source
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method=rendezvous, rank=rank, world_size=2, timeout=timedelta(seconds=60))
    try:
        config = Qwen2Config(
            vocab_size=16,
            hidden_size=16,
            intermediate_size=24,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            num_labels=4,
            pad_token_id=0,
            problem_type='multi_label_classification',
            attn_implementation='sdpa')
        template = _make_template(config, 2)
        batch = template.data_collator([
            {
                'input_ids': [2, 3, 2, 3, 2, 3],
                'labels': [0, 2]
            },
            {
                'input_ids': [3, 2, 3, 2],
                'labels': [1, 3]
            },
        ])
        torch.manual_seed(42)
        model = Qwen2ForSequenceClassification(config)
        model.model_meta = template.model_meta
        model.model_info = template.model_info
        reference = copy.deepcopy(model)
        expected = reference(**batch)
        trainer = Trainer.__new__(Trainer)
        args = TrainingArguments(output_dir=output_dir, use_cpu=True, report_to=[], disable_tqdm=True)
        HfTrainer.__init__(trainer, model=model, args=args)
        trainer.template = template
        trainer.task_type = 'seq_cls'
        trainer.problem_type = config.problem_type
        trainer.custom_metrics = {'train': collections.defaultdict(lambda: MeanMetric(nan_value=None, device='cpu'))}
        sp = sequence_parallel
        sp.world_size = sp.sp_world_size = 2
        sp.dp_world_size = sp.rp_world_size = 1
        # Select CPU explicitly; get_device() may otherwise pick MPS on macOS.
        sp.device_mesh = init_device_mesh('cpu', (1, 2), mesh_dim_names=('data', 'sequence'))
        sp.padding_free = False
        sp.tokenizer = template.tokenizer
        sp.model_dtype = torch.float32
        base_model = get_llm_model(model, model_meta=template.model_meta)
        sp._prepare_flash_attn(base_model)
        sp._prepare_forward_hook(base_model)
        trainer._patch_tasks()
        prepared = trainer._prepare_inputs(copy.deepcopy(batch))
        loss, output = trainer.compute_loss(model, prepared, return_outputs=True)
        torch.testing.assert_close(prepared['labels'], batch['labels'])
        torch.testing.assert_close(output.logits, expected.logits, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(loss, expected.loss, rtol=1e-5, atol=1e-6)
        loss.backward()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    finally:
        dist.destroy_process_group()


class TestSequenceClassificationLabels(unittest.TestCase):

    def setUp(self):
        temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(temporary_directory.cleanup)
        self.tmp_path = Path(temporary_directory.name)

    def _patch_attribute(self, target, name, value):
        patcher = patch.object(target, name, value)
        patcher.start()
        self.addCleanup(patcher.stop)

    def _run_in_cpu_process(self):
        if _cpu_source:
            return False
        env = os.environ.copy()
        source = str(Path(inspect.getfile(Trainer)).resolve().parents[2])
        env[_CPU_TEST_SOURCE] = source
        env['PYTHONPATH'] = os.pathsep.join(filter(None, (source, env.get('PYTHONPATH'))))
        command = [sys.executable, str(Path(__file__).resolve()), f'{type(self).__name__}.{self._testMethodName}']
        process = subprocess.Popen(
            command,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=os.name == 'posix')
        try:
            output, _ = process.communicate(timeout=180)
        except subprocess.TimeoutExpired:
            # Include the Gloo workers in cleanup if their parent is stuck.
            if os.name == 'posix':
                os.killpg(process.pid, signal.SIGKILL)
            else:
                process.kill()
            output, _ = process.communicate()
            self.fail(f'CPU test subprocess timed out:\n{output}')
        self.assertEqual(process.returncode, 0, output)
        return True

    def test_multilabel_targets(self):
        if self._run_in_cpu_process():
            return
        for parallel_size in (1, 2):
            for num_labels in (1, 3, 4, 6, 8):
                _check_multilabel_targets(self.tmp_path, self._patch_attribute, parallel_size, num_labels)

    def test_one_dimensional_targets(self):
        for task_type in ('seq_cls', 'reranker', 'embedding'):
            _check_one_dimensional_targets(self._patch_attribute, task_type)

    def test_token_labels(self):
        _check_token_labels(self._patch_attribute)

    @unittest.skipUnless(dist.is_available() and dist.is_gloo_available(), 'Gloo is not available')
    def test_multilabel_loss_with_two_sequence_parallel_ranks(self):
        if self._run_in_cpu_process():
            return
        mp.spawn(
            _check_distributed_multilabel,
            args=((self.tmp_path / 'rendezvous').as_uri(), str(self.tmp_path), inspect.getfile(Trainer)),
            nprocs=2)


if __name__ == '__main__':
    unittest.main()
