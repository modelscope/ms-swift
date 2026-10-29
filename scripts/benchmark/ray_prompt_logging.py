# Copyright (c) ModelScope Contributors. All rights reserved.
"""Offline CPU replay of completion logging against an unchanged source file."""
import argparse
import importlib.util
import json
import platform
import statistics
import sys
import tempfile
import time
import tokenizers
import transformers
from pathlib import Path
from tokenizers import Tokenizer, decoders, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast
from types import SimpleNamespace

from swift.ray.megatron.base_trainer import BaseRayTrainer
from swift.rl_core.data import OnPolicySample
from swift.template import get_template


class MemoryWriter:

    def append(self, rows):
        self.rows = rows


def make_template():
    vocab = {char: i for i, char in enumerate(sorted(pre_tokenizers.ByteLevel.alphabet()))}
    backend = Tokenizer(models.BPE(vocab=vocab, merges=[]))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        eos_token='<|im_end|>',
        pad_token='<|endoftext|>',
        additional_special_tokens=['<|im_start|>', '<|im_end|>'])
    processor = SimpleNamespace(
        tokenizer=tokenizer,
        model_info=SimpleNamespace(config={}, task_type='causal_lm', max_model_len=32768),
        model_meta=SimpleNamespace(is_multimodal=False))
    template = get_template(processor, template_type='qwen2_5')
    template.set_mode('train')
    return template


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True, help='Unchanged base_trainer.py source')
    parser.add_argument('--rounds', type=int, default=6)
    parser.add_argument('--generations', type=int, choices=[1, 16], default=16, help='Use 1 for the no-reuse control')
    args = parser.parse_args()
    if args.rounds < 1:
        parser.error('--rounds must be positive')
    name = 'swift.ray.megatron._logging_baseline'
    spec = importlib.util.spec_from_file_location(name, args.baseline)
    baseline = importlib.util.module_from_spec(spec)
    sys.modules[name] = baseline
    spec.loader.exec_module(baseline)

    template = make_template()
    samples = []
    num_prompts = 512 // args.generations
    for prompt_id in range(num_prompts):
        text = f'Prompt {prompt_id}. ' + 'Check the toolbar and dialog before choosing the described control. ' * 32
        for response_id in range(args.generations):
            samples.append(
                OnPolicySample(
                    prompt_id=str(prompt_id),
                    messages=[{
                        'role': 'user',
                        'content': text
                    }, {
                        'role': 'assistant',
                        'content': f'click({prompt_id}, {response_id})'
                    }]))
    rewards = [float(i % 7) for i in range(len(samples))]
    trainers = {}
    counts = {}
    for label, cls in [('baseline', baseline.BaseRayTrainer), ('candidate', BaseRayTrainer)]:
        trainer = cls.__new__(cls)
        trainer.args = SimpleNamespace(log_completions=True, num_generations=args.generations)
        trainer.template = template
        trainer._completions_writer = MemoryWriter()
        original = trainer._format_log_prompt
        calls = 0

        def count_format(messages):
            nonlocal calls
            calls += 1
            return original(messages)

        trainer._format_log_prompt = count_format
        trainer._maybe_log_completions(samples, rewards=rewards, gen_step=3)
        counts[label] = calls
        del trainer._format_log_prompt
        trainers[label] = trainer
    assert counts == {'baseline': 512, 'candidate': num_prompts}, counts
    assert trainers['baseline']._completions_writer.rows == trainers['candidate']._completions_writer.rows
    assert all(row['prompt'].startswith('<|im_start|>') for row in trainers['candidate']._completions_writer.rows)

    timings = {label: [] for label in trainers}
    for index in range(args.rounds):
        order = list(trainers) if index % 2 == 0 else list(reversed(trainers))
        for label in order:
            start = time.perf_counter()
            trainers[label]._maybe_log_completions(samples, rewards=rewards, gen_step=3)
            timings[label].append(time.perf_counter() - start)
        assert trainers['baseline']._completions_writer.rows == trainers['candidate']._completions_writer.rows

    with tempfile.TemporaryDirectory() as output_dir:
        contents = []
        for label, trainer in trainers.items():
            trainer.args.output_dir = str(Path(output_dir) / label)
            trainer._completions_writer = None
            trainer._maybe_log_completions(samples, rewards=rewards, gen_step=3)
            contents.append((Path(trainer.args.output_dir) / 'completions.jsonl').read_bytes())
        assert contents[0] == contents[1]
    medians = {label: statistics.median(values) for label, values in timings.items()}
    print(
        json.dumps(
            {
                'python': platform.python_version(),
                'platform': platform.platform(),
                'transformers': transformers.__version__,
                'tokenizers': tokenizers.__version__,
                'fixture': 'Synthetic text prompts/responses; offline Swift qwen2_5 byte tokenizer',
                'prompts': num_prompts,
                'generations_per_prompt': args.generations,
                'timed_scope': 'CPU log row construction with an in-memory writer; excludes disk I/O',
                'warmup_calls_per_variant': 1,
                'rounds': args.rounds,
                'format_calls_per_batch': counts,
                'seconds': timings,
                'median_seconds': medians,
                'speedup': medians['baseline'] / medians['candidate'],
                'rows_equal': True,
                'jsonl_bytes_equal': True,
            },
            indent=2))


if __name__ == '__main__':
    main()
