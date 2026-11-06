# Copyright (c) ModelScope Contributors. All rights reserved.
import copy
import json
import numpy as np
import os
import sys
import tempfile
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import transformers
import unittest
from datetime import timedelta
from pathlib import Path
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import EvalPrediction, PreTrainedTokenizerFast
from unittest.mock import patch


def _create_model(directory):
    from transformers import Qwen3Config, Qwen3ForCausalLM
    torch.manual_seed(42)
    config = Qwen3Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=16,
        pad_token_id=0,
        eos_token_id=3)
    Qwen3ForCausalLM(config).save_pretrained(directory)
    vocabulary = {f't{i}': i for i in range(64)}
    vocabulary['<|endoftext|>'] = vocabulary.pop('t3')
    backend = Tokenizer(models.WordLevel(vocabulary, unk_token='t1'))
    backend.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token='t1', pad_token='t0', eos_token='<|endoftext|>').save_pretrained(directory)


def _rows(kind):
    rows = []
    for i, negatives in enumerate([1, 4, 0, 2, 1, 3, 1, 2]):
        row = {
            'messages': [{
                'role': 'user',
                'content': f't{4 + i} t{12 + i}'
            }],
            'positive_messages': [[{
                'role': 'user',
                'content': f't{20 + i} t{28 + i}'
            }]],
        }
        if kind == 'infonce':
            row['negative_messages'] = [[{'role': 'user', 'content': f't{36 + i} t{44 + j}'}] for j in range(negatives)]
        else:
            row['label'] = i / 8
        rows.append(row)
    # A real repeated row is part of the dataset and must remain in the metrics.
    if kind == 'infonce':
        rows[4] = copy.deepcopy(rows[0])
    return rows


class _EncodedDataset(torch.utils.data.Dataset):

    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return copy.deepcopy(self.rows[index])


def _evaluate_worker(rank, directory):
    directory = Path(directory)
    os.environ.update(
        RANK=str(rank),
        LOCAL_RANK=str(rank),
        WORLD_SIZE='2',
        LOCAL_WORLD_SIZE='2',
        MASTER_ADDR='127.0.0.1',
        MASTER_PORT='29500',
        HF_HUB_OFFLINE='1',
        TOKENIZERS_PARALLELISM='false')
    for name in list(os.environ):
        if name.startswith('INFONCE_'):
            os.environ.pop(name)
    torch.set_num_threads(1)
    dist.init_process_group(
        'gloo', init_method=f'file://{directory / "init"}', rank=rank, world_size=2, timeout=timedelta(seconds=90))
    # This fixture runs on CPU, including when the CI host also has an NPU.
    # Model patches are applied on the first Swift import, before arguments exist.
    with patch.object(sys, 'argv', [sys.argv[0], '--enable_npu_model_patch', 'false']):
        from swift.arguments import SftArguments
        from swift.metrics import eval_metrics_map
        from swift.pipelines.train.sft import SwiftSft
        from swift.trainers import EmbeddingTrainer

    try:
        for kind in ['infonce', 'paired']:
            args = SftArguments(
                model=str(directory / 'model'),
                model_type='qwen3_emb',
                task_type='embedding',
                dataset=[str(directory / 'data.jsonl')],
                loss_type='infonce' if kind == 'infonce' else 'cosine_similarity',
                output_dir=str(directory / kind),
                use_cpu=True,
                enable_npu_model_patch=False,
                device_map='cpu',
                torch_dtype='float32',
                attn_impl='eager',
                ddp_backend='gloo',
                tuner_type='full',
                max_length=64,
                per_device_eval_batch_size=2,
                max_steps=1,
                report_to=[],
                save_strategy='no',
                eval_strategy='steps',
                split_dataset_ratio=0,
                disable_tqdm=True,
                dataloader_num_workers=0,
                gradient_checkpointing=False,
                padding_free=False,
                check_model=False,
                add_version=False,
                use_hf=True)
            pipeline = SwiftSft(args)
            encoded = [pipeline.template.encode(row) for row in _rows(kind)]
            dataset = _EncodedDataset(encoded)
            trainer = EmbeddingTrainer(
                model=pipeline.model,
                args=args.training_args,
                template=pipeline.template,
                train_dataset=dataset,
                eval_dataset=dataset)
            trainer.callback_handler.on_train_begin(trainer.args, trainer.state, trainer.control)
            trainer.model.eval()
            predictions, labels = [], []
            with torch.no_grad():
                for row in encoded:
                    inputs = trainer._prepare_inputs(pipeline.template.data_collator([copy.deepcopy(row)]))
                    labels.append(inputs.pop('labels').cpu().numpy())
                    predictions.append(trainer.model(**inputs)['last_hidden_state'].cpu().numpy())
            metric = eval_metrics_map[kind](None, None)
            for count in [3, 7, 8]:
                current = _EncodedDataset(encoded[:count])
                expected_predictions = np.concatenate(predictions[:count])
                expected_labels = np.concatenate(labels[:count])
                expected = metric.compute_metrics(EvalPrediction(expected_predictions, expected_labels))
                # Repeat to cover HF resetting gather_function after each evaluation.
                for _ in range(2):
                    actual = trainer.evaluate(current)
                    for key, value in expected.items():
                        np.testing.assert_allclose(
                            actual[f'eval_{key}'], value, rtol=1e-5, atol=1e-5, err_msg=f'{kind}: count={count}, {key}')
                output = trainer.predict(current)
                np.testing.assert_allclose(output.predictions, expected_predictions, rtol=1e-5, atol=1e-6)
                np.testing.assert_array_equal(output.label_ids, expected_labels)

            trainer.args.prediction_loss_only = True
            assert 'eval_loss' in trainer.evaluate(dataset)
            trainer.args.prediction_loss_only = False

            def custom_preprocess(logits, label_ids):
                assert isinstance(logits, torch.Tensor)
                assert isinstance(label_ids, torch.Tensor)
                return logits

            # Custom hooks retain their original, flattened input contract.
            trainer.preprocess_logits_for_metrics = custom_preprocess
            trainer.evaluate(dataset)

            # Older Transformers loops have no overridable gather_function hook.
            trainer.preprocess_logits_for_metrics = metric.preprocess_logits_for_metrics
            trainer._supports_embedding_gather = False
            output = trainer.predict(_EncodedDataset(encoded[:3]))
            assert isinstance(output.predictions, np.ndarray)
    finally:
        dist.destroy_process_group()


@unittest.skipUnless(dist.is_available() and dist.is_gloo_available(), 'Gloo is required')
@unittest.skipUnless(hasattr(transformers, 'Qwen3Config'), 'The tiny fixture requires Qwen3 support')
class TestEmbeddingEvaluation(unittest.TestCase):

    def test_excludes_sampler_padding(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            _create_model(directory / 'model')
            (directory / 'data.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in _rows('infonce')))
            mp.spawn(_evaluate_worker, args=(str(directory), ), nprocs=2, join=True)


if __name__ == '__main__':
    unittest.main()
