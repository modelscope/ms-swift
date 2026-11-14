# Copyright (c) ModelScope Contributors. All rights reserved.
import torch
import torch.nn.functional as F
import unittest
from transformers import GPT2Config, GPT2LMHeadModel, T5Config, T5ForConditionalGeneration
from transformers.trainer_pt_utils import LabelSmoother
from types import SimpleNamespace

from swift.trainers.seq2seq_trainer import Seq2SeqTrainer


class _LegacyLabelSmoother:
    """Keep the pre-num_items_in_batch callable interface supported by Swift."""
    ignore_index = -100

    def __call__(self, outputs, labels, shift_labels=False):
        return LabelSmoother(epsilon=0.1)(outputs, labels, shift_labels=shift_labels)


class TestSftLabelSmoothing(unittest.TestCase):

    def _fixture(self, causal):
        torch.manual_seed(42)
        if causal:
            config = GPT2Config(
                vocab_size=16,
                n_embd=8,
                n_layer=1,
                n_head=2,
                n_positions=16,
                resid_pdrop=0,
                embd_pdrop=0,
                attn_pdrop=0,
                bos_token_id=0,
                eos_token_id=1)
            model = GPT2LMHeadModel(config)
        else:
            config = T5Config(
                vocab_size=16,
                d_model=8,
                d_ff=16,
                d_kv=4,
                num_layers=1,
                num_decoder_layers=1,
                num_heads=2,
                dropout_rate=0,
                decoder_start_token_id=0)
            model = T5ForConditionalGeneration(config)
        model.model_info = SimpleNamespace(is_moe_model=False)
        labels = torch.tensor([[7, 3, 4, -100, -100], [8, 5, 6, 3, -100], [9, 2, -100, -100, -100]])
        inputs = {'input_ids': torch.tensor([[1, 2, 3, 4, 5], [2, 3, 4, 5, 6], [3, 4, 5, 6, 7]])}
        if not causal:
            inputs['decoder_input_ids'] = model.prepare_decoder_input_ids_from_labels(labels)
        trainer = SimpleNamespace(
            model=model,
            template=SimpleNamespace(
                sequence_parallel_size=1,
                padding_free=False,
                compute_sft_loss=lambda model, inputs, **kwargs: model(**inputs)),
            label_smoother=LabelSmoother(epsilon=0.1),
            model_accepts_loss_kwargs=True,
            accelerator=SimpleNamespace(unwrap_model=lambda model: model, num_processes=1),
            _compute_acc=lambda *args, **kwargs: None,
            args=SimpleNamespace(
                use_liger_kernel=False,
                past_index=-1,
                enable_dft_loss=False,
                enable_channel_loss=False,
                average_tokens_across_devices=False,
                tuner_backend='peft',
                acc_strategy='token'))
        return model, trainer, inputs, labels

    def _reference(self, model, inputs, labels, causal):
        logits = model(**inputs).logits
        if causal:
            logits, labels = logits[:, :-1], labels[:, 1:]
        return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), label_smoothing=0.1)

    def _grads(self, model):
        return {name: p.grad.clone() for name, p in model.named_parameters() if p.grad is not None}

    def _assert_grads(self, model, expected):
        actual = self._grads(model)
        self.assertEqual(actual.keys(), expected.keys())
        for name in expected:
            torch.testing.assert_close(actual[name], expected[name], atol=2e-6, rtol=2e-5, msg=name)

    def test_accumulated_loss_and_gradients_match_full_batch(self):
        for causal in (True, False):
            model, trainer, inputs, labels = self._fixture(causal)
            expected = self._reference(model, inputs, labels, causal)
            expected.backward()
            expected_grads = self._grads(model)
            count = (labels[:, 1:] if causal else labels).ne(-100).sum()
            # Compare different microbatch partitions with unequal valid-token counts.
            for boundaries in ((0, 3), (0, 1, 3), (0, 1, 2, 3)):
                for legacy in (True, False):
                    with self.subTest(causal=causal, boundaries=boundaries, legacy=legacy):
                        model.zero_grad()
                        trainer.label_smoother = _LegacyLabelSmoother() if legacy else LabelSmoother(epsilon=0.1)
                        actual = 0
                        for start, end in zip(boundaries, boundaries[1:]):
                            batch = {key: value[start:end] for key, value in inputs.items()}
                            batch['labels'] = labels[start:end].clone()
                            actual = actual + Seq2SeqTrainer.compute_loss(
                                trainer, model, batch, num_items_in_batch=int(count) if legacy else count)
                        actual.backward()
                        torch.testing.assert_close(actual, expected)
                        self._assert_grads(model, expected_grads)

    def test_no_count_keeps_microbatch_mean(self):
        for causal in (True, False):
            for training in (True, False):
                with self.subTest(causal=causal, training=training):
                    model, trainer, inputs, labels = self._fixture(causal)
                    model.train(training)
                    expected = self._reference(model, inputs, labels, causal)
                    expected.backward()
                    expected_grads = self._grads(model)
                    model.zero_grad()
                    actual, outputs = Seq2SeqTrainer.compute_loss(
                        trainer, model, {
                            **inputs, 'labels': labels.clone()
                        }, return_outputs=True)
                    actual.backward()
                    self.assertEqual(outputs.logits.shape[:2], labels.shape)
                    torch.testing.assert_close(actual, expected)
                    self._assert_grads(model, expected_grads)

    def test_global_count_preserves_existing_device_scaling(self):
        for causal in (True, False):
            for training in (True, False):
                with self.subTest(causal=causal, training=training):
                    model, trainer, inputs, labels = self._fixture(causal)
                    model.train(training)
                    expected = self._reference(model, inputs, labels, causal)
                    expected.backward()
                    expected_grads = self._grads(model)
                    model.zero_grad()
                    trainer.args.average_tokens_across_devices = True
                    trainer.accelerator.num_processes = 2
                    count = (labels[:, 1:] if causal else labels).ne(-100).sum()
                    losses = []
                    # Check the existing two-rank averaging arithmetic without launching processes.
                    for start, end in ((0, 1), (1, 3)):
                        batch = {key: value[start:end] for key, value in inputs.items()}
                        batch['labels'] = labels[start:end].clone()
                        losses.append(Seq2SeqTrainer.compute_loss(trainer, model, batch, num_items_in_batch=count))
                    actual = sum(losses) / 2
                    actual.backward()
                    torch.testing.assert_close(actual, expected)
                    self._assert_grads(model, expected_grads)


if __name__ == '__main__':
    unittest.main()
