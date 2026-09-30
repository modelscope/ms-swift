# Copyright (c) ModelScope Contributors. All rights reserved.
import collections
import pytest
import torch
import torch.nn.functional as F
from torch import nn
from transformers import Trainer as HfTrainer
from transformers import TrainingArguments
from transformers.modeling_outputs import SequenceClassifierOutput
from types import SimpleNamespace

from swift.loss.reranker import PointwiseRerankerLoss
from swift.metrics import MeanMetric
from swift.trainers import RerankerTrainer, Trainer


class TinyRegressor(nn.Module):

    def __init__(self, accepts_loss_kwargs=True):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([[0.5]]))
        self.config = SimpleNamespace(problem_type='regression')
        self.accepts_loss_kwargs = accepts_loss_kwargs

    def forward(self, input_ids, labels=None, **kwargs):
        logits = input_ids @ self.weight
        loss = None if labels is None else F.mse_loss(logits[:, 0], labels)
        return SequenceClassifierOutput(loss=loss, logits=logits)


@pytest.mark.parametrize('kind', ['trainer', 'trainer_no_loss_kwargs', 'reranker'])
@pytest.mark.parametrize('window_size', [1, 2, 4])
def test_partial_accumulation_matches_full_batch(tmp_path, kind, window_size):
    model = TinyRegressor(accepts_loss_kwargs=kind != 'trainer_no_loss_kwargs')
    trainer_cls = RerankerTrainer if kind == 'reranker' else Trainer
    # Initialize the real HF training/backward machinery without model downloads
    # or Swift's unrelated tokenizer, checkpoint and callback setup.
    trainer = trainer_cls.__new__(trainer_cls)
    args = TrainingArguments(
        output_dir=str(tmp_path), use_cpu=True, gradient_accumulation_steps=4, report_to=[], disable_tqdm=True)
    HfTrainer.__init__(trainer, model=model, args=args)
    trainer.task_type = 'reranker' if kind == 'reranker' else 'seq_cls'
    trainer.problem_type = 'regression'
    trainer.template = SimpleNamespace(sequence_parallel_size=1)
    trainer.custom_metrics = {'train': collections.defaultdict(lambda: MeanMetric(nan_value=None, device='cpu'))}
    args.loss_type = 'pointwise_reranker' if kind == 'reranker' else None
    if kind == 'reranker':
        trainer.compute_loss_func = PointwiseRerankerLoss(args, trainer)
    # SwiftMixin sets Accelerate's num_steps=1; Trainer owns loss normalization.
    trainer.accelerator.gradient_state.plugin_kwargs['num_steps'] = 1
    # HF's epoch loop sets this from the number of prefetched micro-batches.
    trainer.current_gradient_accumulation_steps = window_size
    batches = [{
        'input_ids': torch.tensor([[float(i + 1)], [float(i + 2)]]),
        'labels': torch.tensor([0., 1.])
    } for i in range(window_size)]
    num_items = sum(batch['labels'].numel() for batch in batches)
    accumulated_loss = sum(trainer.training_step(model, dict(batch), num_items_in_batch=num_items) for batch in batches)

    reference = TinyRegressor()
    inputs = torch.cat([batch['input_ids'] for batch in batches])
    labels = torch.cat([batch['labels'] for batch in batches])
    output = reference(input_ids=inputs, labels=labels)
    reference_loss = F.binary_cross_entropy_with_logits(output.logits[:,
                                                                      0], labels) if kind == 'reranker' else output.loss
    reference_loss.backward()
    torch.testing.assert_close(model.weight.grad, reference.weight.grad)
    torch.testing.assert_close(accumulated_loss, reference_loss.detach())
