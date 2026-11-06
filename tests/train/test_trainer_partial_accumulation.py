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

from swift.loss.decision import ScoringLoss
from swift.loss.reranker import PointwiseRerankerLoss
from swift.metrics import MeanMetric
from swift.model.decision_head import ScoringOutput
from swift.trainers import RerankerTrainer, Trainer
from swift.trainers.decision_trainer import ScoringTrainer


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


class TinyDecisionModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([[0.5, -0.5]]))
        self.config = SimpleNamespace()

    def forward(self, input_ids, labels=None, **kwargs):
        logits = input_ids @ self.weight
        return ScoringOutput(logits=logits, option_mask=torch.ones_like(logits, dtype=torch.bool))


@pytest.mark.parametrize('kind', ['trainer', 'trainer_no_loss_kwargs', 'reranker', 'decision'])
@pytest.mark.parametrize('window_size', [1, 2, 4])
def test_partial_accumulation_matches_full_batch(tmp_path, kind, window_size):
    model = TinyDecisionModel() if kind == 'decision' else TinyRegressor(
        accepts_loss_kwargs=kind != 'trainer_no_loss_kwargs')
    trainer_cls = {'reranker': RerankerTrainer, 'decision': ScoringTrainer}.get(kind, Trainer)
    # Initialize the real HF training/backward machinery without model downloads
    # or Swift's unrelated tokenizer, checkpoint and callback setup.
    trainer = trainer_cls.__new__(trainer_cls)
    args = TrainingArguments(
        output_dir=str(tmp_path), use_cpu=True, gradient_accumulation_steps=4, report_to=[], disable_tqdm=True)
    HfTrainer.__init__(trainer, model=model, args=args)
    trainer.task_type = kind if kind in {'reranker', 'decision'} else 'seq_cls'
    trainer.problem_type = 'regression'
    trainer.template = SimpleNamespace(sequence_parallel_size=1)
    trainer.custom_metrics = {'train': collections.defaultdict(lambda: MeanMetric(nan_value=None, device='cpu'))}
    args.loss_type = 'pointwise_reranker' if kind == 'reranker' else None
    if kind == 'reranker':
        trainer.compute_loss_func = PointwiseRerankerLoss(args, trainer)
    elif kind == 'decision':
        trainer.compute_loss_func = ScoringLoss(args, trainer)
    # SwiftMixin sets Accelerate's num_steps=1; Trainer owns loss normalization.
    trainer.accelerator.gradient_state.plugin_kwargs['num_steps'] = 1
    # HF's epoch loop sets this from the number of prefetched micro-batches.
    trainer.current_gradient_accumulation_steps = window_size
    batches = [{
        'input_ids': torch.tensor([[float(i + 1)], [float(i + 2)]]),
        'labels': torch.tensor([[1., 0.], [0., 1.]]) if kind == 'decision' else torch.tensor([0., 1.])
    } for i in range(window_size)]
    num_items = sum(batch['labels'].numel() for batch in batches)
    accumulated_loss = sum(trainer.training_step(model, dict(batch), num_items_in_batch=num_items) for batch in batches)

    reference = TinyDecisionModel() if kind == 'decision' else TinyRegressor()
    inputs = torch.cat([batch['input_ids'] for batch in batches])
    labels = torch.cat([batch['labels'] for batch in batches])
    output = reference(input_ids=inputs, labels=labels)
    if kind == 'decision':
        reference_loss = F.cross_entropy(output.logits, labels)
    else:
        reference_loss = F.binary_cross_entropy_with_logits(output.logits[:, 0],
                                                            labels) if kind == 'reranker' else output.loss
    reference_loss.backward()
    torch.testing.assert_close(model.weight.grad, reference.weight.grad)
    torch.testing.assert_close(accumulated_loss, reference_loss.detach())


@pytest.mark.parametrize('kind', ['trainer', 'trainer_no_loss_kwargs', 'reranker', 'decision'])
@pytest.mark.parametrize('configured_steps,current_steps', [(1, None), (4, None), (4, 1), (4, 2), (4, 4)])
def test_eval_loss_ignores_accumulation(tmp_path, kind, configured_steps, current_steps):
    model = TinyDecisionModel() if kind == 'decision' else TinyRegressor(
        accepts_loss_kwargs=kind != 'trainer_no_loss_kwargs')
    dataset = [{'input_ids': [1.], 'labels': [0., 1.] if kind == 'decision' else 1.} for _ in range(4)]
    trainer_cls = {'reranker': RerankerTrainer, 'decision': ScoringTrainer}.get(kind, Trainer)
    trainer = trainer_cls.__new__(trainer_cls)
    args = TrainingArguments(
        output_dir=str(tmp_path),
        use_cpu=True,
        gradient_accumulation_steps=configured_steps,
        per_device_eval_batch_size=2,
        report_to=[],
        disable_tqdm=True)
    HfTrainer.__init__(trainer, model=model, args=args, eval_dataset=dataset)
    trainer.task_type = kind if kind in {'reranker', 'decision'} else 'seq_cls'
    trainer.problem_type = 'regression'
    trainer.template = SimpleNamespace(sequence_parallel_size=1)
    trainer.custom_metrics = {'eval': collections.defaultdict(lambda: MeanMetric(nan_value=None, device='cpu'))}
    args.loss_type = 'pointwise_reranker' if kind == 'reranker' else None
    if kind == 'reranker':
        trainer.compute_loss_func = PointwiseRerankerLoss(args, trainer)
    elif kind == 'decision':
        trainer.compute_loss_func = ScoringLoss(args, trainer)
    if current_steps is not None:
        trainer.current_gradient_accumulation_steps = current_steps
    # Swift's printer callback initializes its timing state at training start.
    trainer.callback_handler.on_train_begin(args, trainer.state, trainer.control)
    metrics = trainer.evaluate()
    if kind == 'decision':
        expected = F.cross_entropy(torch.tensor([[0.5, -0.5]]), torch.tensor([1])).item()
    elif kind == 'reranker':
        expected = F.binary_cross_entropy_with_logits(torch.tensor([0.5]), torch.tensor([1.])).item()
    else:
        expected = 0.25  # (0.5 - 1.0) ** 2
    assert metrics['eval_loss'] == pytest.approx(expected)
