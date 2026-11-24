"""Compare accumulated causal-LM updates with the combined target-token mean."""
import copy
import importlib.util
import pytest
import torch
import torch.nn.functional as F
from types import SimpleNamespace

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec('fsdp_turbo') is None, reason='requires the optional FSDPTurbo package')


class TinyCausalLM(torch.nn.Module):

    def __init__(self, *, cp_group=None, fp32_loss=False, dtype=torch.float64):
        super().__init__()
        self.embed = torch.nn.Embedding(5, 3, dtype=dtype)
        self.head = torch.nn.Linear(3, 5, bias=False, dtype=dtype)
        self.cp_group = cp_group
        self.fp32_loss = fp32_loss

    def forward(self, input_ids, labels, **kwargs):
        logits = self.head(self.embed(input_ids))
        if self.cp_group is not None:
            from fsdp_turbo.distributed.context_parallel.loss import cp_compute_lm_loss

            loss = cp_compute_lm_loss(logits, labels, self.cp_group)
        else:
            logits = logits.float() if self.fp32_loss else logits
            loss = F.cross_entropy(logits[:, :-1].reshape(-1, 5), labels[:, 1:].reshape(-1), ignore_index=-100)
        return SimpleNamespace(loss=loss)


def token_batch(counts, offset=0):
    ids = (torch.arange(8).expand(len(counts), -1) + offset).remainder(5)
    labels = ids.clone()
    labels[:, 0] = -100
    for row, count in enumerate(counts):
        labels[row, count + 1:] = -100
    return {'input_ids': ids, 'labels': labels}


def accumulation_trainer(model, accumulation, *, state=None, optimizer='SGD'):
    from swift.fsdpturbo.trainer import FSDPTurboTrainer

    trainer = object.__new__(FSDPTurboTrainer)
    model.parallel_state = state or SimpleNamespace(get_group_size=lambda name: 1)
    trainer.model = model
    trainer.args = SimpleNamespace(router_aux_loss_coef=0)
    trainer.config = SimpleNamespace(run=SimpleNamespace(gradient_accumulation_steps=accumulation))
    trainer._accumulated_tokens = 0
    trainer._gradient_scale = 1.0
    trainer._replicated_parameters = []
    trainer._global_step = 0
    trainer.optimizer = getattr(torch.optim, optimizer)(model.parameters(), lr=0.03)
    trainer.lr_scheduler = torch.optim.lr_scheduler.StepLR(trainer.optimizer, step_size=3, gamma=0.9)
    trainer._clip_grad_norm = lambda: torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
    return trainer


@pytest.mark.parametrize('counts', [[(5, )], [(3, ), (3, )], [(1, ), (5, ), (0, )], [(1, ), (2, 5)]])
@pytest.mark.parametrize('optimizer', ['SGD', 'AdamW'])
@pytest.mark.parametrize('dtype', [torch.float64, torch.float32])
def test_accumulation_matches_combined_loss_gradients_and_clipped_updates(counts, optimizer, dtype):
    torch.manual_seed(42)
    model = TinyCausalLM(dtype=dtype)
    reference = copy.deepcopy(model)
    trainer = accumulation_trainer(model, len(counts), optimizer=optimizer)
    reference_optimizer = getattr(torch.optim, optimizer)(reference.parameters(), lr=0.03)
    reference_scheduler = torch.optim.lr_scheduler.StepLR(reference_optimizer, step_size=3, gamma=0.9)
    rtol, atol = (1e-12, 1e-12) if dtype == torch.float64 else (2e-6, 2e-7)
    for step in range(20):
        batches = [token_batch(count, step + micro) for micro, count in enumerate(counts)]
        losses = [trainer.train_step(dict(batch))[0].detach() for batch in batches]
        result = trainer.sync_loss({'loss': losses})
        combined = {name: torch.cat([batch[name] for batch in batches]) for name in ('input_ids', 'labels')}
        expected_loss = reference(**combined).loss
        expected_loss.backward()
        torch.testing.assert_close(result['loss'].double(), expected_loss.double(), rtol=1e-6, atol=1e-7)

        def checked_clip():
            for actual, expected in zip(model.parameters(), reference.parameters()):
                torch.testing.assert_close(actual.grad, expected.grad, rtol=rtol, atol=atol)
            return torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)

        trainer._clip_grad_norm = checked_clip
        trainer._on_optimizer_step()
        torch.nn.utils.clip_grad_norm_(reference.parameters(), 0.1)
        reference_optimizer.step()
        reference_scheduler.step()
        reference_optimizer.zero_grad()
        for actual, expected in zip(model.parameters(), reference.parameters()):
            torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
        assert trainer._accumulated_tokens == 0


def test_empty_update_is_rejected_before_optimizer_step():
    trainer = accumulation_trainer(TinyCausalLM(), 2)
    losses = [trainer.train_step(token_batch((0, )))[0].detach() for _ in range(2)]
    with pytest.raises(ValueError, match='valid next-token label'):
        trainer.sync_loss({'loss': losses})
    assert not trainer.optimizer.state


def test_pending_host_sparse_gradients_are_scaled_before_clipping():
    from fsdp_turbo.models.deepseek.deepseek_v41.engram.engram_optimizer import register_engram_optimizers
    from fsdp_turbo.models.deepseek.deepseek_v41.engram.host_offload_engram import TorchHostOffloadEngram
    from fsdp_turbo.training.factories import OptimizerFactory

    model = torch.nn.Sequential(torch.nn.Linear(4, 4))
    table = TorchHostOffloadEngram(8, 4, layer_id=0, num_max_tokens_per_rank=4, pin_memory=False)
    table.to_empty(device='cpu')
    torch.nn.init.ones_(table.weight)
    model.add_module('table', table)
    trainer = accumulation_trainer(model, 2)
    register_engram_optimizers(OptimizerFactory)
    trainer.optimizer = OptimizerFactory.create('EngramOptimizer', model, [table], dense_optimizer_type='AdamW', lr=0.1)
    trainer.lr_scheduler = None
    trainer._accumulated_tokens = torch.tensor(8)
    trainer.sync_loss({'loss': [torch.tensor(2.)]})
    for parameter in model[0].parameters():
        parameter.grad = torch.full_like(parameter, 4)
    table.accumulate_sparse_gradient(torch.tensor([0]), torch.full((1, 4), 8.))

    def checked_clip():
        for parameter in model[0].parameters():
            torch.testing.assert_close(parameter.grad, torch.ones_like(parameter))
        torch.testing.assert_close(table.pending_sparse_grad().values(), torch.full((1, 4), 2.))
        return 0.

    trainer._clip_grad_norm = checked_clip
    trainer._on_optimizer_step()
    assert table.pending_sparse_grad() is None
