# Copyright (c) ModelScope Contributors. All rights reserved.
import copy
import os
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from collections import defaultdict
from datetime import timedelta
from itertools import product
from torch.distributed import init_device_mesh
from torch.nn.parallel import DistributedDataParallel
from types import MethodType, SimpleNamespace

from swift.model import get_matched_model_meta
from swift.sequence_parallel import sequence_parallel
from swift.trainers import Seq2SeqTrainer
from swift.trainers.mixin import SwiftMixin


class Metric:

    def __init__(self):
        self.values = []

    def update(self, values):
        self.values.extend(values if isinstance(values, list) else [values.detach().clone()])


def make_trainer(model):
    trainer = SimpleNamespace(
        model=model,
        task_type='causal_lm',
        problem_type=None,
        compute_loss_func=None,
        label_smoother=None,
        model_accepts_loss_kwargs=True,
        custom_metrics={
            'train': defaultdict(Metric),
            'eval': defaultdict(Metric)
        },
        template=SimpleNamespace(
            sequence_parallel_size=2,
            padding_free=False,
            is_encoder_decoder=False,
            compute_sft_loss=lambda model, inputs, **kwargs: model(**inputs)),
        accelerator=SimpleNamespace(unwrap_model=lambda model: model, num_processes=2),
        args=SimpleNamespace(
            router_aux_loss_coef=None,
            use_liger_kernel=False,
            past_index=-1,
            enable_dft_loss=False,
            enable_channel_loss=True,
            average_tokens_across_devices=False,
            tuner_backend='peft',
            acc_strategy='token'))
    trainer._compute_acc = MethodType(Seq2SeqTrainer._compute_acc, trainer)
    return trainer


def _check_model(rank, rendezvous, model_kind):
    from transformers import Qwen3Config, Qwen3ForCausalLM

    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method=rendezvous, rank=rank, world_size=2, timeout=timedelta(seconds=90))
    try:
        sp = sequence_parallel
        sp.world_size = sp.sp_world_size = 2
        sp.rp_world_size = 1
        sp.device_mesh = init_device_mesh('cpu', (1, 2), mesh_dim_names=('data', 'sequence'))
        sp.tokenizer = SimpleNamespace(pad_token_id=0)
        sp.model_dtype = torch.float32
        sp.padding_free = False
        torch.manual_seed(42)
        if model_kind == 'qwen3':
            config = Qwen3Config(
                vocab_size=32,
                hidden_size=16,
                intermediate_size=32,
                num_hidden_layers=1,
                num_attention_heads=2,
                num_key_value_heads=2,
                head_dim=8,
                attention_dropout=0.,
                use_cache=False)
            config._attn_implementation = 'sdpa'
            model = Qwen3ForCausalLM(config)
            model.model_info = SimpleNamespace(is_moe_model=False)
            model.model_meta = SimpleNamespace(is_multimodal=False)
        else:
            from transformers import Qwen3_5MoeConfig, Qwen3_5MoeForConditionalGeneration

            from swift.model.models.qwen import _patch_qwen3_5_linear_attention_sequence_parallel

            config = Qwen3_5MoeConfig(
                text_config=dict(
                    vocab_size=32,
                    hidden_size=32,
                    num_hidden_layers=2,
                    num_attention_heads=2,
                    num_key_value_heads=2,
                    head_dim=16,
                    linear_key_head_dim=8,
                    linear_value_head_dim=8,
                    linear_num_key_heads=2,
                    linear_num_value_heads=2,
                    moe_intermediate_size=16,
                    shared_expert_intermediate_size=16,
                    num_experts=2,
                    num_experts_per_tok=1,
                    layer_types=['linear_attention', 'full_attention'],
                    use_cache=False),
                vision_config=dict(
                    depth=1,
                    hidden_size=32,
                    intermediate_size=32,
                    num_heads=2,
                    out_hidden_size=32,
                    num_position_embeddings=16),
                image_token_id=28,
                video_token_id=29,
                vision_start_token_id=30,
                vision_end_token_id=31)
            config._attn_implementation = 'sdpa'
            model = Qwen3_5MoeForConditionalGeneration(config)
            model.model.visual.requires_grad_(False)
            model.model_info = SimpleNamespace(is_moe_model=True)
            model.model_meta = get_matched_model_meta('Qwen/Qwen3.6-35B-A3B')
            _patch_qwen3_5_linear_attention_sequence_parallel()
        base_model = model
        if model_kind == 'qwen3_5_moe_lora':
            from peft import LoraConfig, get_peft_model

            model = get_peft_model(
                model,
                LoraConfig(
                    r=2, lora_alpha=4, target_modules=['q_proj', 'v_proj', 'in_proj_qkv'], task_type='CAUSAL_LM'))
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        reference = copy.deepcopy(model)
        sp._prepare_flash_attn(base_model.model)
        sp._prepare_forward_hook(base_model.model)
        if model_kind != 'qwen3':
            sp._prepare_moe_aux_loss(base_model.model.language_model)
        ddp = DistributedDataParallel(model, find_unused_parameters=model_kind != 'qwen3')
        trainer = make_trainer(model)
        head_lengths = []
        base_model.lm_head.register_forward_pre_hook(lambda module, args: head_lengths.append(args[0].shape[1]))
        for length, target_mode, scale_mode, denominator in product((8, 7), ('tail', 'boundary'),
                                                                    ('none', 'weighted', 'zero'), (None, 11)):
            positions = torch.arange(length)[None]
            ids = (positions + 1) % 32
            labels = ids.clone()
            labels[:, :length - 2] = -100  # Rank zero has no supervised positions.
            if target_mode == 'boundary':
                labels[:, 1] = ids[:, 1]
                labels[:, length // 2] = ids[:, length // 2]
            trainer.args.acc_strategy = 'seq' if target_mode == 'boundary' else 'token'
            trainer.args.enable_dft_loss = target_mode == 'boundary' and denominator is not None
            os.environ['CELOSS_PARALLEL_SIZE'] = '2' if scale_mode == 'weighted' else '0'
            weights = torch.linspace(0.25, 2., length)[None]
            if scale_mode == 'zero':
                weights.zero_()
            count = (labels != -100).sum() if denominator is None else denominator
            # Independent full-sequence model and causal CE reference.
            sp.world_size = 1
            reference.zero_grad(set_to_none=True)
            logits = reference(input_ids=ids, position_ids=positions).logits
            losses = torch.nn.functional.cross_entropy(
                logits[:, :-1].reshape(-1, 32), labels[:, 1:].reshape(-1), reduction='none')
            if trainer.args.enable_dft_loss:
                losses = losses * torch.exp(-losses.detach())
            if scale_mode != 'none':
                losses = losses * weights[:, 1:].flatten()
            reference_loss = losses.sum() / count
            reference_loss.backward()
            sp.world_size = 2
            results = []
            for selected in (False, True):
                model.train()
                ddp.zero_grad(set_to_none=True)
                trainer.custom_metrics = {'train': defaultdict(Metric), 'eval': defaultdict(Metric)}
                inputs = {
                    'input_ids': ids,
                    'position_ids': positions,
                    'attention_mask': torch.ones_like(ids),
                    'labels': labels.clone()
                }
                if model_kind != 'qwen3':
                    inputs['mm_token_type_ids'] = torch.zeros_like(ids)
                if scale_mode != 'none':
                    inputs['loss_scale'] = weights.clone()
                sp.prepare_inputs(inputs)
                original_labels = inputs['labels'].clone()
                if selected:
                    Seq2SeqTrainer.prepare_logits_to_keep(trainer, inputs)
                    torch.testing.assert_close(inputs['labels'], original_labels)
                loss = Seq2SeqTrainer.compute_loss(trainer, ddp, inputs, num_items_in_batch=denominator)
                loss.backward()
                torch.testing.assert_close(loss, reference_loss, rtol=2e-5, atol=2e-6)
                for (name, parameter), (ref_name, ref_parameter) in zip(model.named_parameters(),
                                                                        reference.named_parameters()):
                    assert name == ref_name
                    if not parameter.requires_grad:
                        continue
                    assert parameter.grad is not None
                    torch.testing.assert_close(parameter.grad, ref_parameter.grad, rtol=2e-4, atol=2e-6)
                results.append(trainer.custom_metrics['train'])
                if selected and rank == 0 and target_mode == 'tail':
                    assert head_lengths[-1] == 1  # The ignored sentinel keeps all DDP parameters connected.
            metric = f'{trainer.args.acc_strategy}_acc'
            assert results[0][metric].values and results[0][metric].values == results[1][metric].values
            assert results[0]['loss_None'].values and results[1]['loss_None'].values
            for a, b in zip(results[0]['loss_None'].values, results[1]['loss_None'].values):
                torch.testing.assert_close(a, b)
        if model_kind != 'qwen3':
            # Auxiliary router loss must be unchanged by selecting vocabulary logits.
            trainer.args.router_aux_loss_coef = 0.001
            trainer.args.enable_dft_loss = False
            comparisons = []
            for selected in (False, True):
                ddp.zero_grad(set_to_none=True)
                inputs = {
                    'input_ids': ids,
                    'position_ids': positions,
                    'attention_mask': torch.ones_like(ids),
                    'labels': labels.clone(),
                    'output_router_logits': True
                }
                sp.prepare_inputs(inputs)
                if selected:
                    Seq2SeqTrainer.prepare_logits_to_keep(trainer, inputs)
                loss, outputs = Seq2SeqTrainer.compute_loss(
                    trainer, ddp, inputs, return_outputs=True, num_items_in_batch=11)
                assert outputs.aux_loss is not None and torch.isfinite(outputs.aux_loss)
                loss.backward()
                comparisons.append((loss.detach(), {
                    n: p.grad.clone()
                    for n, p in model.named_parameters() if p.requires_grad and p.grad is not None
                }))
            torch.testing.assert_close(comparisons[0][0], comparisons[1][0])
            assert comparisons[0][1].keys() == comparisons[1][1].keys()
            for name in comparisons[0][1]:
                torch.testing.assert_close(comparisons[0][1][name], comparisons[1][1][name], rtol=2e-4, atol=2e-6)
        model.eval()
        inputs = {'labels': torch.tensor([[-100, 3, 4, -100]])}
        Seq2SeqTrainer.prepare_logits_to_keep(trainer, inputs)
        assert 'logits_to_keep' not in inputs
    except Exception:
        import traceback
        traceback.print_exc()
        raise
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_available() or not dist.is_gloo_available(), reason='Gloo is not available')
@pytest.mark.parametrize('model_kind', ['qwen3', 'qwen3_5_moe', 'qwen3_5_moe_lora'])
def test_sp_logits_selection_matches_full_loss_and_gradients(tmp_path, model_kind):
    module_name = 'qwen3' if model_kind == 'qwen3' else 'qwen3_5_moe'
    pytest.importorskip(f'transformers.models.{module_name}')
    mp.spawn(_check_model, args=((tmp_path / 'rendezvous').as_uri(), model_kind), nprocs=2)


@pytest.mark.parametrize(
    'unsupported',
    ['batch', 'padding_free', 'ring', 'custom_loss', 'smoothing', 'multimodal', 'encoder_decoder', 'unsloth', 'liger'])
def test_sp_logits_selection_rejects_unsupported_configs(monkeypatch, unsupported):
    model = SimpleNamespace(training=True, model_meta=SimpleNamespace(is_multimodal=False, model_type='other'))
    trainer = make_trainer(model)
    monkeypatch.setattr(sequence_parallel, 'rp_world_size', 1)
    inputs = {'labels': torch.tensor([[-100, 1, 2, -100]])}
    if unsupported == 'batch':
        inputs['labels'] = inputs['labels'].repeat(2, 1)
    elif unsupported == 'padding_free':
        trainer.template.padding_free = True
    elif unsupported == 'ring':
        monkeypatch.setattr(sequence_parallel, 'rp_world_size', 2)
    elif unsupported == 'custom_loss':
        trainer.compute_loss_func = object()
    elif unsupported == 'smoothing':
        trainer.label_smoother = object()
    elif unsupported == 'multimodal':
        model.model_meta.is_multimodal = True
    elif unsupported == 'encoder_decoder':
        trainer.template.is_encoder_decoder = True
    elif unsupported == 'unsloth':
        trainer.args.tuner_backend = 'unsloth'
    else:
        trainer.args.use_liger_kernel = True
    with pytest.raises(NotImplementedError, match='SP logits_to_keep'):
        Seq2SeqTrainer.prepare_logits_to_keep(trainer, inputs)
    assert 'logits_to_keep' not in inputs


def test_non_sp_logits_selection_unchanged():
    trainer = Seq2SeqTrainer.__new__(Seq2SeqTrainer)
    trainer.template = SimpleNamespace(sequence_parallel_size=1)
    inputs = {'labels': torch.tensor([[-100, -100, 2, 3]]), 'loss_scale': torch.tensor([[0., 0., 0.5, 2.]])}
    Seq2SeqTrainer.prepare_logits_to_keep(trainer, inputs)
    torch.testing.assert_close(inputs['labels'], torch.tensor([[-100, 2, 3]]))
    torch.testing.assert_close(inputs['loss_scale'], torch.tensor([[0., 0.5, 2.]]))
    torch.testing.assert_close(inputs['logits_to_keep'], torch.tensor([False, True, True, True]))


@pytest.mark.parametrize('training', [True, False])
def test_shared_mixin_still_rejects_sp_logits_selection(training):
    # DPO, KTO and GKD use this shared method, without SFT's loss restoration.
    trainer = SimpleNamespace(
        template=SimpleNamespace(sequence_parallel_size=2), model=SimpleNamespace(training=training))
    with pytest.raises(NotImplementedError):
        SwiftMixin.prepare_logits_to_keep(trainer, {'labels': torch.tensor([[-100, 1]])})


@pytest.mark.parametrize('key', [
    'pixel_values', 'pixel_values_videos', 'image_grid_thw', 'video_grid_thw', 'mm_token_type_ids', 'inputs_embeds',
    'missing_input_ids'
])
def test_qwen_moe_sp_selection_rejects_non_text_inputs(monkeypatch, key):
    model = SimpleNamespace(training=True, model_meta=get_matched_model_meta('Qwen/Qwen3.6-35B-A3B'))
    trainer = make_trainer(model)
    monkeypatch.setattr(sequence_parallel, 'rp_world_size', 1)
    inputs = {'input_ids': torch.tensor([[1, 2]]), 'labels': torch.tensor([[-100, 2]])}
    if key == 'missing_input_ids':
        inputs.pop('input_ids')
    else:
        inputs[key] = torch.ones(1)
    with pytest.raises(NotImplementedError, match='text-only inputs'):
        Seq2SeqTrainer.prepare_logits_to_keep(trainer, inputs)
    assert 'logits_to_keep' not in inputs
