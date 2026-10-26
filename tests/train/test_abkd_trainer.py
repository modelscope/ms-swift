# Copyright (c) ModelScope Contributors. All rights reserved.
import copy
import json
import pytest
import torch
import torch.nn.functional as F
from datasets import Dataset
from test_abkd_loss import reference_abkd
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import HfArgumentParser, PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM

from swift import RLHFArguments
from swift.model import get_model_processor
from swift.rlhf_trainers import GKDConfig, GKDTrainer
from swift.rlhf_trainers.gkd_loss import DataSource
from swift.template import get_template
from swift.trainers.trainer_factory import TrainerFactory


def create_tiny_model(path, hidden_size):
    words = ['[PAD]', '[EOS]', '[UNK]', 'Solve', '1', '+', '2', '3', '4', 'Answer']
    words.extend(f'token{i}' for i in range(54))
    tokenizer = Tokenizer(WordLevel({word: i for i, word in enumerate(words)}, unk_token='[UNK]'))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, pad_token='[PAD]', eos_token='[EOS]', unk_token='[UNK]')
    config = Qwen3Config(
        vocab_size=len(words),
        hidden_size=hidden_size,
        intermediate_size=hidden_size * 2,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=128,
        pad_token_id=0,
        eos_token_id=1,
        tie_word_embeddings=False)
    model = Qwen3ForCausalLM(config)
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)
    return get_model_processor(
        str(path), model_type='qwen3', torch_dtype=torch.float32, device_map='cpu', attn_impl='eager')


def test_abkd_cli_parameters_reach_gkd_config(tmp_path):
    create_tiny_model(tmp_path / 'model', 32)
    data_path = tmp_path / 'data.jsonl'
    data_path.write_text(
        json.dumps({'messages': [{
            'role': 'user',
            'content': 'Solve 1 + 1'
        }, {
            'role': 'assistant',
            'content': '2'
        }]}) + '\n')
    args, = HfArgumentParser(RLHFArguments).parse_args_into_dataclasses([
        '--rlhf_type',
        'gkd',
        '--model',
        str(tmp_path / 'model'),
        '--model_type',
        'qwen3',
        '--dataset',
        str(data_path),
        '--template',
        'default',
        '--loss_type',
        'abkd',
        '--abkd_alpha',
        '1.2',
        '--abkd_beta',
        '-0.2',
        '--torch_dtype',
        'float32',
        '--report_to',
        'none',
        '--output_dir',
        str(tmp_path / 'output'),
        '--use_cpu',
        str(not torch.cuda.is_available()).lower(),
        '--dataloader_num_workers',
        '0',
        '--dataloader_persistent_workers',
        'false',
    ])
    training_args = TrainerFactory.get_training_args(args)
    assert isinstance(training_args, GKDConfig)
    assert training_args.loss_type == 'abkd'
    assert training_args.abkd_alpha == 1.2
    assert training_args.abkd_beta == -0.2
    assert TrainerFactory.get_trainer_cls(args) is GKDTrainer


@pytest.mark.parametrize('loss_type,topk,use_liger,sft_alpha,lmbda', [
    ('jsd', None, False, 0., 1.),
    ('abkd', None, False, 0., 1.),
    ('jsd', 8, False, 0., 1.),
    ('abkd', 8, False, 0., 1.),
    pytest.param(
        'abkd',
        8,
        True,
        0.3,
        0.,
        marks=pytest.mark.skipif(not torch.cuda.is_available(), reason='Liger kernels require CUDA')),
])
def test_training_matches_reference_and_updates_only_student(tmp_path, loss_type, topk, use_liger, sft_alpha, lmbda):
    torch.manual_seed(42)
    student, processor = create_tiny_model(tmp_path / 'student', 32)
    teacher, _ = create_tiny_model(tmp_path / 'teacher', 48)
    template = get_template(processor, template_type='default', max_length=96, truncation_strategy='right')
    template.set_mode('train')
    template.model = student
    dataset = Dataset.from_list([{
        'messages': [{
            'role': 'user',
            'content': f'Solve {i} + 1'
        }, {
            'role': 'assistant',
            'content': str(i + 1)
        }]
    } for i in range(1, 5)])
    args, = HfArgumentParser(GKDConfig).parse_dict({
        'output_dir': str(tmp_path / 'output'),
        'loss_type': loss_type,
        'abkd_alpha': 0.2,
        'abkd_beta': 0.7,
        'beta': 0.5,
        'temperature': 0.8,
        'lmbda': lmbda,
        'max_completion_length': 4,
        'max_steps': 3,
        'use_liger_kernel': use_liger,
        'sft_alpha': sft_alpha,
        'per_device_train_batch_size': 4,
        'gradient_accumulation_steps': 1,
        'learning_rate': 1e-3,
        'gradient_checkpointing': False,
        'use_logits_to_keep': False,
        'save_strategy': 'no',
        'report_to': [],
        'disable_tqdm': True,
        'dataloader_num_workers': 0,
        'dataloader_persistent_workers': False,
        'check_model': False,
        'use_cpu': not torch.cuda.is_available(),
        'remove_unused_columns': False,
    })
    trainer = GKDTrainer(
        model=student, teacher_model=teacher, args=args, template=template, train_dataset=dataset, gkd_logits_topk=topk)
    assert student.device.type == ('cuda' if torch.cuda.is_available() else 'cpu')
    before_student = {name: value.detach().clone() for name, value in student.named_parameters()}
    before_teacher = {name: value.detach().clone() for name, value in teacher.named_parameters()}

    inputs = trainer._generate_and_score_completions(list(dataset))[0]
    source = DataSource.STUDENT if lmbda == 1. else DataSource.DATASET
    assert inputs['gkd_batch'].data_source == source
    actual = trainer.compute_loss(student, copy.deepcopy(inputs))
    model_inputs = {key: value for key, value in inputs['model_inputs'].items() if key != 'labels'}
    teacher_inputs = {key: value for key, value in inputs['teacher_model_inputs'].items() if key != 'labels'}
    student_logits = student(**model_inputs).logits
    full_student_logits = student_logits
    with torch.no_grad():
        teacher_logits = teacher(**teacher_inputs).logits
    mask = torch.roll(inputs['model_inputs']['labels'], -1, 1) != -100
    assert mask.sum().item() > 0
    student_logits, teacher_logits = student_logits[mask], teacher_logits[mask]
    if topk is not None:
        teacher_logits, indices = teacher_logits.topk(topk, dim=-1)
        student_logits = student_logits.gather(-1, indices)
    if loss_type == 'abkd':
        expected = reference_abkd(student_logits, teacher_logits, 0.2, 0.7, 0.8) / mask.sum()
    else:
        s_log, t_log = (student_logits.double() / 0.8).log_softmax(-1), (teacher_logits.double() / 0.8).log_softmax(-1)
        mixture = ((s_log.exp() + t_log.exp()) / 2).log()
        expected = (0.5 * (s_log.exp() * (s_log - mixture) + t_log.exp() * (t_log - mixture))).sum() / mask.sum()
    if sft_alpha > 0:
        expected = expected + sft_alpha * F.cross_entropy(
            full_student_logits[:, :-1].float().reshape(-1, full_student_logits.shape[-1]),
            inputs['model_inputs']['labels'][:, 1:].reshape(-1))
    torch.testing.assert_close(actual.double(), expected, rtol=2e-4, atol=1e-6)
    actual_gradient, = torch.autograd.grad(actual, student.lm_head.weight)
    expected_gradient, = torch.autograd.grad(expected, student.lm_head.weight)
    torch.testing.assert_close(actual_gradient, expected_gradient, rtol=2e-4, atol=2e-6)

    result = trainer.train()
    assert trainer.state.global_step == 3
    assert torch.isfinite(torch.tensor(result.training_loss))
    assert trainer._data_source == source
    assert any(not torch.equal(value, before_student[name]) for name, value in student.named_parameters())
    for name, value in teacher.named_parameters():
        torch.testing.assert_close(value, before_teacher[name], rtol=0, atol=0)
        assert value.grad is None
