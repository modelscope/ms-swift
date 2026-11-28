"""S8c — the scalar reward-model row protocol + the LLM-judge text kernels (L1).

``RewardModel.score`` is the seq_cls RM handle: it encodes message rows and forwards them through
the score head, one float per row. The contract that matters is ``task='seq_cls'`` -- forwarding the
default 'causal_lm' would gather label logps out of a pooled [B, 1] score (a shape error on
transformers, and megatron would miss the seq_cls pooling branch entirely). The judge kernels
(render/parse/query) are the pure-text half of the generative/api judge. Stubs stand in for the
model/template; no GPU.

Maps to ut_plan S8-43 (+ the judge text kernels behind S8-44's channel).
"""
import pytest


class _StubTemplate:

    def __init__(self):
        self.rows = []

    def encode(self, row):
        self.rows.append(row)
        return {'input_ids': [len(row['messages'])]}


class _StubSeqClsModel:
    """A frozen scalar RM: forward_only must be routed as seq_cls and emits a [B, 1] score."""

    def __init__(self, outputs=None):
        self._outputs = outputs
        self.calls = []

    def forward_only(self, *, inputs, return_logits, task):
        self.calls.append((return_logits, task))
        assert return_logits is True
        assert task == 'seq_cls', f'scalar RM forward_only must be routed as seq_cls, got {task!r}'
        if self._outputs is not None:
            return self._outputs
        import torch
        return {'logits': torch.tensor([[float(item['input_ids'][0])] for item in inputs])}


def test_reward_model_scores_rows_via_seq_cls_head():
    """S8-43: each message row is deep-copied, encoded, and scored through the seq_cls head -> one
    float per row (the [B, 1] score flattened); the metric label is the handle's display_name."""
    from swift.dev.reward.scorer import RewardModel

    template = _StubTemplate()
    model = _StubSeqClsModel()
    rm = RewardModel(model, template, display_name='my_rm')
    rows = [
        {'messages': [{'role': 'user', 'content': 'q'}, {'role': 'assistant', 'content': 'a'}]},
        {'messages': [{'role': 'user', 'content': 'q2'}]},
    ]
    scores = rm.score(rows)
    assert scores == [2.0, 1.0]  # len(messages) per row, read back through the head
    assert model.calls == [(True, 'seq_cls')]  # routed as seq_cls, logits requested
    assert template.rows[0] is not rows[0]  # deep-copied before encode, never mutated in place
    assert rm.display_name == 'my_rm'


def test_reward_model_raises_when_forward_returns_no_logits():
    """S8-43 guard: a forward yielding no logits is a hard error, not a silent zero score."""
    from swift.dev.reward.scorer import RewardModel

    rm = RewardModel(_StubSeqClsModel(outputs={}), _StubTemplate())
    with pytest.raises(RuntimeError, match='returned no logits'):
        rm.score([{'messages': [{'role': 'user', 'content': 'q'}]}])


# ----------------------------------------------------------------------
# The LLM-judge text kernels (the generative/api judge's pure half)
# ----------------------------------------------------------------------
@pytest.mark.parametrize('text, expected', [
    ('verdict Reward: 0.8 done', 0.8),  # the prompted Reward: convention wins
    ('Reward: 5', 5.0),  # out of the [0,1] Reward range -> falls back to the first bare number
    ('the answer scored 42 ok', 42.0),  # no Reward: tag -> first bare number
    ('got -3.5 here', -3.5),  # a signed decimal
    ('no digits at all', None),  # nothing numeric -> None (recorded as nan), never a guessed 0
    ('', None),
    (None, None),
])
def test_parse_judge_score(text, expected):
    """S8-44 support: a judge's free-form verdict parses to a float, or None when unparseable (a
    silent 0 would rank an unparseable verdict as a genuinely bad candidate)."""
    from swift.dev.reward.scorer import parse_judge_score

    assert parse_judge_score(text) == expected


def test_render_conversation_flattens_and_skips_incomplete():
    """S8-44 support: messages flatten to ``Role: content`` lines; contentless or non-dict entries skip."""
    from swift.dev.reward.scorer import render_conversation

    messages = [
        {'role': 'user', 'content': 'hi'},
        {'role': 'assistant', 'content': 'yo'},
        {'role': 'system'},  # no content -> skipped
        'junk',  # not a dict -> skipped
    ]
    assert render_conversation(messages) == 'User: hi\nAssistant: yo'
    assert render_conversation(None) == ''


def test_judge_query_strips_trailing_assistant_and_formats():
    """S8-44 support: the candidate (trailing assistant turn) is stripped from the rendered prompt
    and supplied verbatim through {completion}; the conversation alone goes through {prompt}."""
    from swift.dev.reward.scorer import judge_query

    messages = [{'role': 'user', 'content': 'q'}, {'role': 'assistant', 'content': 'cand'}]
    assert judge_query('P={prompt} C={completion}', messages, 'cand') == 'P=User: q C=cand'


# --- defect123: process-RM (token-level) score reduction -----------------------------------------
# PLAN (case | input | expected | failure it kills):
#  T1 token-level [B,L,C>1], right-padded rows, through twinkle's REAL seq_cls pooler | per-row
#     last-valid-token softmax positive class | reshape(-1) leaking B*L*C floats / reading the
#     first prompt token's logit as the score.
#  T2 token-level [B,L,C>1] handed to score WITHOUT the pooler | RuntimeError naming the shape |
#     a scorer-side last-valid-token branch (07-F-1) silently duplicating twinkle's pooling rule.
#  T3 pooled [B,1] (sequence RM) | unchanged flatten | regressing the existing scalar-RM path.
#  T4 twinkle-pooled [B,C>1] | per-row softmax positive class | the real-ray failure: reshape(-1)
#     leaking C class logits per row ("FunctionWorker returned 2 scores for 1 trajectories").
# Independent oracle: manual per-row last-valid-token extraction computed in the test, not the
# scorer's own index math. Stubs stand in for model/template (no GPU); the real ray PRM run signs E2E.
class _StubTokenModel:

    def __init__(self, logits):
        self._logits = logits

    def forward_only(self, *, inputs, return_logits, task):
        assert task == 'seq_cls'
        return {'logits': self._logits}


class _RowsTemplate:

    def encode(self, row):
        return {'input_ids': [1, 2, 3]}


def _prm_rows(n):
    return [{'messages': [{'role': 'user', 'content': f'q{i}'}]} for i in range(n)]


def _twinkle_pool(logits, masks):
    """Run the per-token head through twinkle's own seq_cls postprocess, as the model does."""
    import torch
    from twinkle.processor.base import InputProcessor

    processor = InputProcessor.__new__(InputProcessor)  # the pooling methods read no instance state
    _, outputs = processor._postprocess_seq_cls({'attention_mask': torch.tensor(masks)}, {'logits': logits})
    return outputs['logits']


def test_defect123_token_level_multiclass_pooled_by_twinkle_then_reduced():
    import torch
    from swift.dev.reward.scorer import RewardModel

    torch.manual_seed(0)
    logits = torch.randn(3, 7, 2)
    masks = [[1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 0, 0, 0], [1, 1, 1, 1, 1, 0, 0]]
    rm = RewardModel(_StubTokenModel(_twinkle_pool(logits, masks)), _RowsTemplate())
    scores = rm.score(_prm_rows(3))
    oracle = [torch.softmax(logits[i, sum(masks[i]) - 1].float(), dim=-1)[-1].item() for i in range(3)]
    assert len(scores) == 3  # one score per row, NOT B*L*C
    assert all(abs(a - b) < 1e-6 for a, b in zip(scores, oracle))


def test_unpooled_token_level_logits_are_refused():
    import torch
    from swift.dev.reward.scorer import RewardModel

    rm = RewardModel(_StubTokenModel(torch.randn(2, 5, 2)), _RowsTemplate())
    with pytest.raises(RuntimeError, match=r'must be pooled \[B, C\].*\(2, 5, 2\)'):
        rm.score(_prm_rows(2))


def test_defect123_pooled_sequence_rm_path_unchanged():
    import torch
    from swift.dev.reward.scorer import RewardModel

    logits = torch.tensor([[0.25], [0.75], [1.5]])
    rm = RewardModel(_StubTokenModel(logits), _RowsTemplate())
    assert rm.score(_prm_rows(3)) == [0.25, 0.75, 1.5]


def test_defect123_twinkle_pooled_multiclass_reduces_to_positive_class():
    """T5: twinkle's _postprocess_seq_cls pools [B,T,C] -> [B,C] BEFORE score sees it; a C>1 head
    must still yield ONE score per row (the positive-class softmax probability), not C logits."""
    import torch
    from swift.dev.reward.scorer import RewardModel

    torch.manual_seed(1)
    logits = torch.randn(3, 2)  # the exact shape the real PRM run delivered for 1-row batches
    rm = RewardModel(_StubTokenModel(logits), _RowsTemplate())
    scores = rm.score(_prm_rows(3))
    oracle = torch.softmax(logits.float(), dim=-1)[..., -1].tolist()
    assert len(scores) == 3  # one per row, NOT 3*2 class logits
    assert all(abs(a - b) < 1e-6 for a, b in zip(scores, oracle))


# --- 15-F-2: an attached-head RM checkpoint loads its trained head in EVERY mode -------------------
# PLAN (case | input | expected | failure it kills):
#  F1 adapter dir WITH task_head.safetensors, task given | add_adapter_to_model(task, num_labels) with
#     the trained target_modules read off the weights, then load(adapter_name), set_template; NO
#     set_processor | the ray path scoring through a random in-model head (15-F-2) / all-linear
#     re-expansion wrapping untrained modules / explicit position_ids changing the host forward.
#  F2 same dir, task=None | ValueError naming the head file | a silent random head.
#  F3 plain LoRA dir (no task_head.safetensors) | legacy path unchanged: dir passed through,
#     set_processor + set_template bound | regressing every non-attached frozen adapter.
#  F4 twinkle TransformersModel, adapter dir with a head + task='seq_cls' | the head restored at
#     add time (a second load is not required) | add_adapter_to_model(dir) leaving the head random.
class _FrozenModel:

    def __init__(self):
        self.calls = []

    def add_adapter_to_model(self, name, config_or_dir, **kwargs):
        self.calls.append(('add', name, config_or_dir, kwargs))

    def load(self, path, **kwargs):
        self.calls.append(('load', path, kwargs))

    def set_template(self, template, **kwargs):
        self.calls.append(('template', kwargs))

    def set_processor(self, processor_cls, **kwargs):
        self.calls.append(('processor', kwargs))


def _write_adapter(tmp_path, *, with_head):
    import json

    from safetensors.torch import save_file

    adapter = tmp_path / 'adapter'
    adapter.mkdir()
    (adapter / 'adapter_config.json').write_text(json.dumps({
        'peft_type': 'LORA', 'task_type': 'CAUSAL_LM', 'r': 8, 'lora_alpha': 16, 'lora_dropout': 0.0,
        'target_modules': 'all-linear', 'base_model_name_or_path': 'base'}))
    # one trained suffix only: the config's 'all-linear' would wrap far more on another architecture
    save_file({'base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight': torch.zeros(8, 4)},
              str(adapter / 'adapter_model.safetensors'))
    if with_head:
        save_file({'task_head.weight': torch.arange(6, dtype=torch.float32).view(2, 3)},
                  str(adapter / 'task_head.safetensors'))
    return str(adapter)


def test_attached_head_adapter_goes_through_the_adapter_channel_with_its_trained_modules(tmp_path):
    from swift.dev.builder.adapter import configure_frozen_adapter

    adapter = _write_adapter(tmp_path, with_head=True)
    template = object()
    model = _FrozenModel()
    configure_frozen_adapter(model, template, [adapter], role='reward', task='seq_cls', num_labels=3)
    kind, name, config, kwargs = model.calls[0]
    assert kind == 'add' and name == 'reward_adapter'
    assert kwargs == {'is_trainable': False, 'task': 'seq_cls', 'num_labels': 3}
    assert config.target_modules == ['q_proj']  # read off the weights, not re-expanded all-linear
    assert model.calls[1][0] == 'load' and model.calls[1][2] == {'adapter_name': 'reward_adapter', 'load_optimizer': False}
    assert model.calls[2] == ('template', {'adapter_name': 'reward_adapter'})
    assert not any(call[0] == 'processor' for call in model.calls)  # host forward: no explicit position_ids


def test_attached_head_adapter_without_a_declared_task_is_refused(tmp_path):
    from swift.dev.builder.adapter import configure_frozen_adapter

    adapter = _write_adapter(tmp_path, with_head=True)
    with pytest.raises(ValueError, match='task_head.safetensors.*no readout task'):
        configure_frozen_adapter(_FrozenModel(), None, [adapter], role='reward')


def test_plain_frozen_adapter_keeps_the_legacy_path(tmp_path):
    from swift.dev.builder.adapter import configure_frozen_adapter

    adapter = _write_adapter(tmp_path, with_head=False)
    model = _FrozenModel()
    configure_frozen_adapter(model, None, [adapter], role='judge', task='seq_cls', num_labels=2)
    assert model.calls[0] == ('add', 'judge_adapter', adapter, {'is_trainable': False})
    assert model.calls[1] == ('processor', {'adapter_name': 'judge_adapter'})
    assert model.calls[2] == ('template', {'adapter_name': 'judge_adapter'})


def test_transformers_model_restores_the_head_when_added_from_a_directory(tmp_path):
    pytest.importorskip('peft')
    import transformers
    from peft import LoraConfig, get_peft_model
    from safetensors.torch import save_file
    from tokenizers import Tokenizer, models
    from twinkle.model import TransformersModel

    from swift.dev.builder.model import _resolve_model_loader
    from swift.dev.config import ModelConfig

    # A tiny hermetic base (same shape family as tests/v5/component/model/test_model_loading_e2e.py).
    base, adapter = tmp_path / 'base', tmp_path / 'adapter'
    base.mkdir()
    config = transformers.Qwen2Config(
        vocab_size=64, hidden_size=16, intermediate_size=32, num_hidden_layers=2, num_attention_heads=4,
        num_key_value_heads=2, max_position_embeddings=64, tie_word_embeddings=True)
    transformers.Qwen2ForCausalLM(config).save_pretrained(base)
    transformers.PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(models.WordLevel({f'w{i}': i for i in range(64)}, unk_token='w0')),
        unk_token='w0',
        pad_token='w0',
        bos_token_id=1,
        eos_token_id=2).save_pretrained(base)
    # The LoRA is built by peft on this checkpoint: from_pretrained resolves the config's target
    # modules against the real model, so a hand-written adapter file would not load.
    get_peft_model(
        transformers.Qwen2ForCausalLM(config),
        LoraConfig(task_type='CAUSAL_LM', r=4, lora_alpha=8, target_modules=['q_proj'])).save_pretrained(adapter)
    # The trained head saved beside the LoRA weights, in the key form twinkle's head state uses.
    head_weight = torch.arange(32, dtype=torch.float32).view(2, 16)
    save_file({'task_head.classifier.weight': head_weight}, str(adapter / 'task_head.safetensors'))

    loader = _resolve_model_loader(ModelConfig(model=str(base), model_type='qwen2_5'))
    model = TransformersModel(model_id=str(base), model_loader=loader, mixed_precision='no')
    model.add_adapter_to_model('reward_adapter', str(adapter), is_trainable=False, task='seq_cls', num_labels=2)

    head = model._attached_head('reward_adapter')
    assert head is not None and head.classifier.weight.shape == (2, 16)
    # The values read back from disk, not the random init a freshly built Linear carries.
    assert torch.equal(head.classifier.weight.detach().cpu(), head_weight)
