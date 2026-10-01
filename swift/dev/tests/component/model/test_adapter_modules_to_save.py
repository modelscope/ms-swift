"""Guards for the seq_cls / reranker classification head landing in LoRA's modules_to_save.

LoRA only trains and persists ``target_modules``; a seq_cls/reranker run builds a fresh
classification head on top of the base model, so unless that head is listed in ``modules_to_save``
it is neither trained nor saved -- the reloaded checkpoint carries a randomly-initialized head and
the model's predictions are meaningless. dev cannot probe the head's real name from the built model
here (under Ray ``apply_tuner`` runs against a driver-side proxy and the config is serialized before
the workers build the adapter), so it lists both candidate HF head names; peft matches
``modules_to_save`` by ``endswith`` and silently ignores the entry the family lacks.

Pure config building -- no weights, no GPU, no Ray.
"""
import pytest

from swift.dev.adapter import (_HEAD_SAVING_TASK_TYPES, _SEQ_CLS_HEAD_MODULES, _build_adapter_config,
                               _lora_common_kwargs, apply_tuner)
from swift.dev.config import TunerConfig


# --- _lora_common_kwargs: the head-append seam --------------------------------------------------------


def test_seq_cls_appends_both_head_names():
    kwargs = _lora_common_kwargs(TunerConfig(), task_type='seq_cls')
    assert set(_SEQ_CLS_HEAD_MODULES).issubset(set(kwargs['modules_to_save']))


@pytest.mark.parametrize('task_type', _HEAD_SAVING_TASK_TYPES)
def test_every_head_saving_task_type_appends(task_type):
    kwargs = _lora_common_kwargs(TunerConfig(), task_type=task_type)
    assert set(_SEQ_CLS_HEAD_MODULES).issubset(set(kwargs['modules_to_save']))


def test_causal_lm_leaves_modules_to_save_unset():
    # The default path: no head, and an empty user list collapses to None (peft's "nothing extra").
    assert _lora_common_kwargs(TunerConfig(), task_type=None)['modules_to_save'] is None
    assert _lora_common_kwargs(TunerConfig(), task_type='causal_lm')['modules_to_save'] is None


def test_generative_reranker_is_excluded():
    # generative_reranker keeps the CausalLM vocab head -- no new module to save.
    assert _lora_common_kwargs(TunerConfig(), task_type='generative_reranker')['modules_to_save'] is None


def test_user_modules_to_save_are_preserved_and_not_duplicated():
    cfg = TunerConfig(modules_to_save=['score', 'embed_tokens'])
    kwargs = _lora_common_kwargs(cfg, task_type='seq_cls')
    saved = kwargs['modules_to_save']
    # 'score' was already there -> not duplicated; 'classifier' added; user's 'embed_tokens' kept.
    assert saved.count('score') == 1
    assert 'classifier' in saved
    assert 'embed_tokens' in saved


def test_cfg_modules_to_save_is_not_mutated():
    cfg = TunerConfig()
    _lora_common_kwargs(cfg, task_type='seq_cls')
    # The list is copied, not mutated, so a reused TunerConfig keeps exactly what the user set.
    assert cfg.modules_to_save == []


# --- _build_adapter_config: the head reaches the built peft config ------------------------------------


def test_lora_config_carries_head():
    config = _build_adapter_config(TunerConfig(), task_type='seq_cls')
    assert set(_SEQ_CLS_HEAD_MODULES).issubset(set(config.modules_to_save))


def test_lora_config_without_task_type_has_no_head():
    config = _build_adapter_config(TunerConfig())
    assert not config.modules_to_save


def test_adalora_config_carries_head():
    config = _build_adapter_config(TunerConfig(tuner='adalora'), num_training_steps=10, task_type='reranker')
    assert set(_SEQ_CLS_HEAD_MODULES).issubset(set(config.modules_to_save))


# --- apply_tuner: task_type is threaded to the model's adapter ----------------------------------------


class _RecordingModel:
    """Captures the peft config handed to add_adapter_to_model."""

    def __init__(self):
        self.config = None
        self.kwargs = None

    def add_adapter_to_model(self, adapter_name, adapter_config, **kwargs):
        self.config = adapter_config
        self.kwargs = kwargs


def test_apply_tuner_forwards_task_type_head():
    model = _RecordingModel()
    apply_tuner(model, TunerConfig(), task_type='seq_cls')
    assert set(_SEQ_CLS_HEAD_MODULES).issubset(set(model.config.modules_to_save))


def test_apply_tuner_default_has_no_head():
    model = _RecordingModel()
    apply_tuner(model, TunerConfig())
    assert not model.config.modules_to_save
