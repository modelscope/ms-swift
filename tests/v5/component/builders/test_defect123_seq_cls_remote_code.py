"""defect123 — seq_cls model-class resolution for remote-code reward checkpoints.

A process/sequence reward model loaded through remote code (Qwen2.5-Math-PRM, internlm2-reward)
maps only ``AutoModel`` in its ``auto_map`` and brings its own head. Forcing
``AutoModelForSequenceClassification`` makes ``from_pretrained`` raise "Unrecognized configuration
class" because that auto class is absent from the auto_map. ``_apply_seq_cls_head`` must honor the
family loader's declared class when it resolves the config's auto_map, and pass trust_remote_code so
the worker-side load resolves the checkpoint's own class.

PLAN (case | input | expected | failure it kills):
 L1 remote-code auto_map={AutoConfig,AutoModel}, loader->AutoModel | model_cls=AutoModel + trust
    | the forced seq-cls auto class crashing on a remote-code reward config (the actor death).
 L2 native config (no auto_map) | AutoModelForSequenceClassification, no trust kwarg | regressing
    the plain seq_cls head path.
 L3 remote-code auto_map includes AutoModelForSequenceClassification | stays SeqCls | overriding a
    checkpoint that genuinely maps the seq-cls auto class.
 L4 no loader + remote-code auto_map=AutoModel only | stays SeqCls (nothing to resolve) | calling
    resolve_model_cls on a None loader.
Independent oracle: the transformers auto-factory rule (cls.__name__ must be in config.auto_map).
"""
from types import SimpleNamespace

from transformers import AutoModel, AutoModelForSequenceClassification


class _StubLoader:

    def __init__(self, cls):
        self._cls = cls

    def resolve_model_cls(self, config=None):
        return self._cls


def _cfg(auto_map):
    return SimpleNamespace(auto_map=auto_map, pad_token_id=0, num_labels=None, problem_type=None,
                           tie_word_embeddings=True, text_config=None)


def _mc():
    return SimpleNamespace(num_labels=1, task_type='seq_cls', problem_type=None, model='x', model_revision=None)


def _apply(auto_map, loader):
    from swift.dev.builder.model import _apply_seq_cls_head

    kwargs = {}
    _apply_seq_cls_head(kwargs, _mc(), _cfg(auto_map), loader)
    return kwargs


def test_defect123_remote_code_automodel_only_honors_loader():
    kwargs = _apply({'AutoConfig': 'c.C', 'AutoModel': 'm.M'}, _StubLoader(AutoModel))
    assert kwargs['model_cls'] == 'AutoModel'
    assert kwargs['trust_remote_code'] is True


def test_defect123_native_config_keeps_seq_cls():
    kwargs = _apply(None, _StubLoader(AutoModel))
    assert kwargs['model_cls'] == 'AutoModelForSequenceClassification'
    assert 'trust_remote_code' not in kwargs


def test_defect123_remote_code_maps_seq_cls_keeps_seq_cls():
    kwargs = _apply({'AutoConfig': 'c.C', 'AutoModelForSequenceClassification': 'm.M'}, _StubLoader(AutoModel))
    assert kwargs['model_cls'] == 'AutoModelForSequenceClassification'
    assert 'trust_remote_code' not in kwargs


def test_defect123_no_loader_keeps_seq_cls():
    kwargs = _apply({'AutoConfig': 'c.C', 'AutoModel': 'm.M'}, None)
    assert kwargs['model_cls'] == 'AutoModelForSequenceClassification'
