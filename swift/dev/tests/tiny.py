# Copyright (c) ModelScope Contributors. All rights reserved.
"""Offline stand-ins for a real checkpoint and a real dataset.

Tests that *run* training (as opposed to inspecting a config) used to each download a real
0.5B checkpoint. That makes the suite depend on the hub being up, on ~1 GB of cache per model, and
on a warm-up that dwarfs the thing under test. Everything here is built locally instead: a 4-layer
model with random weights is enough to prove that a code path is *wired*, which is what feature and
capability tests assert. Numerical claims still need real weights -- see ``feature/sft/test_alignment.py``.

Nothing here imports ``swift.dev``: legacy swift's tests want the same two stand-ins and have no
``ModelLoader`` registry to build a model through, so this module stays on the plain transformers path
and the dev-only way of building a *family's own* architecture lives next door in ``tiny_loader.py``,
reaching :meth:`TinyModel.build` as its ``builder`` argument.
"""
import importlib
import math
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import json

#: How the model itself gets constructed. A builder receives the snapshot directory holding a real
#: checkpoint's config/tokenizer files and the size fields to force on it, and returns a fresh,
#: randomly-initialised model. Everything else -- which tokenizer, which sizes, where the result is
#: written -- stays with :meth:`TinyModel.build`.
Builder = Callable[[str, Dict[str, Any]], Any]


class TinyModel:
    """Build a 4-layer randomly-initialised model and save it as an ordinary model directory.

    The tokenizer is a *real* one, so templates render real chat markup and ``apply_chat_template``
    stays a meaningful reference. That keeps ``vocab_size`` at ~151k, and the embedding then
    dominates the parameter count -- which is why the dimensions below are as small as they are and
    why weights are tied and stored in bf16. Measured on Qwen2:

        hidden=256, untied, fp32 -> 80.0M params, 320 MB
        hidden=128, tied,   bf16 -> 39.4M params,  79 MB   <- these settings

    ``num_attention_heads`` and ``num_key_value_heads`` stay divisible by 2 so a tp=2 / cp=2 run on
    a two-GPU slot can shard them.

    Weights are randomly initialised, NOT left uninitialised: ``no_init_weights()`` hands back
    whatever was in the allocated memory, which measured non-finite and drove the first loss to nan.
    Letting ``from_config`` run the real initialiser is both correct and faster -- 1.45s versus
    3.74s -- and lands the first loss on ``ln(vocab_size)``, which is what ``initial_loss`` asserts.
    """

    LAYERS = 4
    TOKENIZER = 'Qwen/Qwen2.5-0.5B-Instruct'
    CONFIG_CLS = 'transformers.Qwen2Config'
    #: Every dense architecture is ambiguous in the registry (``Qwen2ForCausalLM`` matches both
    #: ``qwen2`` and ``qwen2_gte``; ``LlamaForCausalLM`` matches four), and a locally built directory
    #: has no hub id to disambiguate it -- so callers must always pass model_type explicitly.
    MODEL_TYPE = 'qwen2'
    TEMPLATE = 'qwen2_5'
    DIMS = {
        'hidden_size': 128,
        'intermediate_size': 256,
        'num_attention_heads': 4,
        'num_key_value_heads': 2,
        'max_position_embeddings': 1024,
        'tie_word_embeddings': True,
    }

    @staticmethod
    def tokenizer_dir(model_id: Optional[str] = None) -> str:
        """Snapshot only the tokenizer/config files -- never the weights."""
        from modelscope import snapshot_download
        return snapshot_download(
            model_id or TinyModel.TOKENIZER,
            allow_patterns=['*.json', '*.txt', '*.jinja', '*.model', '*.py'],
        )

    @staticmethod
    def transformers_builder(config_cls: Optional[str] = None) -> Builder:
        """The default builder: instantiate a config class directly, then ``from_config`` it.

        ``config_cls`` is a dotted path (``transformers.Qwen3MoeConfig``) so a model table can name an
        architecture declaratively. Deliberately depends on nothing but transformers -- legacy swift
        wants the same tiny checkpoint and has no :class:`~swift.dev.model.loader.ModelLoader`
        registry to build it through, so the default path must stay free of dev-only concepts. The
        snapshot directory is unused here: the architecture is built from scratch rather than read off
        the real checkpoint (which is what the loader-backed builder in ``tiny_loader.py`` does).
        """

        def _build(snapshot_dir: str, dims: Dict[str, Any]):
            from transformers import AutoModelForCausalLM
            module, _, name = (config_cls or TinyModel.CONFIG_CLS).rpartition('.')
            config = getattr(importlib.import_module(module), name)(**dims)
            return AutoModelForCausalLM.from_config(config)

        return _build

    @staticmethod
    def build(dest: Union[str, Path],
              config_cls: Optional[str] = None,
              *,
              tokenizer_id: Optional[str] = None,
              builder: Optional[Builder] = None,
              **overrides) -> str:
        """Write a loadable model directory to ``dest`` and return its path.

        ``overrides`` go straight into the config, which is how a MoE row shrinks ``num_experts`` or
        an MTP row adds its own knobs. ``config_cls`` selects the architecture for the default
        builder; pass ``builder`` instead to construct the model some other way -- dev's own model
        tables hand in ``tiny_loader.loader_builder(model_type)``, which inherits the registered
        family loader so a row names a family rather than restating its transformers classes.
        """
        import torch
        from transformers import AutoTokenizer

        tok_dir = TinyModel.tokenizer_dir(tokenizer_id)
        tokenizer = AutoTokenizer.from_pretrained(tok_dir, trust_remote_code=True)

        dims = {
            'num_hidden_layers': TinyModel.LAYERS,
            'vocab_size': len(tokenizer),
            'dtype': torch.bfloat16,
            **TinyModel.DIMS,
            **overrides,
        }
        model = (builder or TinyModel.transformers_builder(config_cls))(tok_dir, dims)

        dest = Path(dest)
        model.save_pretrained(dest)
        tokenizer.save_pretrained(dest)
        return str(dest)

    @staticmethod
    def initial_loss(model_dir: Union[str, Path]) -> float:
        """The loss a randomly-initialised model must start near: ``ln(vocab_size)``.

        Any real training path that reports a first loss far from this is not reading the labels it
        thinks it is -- a shifted, masked or double-normalised loss shows up here immediately, which
        a plain "is it finite" check would miss.
        """
        with (Path(model_dir) / 'config.json').open() as f:
            return math.log(json.load(f)['vocab_size'])


class TinyData:
    """Write the smallest dataset each task type accepts, in dev's own column names."""

    PROMPTS = ('What is 2+2?', 'Name a colour.', 'Say hello.', 'Count to three.')
    ANSWERS = ('4', 'blue', 'hello', 'one two three')

    @staticmethod
    def _dump(path: Union[str, Path], rows: List[dict]) -> str:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('w', encoding='utf-8') as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + '\n')
        return str(path)

    @staticmethod
    def _turns(i: int) -> List[dict]:
        n = len(TinyData.PROMPTS)
        return [
            {
                'role': 'user',
                'content': TinyData.PROMPTS[i % n]
            },
            {
                'role': 'assistant',
                'content': TinyData.ANSWERS[i % n]
            },
        ]

    @staticmethod
    def sft(path: Union[str, Path], n: int = 8) -> str:
        return TinyData._dump(path, [{'messages': TinyData._turns(i)} for i in range(n)])

    @staticmethod
    def pretrain(path: Union[str, Path], n: int = 8) -> str:
        rows = [{'messages': [{'role': 'assistant', 'content': f'{p} {a}'}]} for p, a in zip(TinyData.PROMPTS * n,
                                                                                            TinyData.ANSWERS * n)]
        return TinyData._dump(path, rows[:n])

    @staticmethod
    def preference(path: Union[str, Path], n: int = 8) -> str:
        """dpo / orpo / simpo / cpo / rm: a chosen turn plus a ``rejected_response``."""
        rows = [{'messages': TinyData._turns(i), 'rejected_response': 'wrong'} for i in range(n)]
        return TinyData._dump(path, rows)

    @staticmethod
    def prompt_only(path: Union[str, Path], n: int = 8) -> str:
        """ppo / grpo / sampling / infer: prompts with no assistant turn to learn from."""
        rows = [{'messages': [TinyData._turns(i)[0]]} for i in range(n)]
        return TinyData._dump(path, rows)

    @staticmethod
    def seq_cls(path: Union[str, Path], n: int = 8, num_labels: int = 2) -> str:
        """seq_cls: one integer ``label`` per sequence rather than a target turn."""
        rows = [{'messages': [TinyData._turns(i)[0]], 'label': i % num_labels} for i in range(n)]
        return TinyData._dump(path, rows)

    @staticmethod
    def embedding(path: Union[str, Path], n: int = 8) -> str:
        """embedding: ``positive_messages`` / ``negative_messages`` are lists OF message lists.

        Shape copied from a registered loader (``StsbPreprocessor``): each entry is its own
        conversation holding a single user turn, not a user/assistant pair.
        """
        rows = [{
            'messages': [{
                'role': 'user',
                'content': TinyData.PROMPTS[i % len(TinyData.PROMPTS)]
            }],
            'positive_messages': [[{
                'role': 'user',
                'content': TinyData.ANSWERS[i % len(TinyData.ANSWERS)]
            }]],
            'negative_messages': [[{
                'role': 'user',
                'content': 'unrelated text'
            }]],
        } for i in range(n)]
        return TinyData._dump(path, rows)
