# Copyright (c) ModelScope Contributors. All rights reserved.
"""Every hand-writable swift plugin kind in ONE file, loaded by ``--external_plugins``.

swift's extension points all resolve the same way (``swift.dev.naming.resolve_plugin_class`` /
``swift.dev.plugin.PluginRegistry``): a name registered by an imported file, a class, or an external
source. ``--external_plugins THIS_FILE`` imports this module before anything is built
(``PluginRegistry.load_configured``), so every ``@register`` below has run by the time a run selects an
implementation by name. One file therefore carries all six kinds; a real project usually splits them.

    swift infer ... --external_plugins examples/v5/infer/custom_plugins.py \
        --model_type demo_qwen2 --template demo_chatml --dataset demo_synthetic \
        --orm length_bonus async_length_bonus --tools word_count --max_turns 4

The kinds demonstrated here, and the field that selects each:

  * reward (sync)   -- ``RewardPlugin.__call__``                       -> ``--orm`` / ``--prm`` by name
  * reward (async)  -- the SAME base, ``async def __call__``           -> ``--orm`` / ``--prm`` by name
  * tool            -- ``ToolPlugin.build(env)``                       -> ``--tools`` by name
  * model loader    -- ``ModelLoader`` + ``@register_model``           -> ``--model_type`` by name
  * dataset loader  -- ``DatasetLoader`` + ``@register_dataset``       -> ``--dataset`` by name
  * template        -- ``register_template(TemplateMeta(...))``        -> ``--template`` by name

Two things are deliberately NOT here:

  * A custom ``--sampler`` lives in ``custom_sampler.py``, a SEPARATE file. The 'sampler' kind is
    declared lazily (importing ``twinkle.sampler`` pulls in vLLM + torch), so a sampler is selected by
    pointing ``--sampler`` AT ITS FILE (resolved through the same loader), never by ``@register`` at
    import time -- and putting the heavy base import in THIS file would tax every ``--external_plugins``
    run, including training runs that build no sampler.
  * ``loss`` / ``lr_scheduler`` / ``metric`` are NOT external plugin kinds. ``--loss`` / ``--lr_scheduler``
    name a member of a fixed built-in roster (``swift.dev.naming.resolve_loss`` / ``resolve_scheduler``);
    an unknown name raises rather than importing a file. There is no metric selector dev reads at all.
    So there is nothing to register for them, and no honest example to write.
"""
from __future__ import annotations

import asyncio
from typing import Any, Dict, List

from datasets import Dataset as HfDataset

from swift.dev.dataset.loader import DatasetLoader, register_dataset
from swift.dev.model.loader import ModelLoader, register_model
from swift.dev.plugin import PluginRegistry, RewardPlugin, ToolPlugin
from swift.template import TemplateMeta, register_template
from twinkle.data_format.message import Tool as ToolInfo

# --- reward: a sync rule and an async rule, one extension point ---------------------------
# A reward plugin scores model completions against the dataset's other columns: __call__ receives the
# list of completions plus every dataset column as a keyword (so `solution=`, `messages=`, ... arrive
# when the dataset has them), and returns one float per completion, in order. Returning None for an
# item records it as NaN rather than a guessed 0.


@PluginRegistry.register('reward', 'length_bonus')
class LengthBonus(RewardPlugin):
    """A synchronous reward: longer answers score higher, capped at 1.0. Selected with ``--orm length_bonus``."""

    def __call__(self, completions: List[str], **columns: Any) -> List[float]:
        return [min(len(text) / 100.0, 1.0) for text in completions]


@PluginRegistry.register('reward', 'async_length_bonus')
class AsyncLengthBonus(RewardPlugin):
    """The SAME contract written ``async`` -- there is no separate async base to choose.

    ``swift.dev.reward.compute_rewards_per_func`` detects the returned coroutines and resolves a batch
    with one ``asyncio.gather``, so the I/O of many completions overlaps instead of running one after
    another. Use it when scoring does real I/O (an API call, a database lookup); the ``await`` here stands
    in for that. A run may mix sync and async rewards freely in one ``--orm`` list.
    """

    async def __call__(self, completions: List[str], **columns: Any) -> List[float]:
        await asyncio.sleep(0)  # the yield point a real async scorer's I/O would provide
        return [min(len(text) / 200.0, 1.0) for text in completions]


# --- tool: a multi-turn rollout tool ------------------------------------------------------
# A ToolPlugin is built once per run and asked for its tools per episode: build(env) receives the leased
# sandbox Env and returns the twinkle Tool objects the model may call that turn (return [] to contribute
# nothing). Each Tool implements __call__(tool_name, arguments) -> str (the observation fed back to the
# model) and tool_info() -> the OpenAI-shaped schema advertised in the prompt. Tools are multi-turn, so a
# run naming one also needs --max_turns.


class _WordCount:
    """A minimal twinkle ``Tool``: count the words of a passage and return the tally as a string."""

    def __call__(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        text = str(arguments.get('text', ''))
        return str(len(text.split()))

    def tool_info(self) -> ToolInfo:
        return {
            'type': 'function',
            'function': {
                'name': 'word_count',
                'description': 'Count the whitespace-separated words in a passage and return the number.',
                'parameters': {
                    'type': 'object',
                    'properties': {
                        'text': {
                            'type': 'string',
                            'description': 'The passage to count words in.'
                        }
                    },
                    'required': ['text'],
                },
            },
        }


@PluginRegistry.register('tool', 'word_count')
class WordCountTools(ToolPlugin):
    """Expose the word_count tool. Selected with ``--tools word_count``. It touches no workspace, so the
    leased env is ignored; a tool that acts in the sandbox would wrap ``env`` (see ``EnvTool.from_env``)."""

    def build(self, env: Any) -> List[Any]:
        return [_WordCount()]


# --- model loader: a family declaration ---------------------------------------------------
# A ModelLoader is the per-family unit: it declares which checkpoint ids it covers, the template it
# speaks, and the transformers class it loads with, and overrides build_*/process_* hooks only where the
# plain AutoConfig/AutoProcessor/from_pretrained path differs. --model_type selects a registered family by
# name; a name swift does not know is treated as an external class/source and resolved through the loader.


@register_model
class DemoQwen2Loader(ModelLoader):
    """A thin Qwen2 family, registered under its own ``model_type`` so ``--model_type demo_qwen2`` picks it.

    Everything but the identity is the plain transformers path: ``config_cls`` / ``processor_cls`` fall
    back to Auto*, and ``model_cls`` names the class ``build_model`` loads. A real family that needs a
    specific config/processor class or post-load fixups points ``config_cls``/``processor_cls`` at them or
    overrides ``process_model``.
    """

    model_type = 'demo_qwen2'
    models = ['Qwen/Qwen2.5-0.5B-Instruct']
    architectures = ['Qwen2ForCausalLM']
    template = 'qwen2_5'
    model_cls = 'transformers:Qwen2ForCausalLM'


# --- dataset loader: a family that synthesises its own rows -------------------------------
# A DatasetLoader declares the ids/paths it covers and how to turn them into standard rows. Most families
# differ only in *what* they are (ids, subsets, which raw column means `query`) and need no code at all --
# those live in dataset_info.json. This one overrides the build_dataset hook to synthesise rows in code,
# which is what a dataset with no hub home (a generator, an in-memory fixture) does. --dataset selects a
# registered family by name; an unregistered name is treated as a hub id / file / directory, not a class.


@register_dataset
class DemoSyntheticLoader(DatasetLoader):
    """A self-contained dataset: ``--dataset demo_synthetic`` loads these rows with no hub access.

    Rows are already in the standard ``messages`` layout, so the auto-detecting preprocessor passes them
    through. Overriding ``build_dataset`` (rather than declaring ``datasets``/``dataset_paths``) is the hook
    a dataset that produces its own rows uses; the returned object is any ``datasets`` Dataset.
    """

    dataset_type = 'demo_synthetic'

    def build_dataset(self, subset, split, **kwargs):
        rows = [{
            'messages': [{
                'role': 'user',
                'content': question
            }]
        } for question in ('What is 2+2?', 'Name a primary colour.', 'Say hello in one word.')]
        return HfDataset.from_list(rows)


# --- template: a chat format --------------------------------------------------------------
# A template is registered as data (a TemplateMeta), not as a PluginRegistry kind, because dev derives its
# training-time labels from the legacy template machinery. register_template writes it into the legacy
# TEMPLATE_MAPPING that builders/template.py resolves; --template selects it by name. The Prompt lists mix
# literal strings with token-attribute placeholders (e.g. [['eos_token_id']]) resolved against the tokenizer.
#
# agent_template declares the tool-call format this template renders and parses. It is REQUIRED to drive a
# multi-turn tool rollout (as this example does with --tools word_count): the rollout asks the template for
# its agent_template to bake the tool schema into the prompt and to parse the model's tool calls, and a
# template with none raises. dev removed the legacy --agent_template flag, so this TemplateMeta field is the
# only place to set it. 'hermes' is the format Qwen2.5 speaks; a plain generation run may leave it None.

register_template(
    TemplateMeta(
        template_type='demo_chatml',
        prefix=[],
        prompt=['<|im_start|>user\n{{QUERY}}<|im_end|>\n<|im_start|>assistant\n'],
        chat_sep=['<|im_end|>\n'],
        suffix=['<|im_end|>'],
        system_prefix=['<|im_start|>system\n{{SYSTEM}}<|im_end|>\n'],
        agent_template='hermes',
    ))