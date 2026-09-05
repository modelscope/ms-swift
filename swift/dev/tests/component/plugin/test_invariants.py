# Copyright (c) ModelScope Contributors. All rights reserved.
"""Invariants of the extension mechanism, asserted against the source instead of promised in prose.

Three ways a plugin system rots, so three tests:

1. A kind is declared and nothing ever reads it -- the extension point exists on paper only.
2. A Config offers a knob that names an implementation, and nothing consumes it. That is the failure
   mode dev inherited: eight legacy plugin fields survived the port as declarations with no reader, so
   ``--agent_template my_template`` parsed, ran, and did nothing. Such a field is allowed to exist, but
   only as a recorded decision with a reason.
3. A name string is handed to twinkle. twinkle's ``construct_class`` resolves a str through
   ``Plugin.load_plugin``, which only understands ``hf://`` / ``ms://`` ids and forces
   trust_remote_code -- so a swift plugin name would either miss or fetch something from a hub. swift
   resolves names itself and hands twinkle constructed objects (or classes).
"""
import dataclasses
import re
from pathlib import Path

import swift.dev
import swift.dev.config as configs
import swift.dev.rewards  # noqa: F401 -- importing the declaring module is what registers its kind
from swift.dev.plugin import PluginRegistry

DEV = Path(swift.dev.__file__).parent

#: Field names shaped like "pick an implementation by name", hence candidates for an extension point.
#: A heuristic on purpose: a new ``*_plugin`` / ``*_funcs`` / ``callbacks`` field must be classified
#: below, and this is what forces that instead of letting it slip in unread.
_LOOKS_LIKE_A_PLUGIN = re.compile(r'(^|_)(plugins?|funcs?|callbacks|metric)$')

#: Candidates whose names do not follow that shape but which do name an implementation.
_ALSO_CANDIDATES = {
    ('ModelConfig', 'custom_register_path'),
    ('TemplateConfig', 'agent_template'),
    ('RLHFConfig', 'multi_turn_scheduler'),
    ('RLHFConfig', 'loss_scale'),
}

#: Consumed, but not through PluginRegistry -- with the consumer named, so the claim is checkable.
WIRED_ELSEWHERE = {
    ('ModelConfig', 'external_plugins'): 'PluginRegistry.load_configured, via TrainAssembly.prepare',
    ('ModelConfig', 'custom_register_path'): 'PluginRegistry.load_configured (joined to external_plugins)',
    ('TemplateConfig', 'loss_scale'): "builders/template.py passes it to legacy get_template, whose own "
    'loss_scale registry resolves the name',
    ('SamplingConfig', 'prm_funcs'): 'swift.dev.reward.get_reward_funcs -> the reward kind',
}

#: Declared and reaching NOTHING. Allowed, but each row states why -- an ignored knob must be a
#: decision, not a surprise. Deleting the field is what removes the row.
UNWIRED = {
    ('TrainConfig', 'callbacks'): 'legacy callbacks are HfTrainer TrainerCallbacks; twinkle drives its own '
    'loop, so there is no object for them to attach to. A dev callback point would be a new design.',
    ('TrainConfig', 'eval_metric'): 'metrics are twinkle Metric objects added to the optimizer status; dev '
    'never reads this name. Wiring it means a metric kind whose base is twinkle Metric.',
    ('TemplateConfig', 'agent_template'): 'agent templates format tool calls inside the legacy template; the '
    'dev template inherits the legacy behavior and exposes no selector.',
    ('RLHFConfig', 'multi_turn_scheduler'): 'multi-turn rollout is not implemented in dev GRPO -- a scheduler '
    'has no loop to schedule.',
    ('RLHFConfig', 'reward_model_plugin'): 'dev PPO scores with reward MODELS built by _build_reward_models; '
    'the legacy per-model plugin hook has no equivalent.',
    ('RLHFConfig', 'loss_scale'): 'the RLHF losses come from twinkle and take no loss_scale; the template-side '
    'loss_scale (TemplateConfig) is the one that is honoured.',
    ('InferConfig', 'metric'): 'InferConfig is not threaded into run_infer at all yet (it takes metric as a '
    "parameter, and the value is a closed 'acc' / 'rouge' choice rather than a registry lookup).",
}


def config_classes():
    """dev's atomic Configs, by name."""
    return {
        name: obj
        for name, obj in vars(configs).items() if dataclasses.is_dataclass(obj) and isinstance(obj, type)
    }


def dev_sources(exclude=('tests', )):
    """dev's own modules -- the places a Config field can be consumed."""
    return [path for path in DEV.rglob('*.py') if not set(path.relative_to(DEV).parts) & set(exclude)]


def readers_of(field, *, owner=None, skip=('config', )):
    """Files that read this Config field, excluding the Config declarations themselves.

    ``owner`` qualifies the search with the Config's conventional variable name
    (``RLHFConfig`` -> ``rlhf_config.loss_scale``), which is what tells two same-named fields on
    different Configs apart -- ``loss_scale`` exists on both TemplateConfig (honoured) and RLHFConfig
    (ignored), and an unqualified search credits the second with the first's consumer.
    """
    prefix = rf'\b{re.escape(_config_variable(owner))}\.' if owner else r'\.'
    pattern = re.compile(prefix + re.escape(field) + r'\b')
    hits = []
    for path in dev_sources():
        if set(path.relative_to(DEV).parts) & set(skip):
            continue
        if pattern.search(path.read_text(encoding='utf-8')):
            hits.append(str(path.relative_to(DEV)))
    return hits


def _config_variable(config_name):
    """``RLHFConfig`` -> ``rlhf_config``: the name every recipe gives that Config."""
    return re.sub(r'(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])', '_', config_name).lower()


def candidate_fields():
    """(Config name, field) pairs that look like an extension point selector."""
    found = set(_ALSO_CANDIDATES)
    for name, config in config_classes().items():
        for field in dataclasses.fields(config):
            if _LOOKS_LIKE_A_PLUGIN.search(field.name):
                found.add((name, field.name))
    return found


def test_every_extension_point_is_selectable_and_consumed():
    """A declared kind must have a Config field that selects it, and a consumer that reads it.

    Otherwise the point exists only in the registry: nothing lets a user pick an implementation, and
    nothing would notice if they did.
    """
    assert PluginRegistry.KINDS, 'no plugin kinds are registered -- did the declaring module move?'
    by_name = config_classes()
    for kind in PluginRegistry.KINDS.values():
        assert kind.config_field, f'kind {kind.name!r} declares no config_field: nothing can select it.'
        owners = [
            name for name, config in by_name.items()
            if kind.config_field in {f.name
                                     for f in dataclasses.fields(config)}
        ]
        assert owners, (f'kind {kind.name!r} selects on {kind.config_field!r}, which no Config declares.')
        assert readers_of(kind.config_field), (f'kind {kind.name!r}: no dev module reads '
                                               f'{kind.config_field!r} -- the point is unreachable.')


def test_no_plugin_field_is_silently_ignored():
    """Every plugin-shaped Config field is either wired, or listed as unwired WITH a reason."""
    kind_fields = {kind.config_field for kind in PluginRegistry.KINDS.values()}
    unclassified = sorted(
        pair for pair in candidate_fields()
        if pair[1] not in kind_fields and pair not in WIRED_ELSEWHERE and pair not in UNWIRED)
    assert not unclassified, (f'these Config fields name an implementation but are classified nowhere: '
                             f'{unclassified}. Wire them to a plugin kind, or add them to UNWIRED with the '
                             f'reason they do nothing.')
    for pair, reason in {**WIRED_ELSEWHERE, **UNWIRED}.items():
        assert reason.strip(), f'{pair} needs a reason, not an empty string.'


def test_the_tables_still_describe_the_code():
    """Both tables rot the moment the code moves, so both are checked against it.

    A field that was wired since the row was written is the interesting direction: the row would keep
    claiming the knob does nothing long after it started working.
    """
    declared = {(name, field.name) for name, config in config_classes().items() for field in dataclasses.fields(config)}
    stale = sorted(pair for pair in {**WIRED_ELSEWHERE, **UNWIRED} if pair not in declared)
    assert not stale, f'these rows name Config fields that no longer exist: {stale}. Delete the rows.'

    for pair in WIRED_ELSEWHERE:
        assert readers_of(pair[1], owner=pair[0]), f'{pair} is listed as wired, but no dev module reads it.'
    now_wired = {pair: readers for pair in UNWIRED if (readers := readers_of(pair[1], owner=pair[0]))}
    assert not now_wired, (f'these fields are listed as doing nothing, but something reads them now: '
                           f'{now_wired}. Move them out of UNWIRED.')


#: twinkle setters that run their argument through ``construct_class`` (str -> hub download).
_TWINKLE_SETTERS = re.compile(r'\.(set_loss|set_template|set_processor|set_optimizer|set_lr_scheduler|add_metric)'
                              r'\(\s*[\'"]')


def test_no_name_string_is_handed_to_twinkles_loader():
    """dev resolves plugin names itself; twinkle receives objects or classes, never a name.

    A str reaching ``construct_class`` is routed to ``Plugin.load_plugin``, which understands only
    ``hf://`` / ``ms://`` ids and forces trust_remote_code -- so a swift plugin name would fail to
    resolve, or worse, resolve to something fetched from a hub.
    """
    offenders = []
    for path in dev_sources():
        for lineno, line in enumerate(path.read_text(encoding='utf-8').splitlines(), start=1):
            if _TWINKLE_SETTERS.search(line):
                offenders.append(f'{path.relative_to(DEV)}:{lineno}: {line.strip()}')
    assert not offenders, ('these call sites hand twinkle a name string; resolve it to a class or instance '
                           'first (PluginRegistry.get / .resolve):\n' + '\n'.join(offenders))
