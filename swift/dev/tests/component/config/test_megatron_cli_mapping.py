"""Fast tests for the Megatron-args -> dev-Config mapping (swift.dev.cli.megatron).

Mirrors test_cli_mapping.py: a SimpleNamespace stands in for a parsed MegatronSftArguments, so
nothing loads a model or dataset and these run in the cheap suite.

The load-bearing test here is the COVERAGE GUARD (TestCoverageGuard): it asserts every dev Config
field is accounted for as name-hit / renamed / derived / explicitly-absent. That is what stops the
next person who adds a Config field from silently leaving it unmapped -- the exact failure mode
that made cli/sft.py need a `train_iters` sentinel in the first place. The guard's own
fail-when-broken behaviour is tested too (a whitelist that never fires is worthless).
"""
from __future__ import annotations
import dataclasses
from types import SimpleNamespace

import pytest

from swift.dev.cli.megatron import (
    ABSENT,
    DERIVED,
    RENAMES,
    SUPERSEDED,
    _decay_style_to_swift,
    _derive_gradient_accumulation_steps,
    audit_coverage,
    megatron_args_to_configs,
    parse_megatron_configs,
)
from swift.dev.config import (
    CheckpointConfig,
    ConvertConfig,
    DatasetConfig,
    DistributedConfig,
    MegatronConfig,
    ModelConfig,
    MoEConfig,
    TemplateConfig,
    TrainConfig,
    TunerConfig,
    process_configs,
)

_CONFIGS = (ModelConfig, TemplateConfig, DatasetConfig, TrainConfig, DistributedConfig, CheckpointConfig, TunerConfig,
            MegatronConfig, MoEConfig)


def _legacy_arg_names():
    from swift.megatron.arguments import MegatronSftArguments

    return {field.name for field in dataclasses.fields(MegatronSftArguments)}


def test_self_parser_maps_legacy_compat_fields_without_legacy_arguments():
    model, _, _, train, dist, checkpoint, logging, tuner, quantize, megatron, moe = parse_megatron_configs([
        '--model', 'm', '--dataset', 'd', '--tuner_type', 'full', '--bf16', 'true', '--attention_backend', 'fused',
        '--lr', '0.0002', '--train_iters', '20', '--micro_batch_size', '2', '--global_batch_size', '8',
        '--tensor_model_parallel_size', '2', '--save_steps', '0.25', '--logging_steps', '2',
    ], world_size=4)
    assert model.torch_dtype == 'bfloat16' and model.attn_impl == 'fused'
    assert train.learning_rate == 0.0002 and train.max_steps == 20
    assert train.gradient_accumulation_steps == 2
    assert dist.backend == 'megatron' and dist.nproc_per_node == 4
    assert checkpoint.save_steps == 0.25
    assert logging.logging_steps == 2
    assert tuner is None
    assert quantize.quant_method is None
    assert isinstance(megatron, MegatronConfig)
    assert isinstance(moe, MoEConfig)


def test_self_parser_rejects_legacy_field_without_dev_consumer():
    with pytest.raises(ValueError, match='no dev Config consumer'):
        parse_megatron_configs(['--model', 'm', '--dataset', 'd', '--num_layers', '24'])


def test_self_parser_accepts_tracker_config():
    configs = parse_megatron_configs(['--model', 'm', '--dataset', 'd', '--report_to', 'wandb'])
    assert configs[6].report_to == ['wandb']


@pytest.mark.parametrize(('command', 'argv'), [
    ('megatron_sft', ['--model', 'm', '--dataset', 'd', '--mcore_model', '/legacy/mcore']),
    ('megatron_pt', ['--model', 'm', '--dataset', 'd', '--mcore_adapter', '/legacy/adapter']),
])
def test_native_mcore_training_checkpoints_fail_with_conversion_hint(command, argv):
    with pytest.raises(ValueError, match='megatron export --to_hf'):
        parse_megatron_configs(argv, command=command)


def test_native_mcore_reference_checkpoint_fails_with_conversion_hint():
    from swift.dev.cli.rlhf import parse_rlhf_configs

    with pytest.raises(ValueError, match='megatron export --to_hf'):
        parse_rlhf_configs(
            ['--model', 'm', '--dataset', 'd', '--mcore_ref_model', '/legacy/reference'], megatron=True)


@pytest.mark.parametrize(('finetune', 'resume_only_model'), [('true', True), ('false', False)])
def test_megatron_finetune_derives_dev_checkpoint_resume_mode(finetune, resume_only_model):
    configs = parse_megatron_configs([
        '--model', 'm', '--dataset', 'd', '--resume_from_checkpoint', '/tmp/dev-checkpoint', '--finetune', finetune
    ])
    model, template, dataset, train, dist, checkpoint, _, tuner, quantize, megatron, _ = configs
    process_configs(
        model,
        template,
        dataset,
        train,
        dist,
        checkpoint,
        tuner,
        megatron_config=megatron,
        quantize_config=quantize)
    assert checkpoint.resume_only_model is resume_only_model


def test_megatron_finetune_resume_conflicts_fail_and_false_requires_checkpoint():
    configs = parse_megatron_configs([
        '--model', 'm', '--dataset', 'd', '--resume_from_checkpoint', '/tmp/dev-checkpoint', '--finetune', 'true',
        '--resume_only_model', 'false'
    ])
    model, template, dataset, train, dist, checkpoint, _, tuner, quantize, megatron, _ = configs
    with pytest.raises(ValueError, match='conflicts with resume_only_model'):
        process_configs(
            model,
            template,
            dataset,
            train,
            dist,
            checkpoint,
            tuner,
            megatron_config=megatron,
            quantize_config=quantize)

    configs = parse_megatron_configs(['--model', 'm', '--dataset', 'd', '--finetune', 'false'])
    model, template, dataset, train, dist, checkpoint, _, tuner, quantize, megatron, _ = configs
    with pytest.raises(ValueError, match='requires --resume_from_checkpoint'):
        process_configs(
            model,
            template,
            dataset,
            train,
            dist,
            checkpoint,
            tuner,
            megatron_config=megatron,
            quantize_config=quantize)


def _args(**overrides):
    """A parsed-MegatronSftArguments stand-in.

    Defaults mirror the real surface for every field the mapping reads BY NAME (the renames, the
    GA derivation inputs, the reject checks and tuner dispatch). Fields not set here are simply
    absent, which exercises the same "leave the Config default" path as a None value.
    """
    base = {
        'model': 'm',
        'dataset': ['d'],
        'tuner_type': 'full',
        'torch_dtype': None,
        # renames
        'lr': None,
        'train_iters': None,
        'micro_batch_size': 1,
        'adam_eps': None,
        'lr_warmup_fraction': None,
        'lr_decay_style': 'cosine',
        # The attention kernel comes from --attention_backend on this surface, NOT from the
        # transformers-surface attn_impl that MegatronSftArguments also inherits. Both are present on
        # the real object, so both are present here -- a stub with only attn_impl would hide the
        # rename entirely.
        'attention_backend': 'flash',
        'attn_impl': None,
        # GA derivation
        'global_batch_size': 16,
        'tensor_model_parallel_size': 1,
        'pipeline_model_parallel_size': 1,
        'context_parallel_size': 1,
        # reject checks
        'lr_warmup_iters': 0,
        'optimizer': 'adam',
        # read by name for the derived-vs-intent decision (always present on the real surface)
        'weight_decay_incr_style': 'constant',
        'start_weight_decay': None,
        'end_weight_decay': None,
        # name-hit sample
        'save_steps': 500,
    }
    base.update(overrides)
    return SimpleNamespace(**base)


# === Coverage guard (the core deliverable) ========================================================


class TestCoverageGuard:

    def test_every_config_field_is_accounted_for(self):
        """No dev Config field may be unclassified against the Megatron arg surface.

        A field in `unaccounted` is one the CLI would leave at its dev default while the user
        believes their flag applied -- silent, and exactly what this guard exists to prevent.
        """
        report = audit_coverage(_legacy_arg_names())
        unaccounted = {cfg: b['unaccounted'] for cfg, b in report.items() if b['unaccounted']}
        assert not unaccounted, (f'Config fields not accounted for by the Megatron mapping: {unaccounted}. '
                                 'Add each to RENAMES / DERIVED / ABSENT in swift/dev/cli/megatron.py.')

    def test_guard_fails_when_a_field_is_unmapped(self):
        """The guard must actually fire. Simulate a newly added Config field by shrinking the arg
        surface: dropping `seed` from the surface makes TrainConfig.seed unmapped, and the audit
        has to report it rather than quietly bucket it."""
        arg_names = _legacy_arg_names()
        report = audit_coverage(arg_names - {'seed'})
        assert 'seed' in report['TrainConfig']['unaccounted']

    def test_report_partitions_fields_exactly_once(self):
        """The buckets must partition each Config's fields (no double counting, none lost)."""
        report = audit_coverage(_legacy_arg_names())
        for cls in _CONFIGS:
            buckets = report[cls.__name__]
            flat = [
                n for key in ('name_hit', 'renamed', 'derived', 'superseded', 'absent', 'unaccounted')
                for n in buckets[key]
            ]
            assert sorted(flat) == sorted(f.name for f in dataclasses.fields(cls)), cls.__name__
            assert len(flat) == len(set(flat)), f'{cls.__name__}: field counted twice'

    def test_measured_gap_matches_documented_gap(self):
        """Pins the field-surface facts the design note records: the gap is 53/59 TrainConfig, 12
        DistributedConfig, 13 CheckpointConfig, 3 ModelConfig, 1 DatasetConfig, and 0 for Template. If
        an upstream rename shifts these, this fails and the note gets revisited instead of the numbers
        quietly rotting.

        ModelConfig went 0 -> 1 when attn_impl became a rename. It is a genuine gap, not bookkeeping:
        legacy describes the attention kernel with a DIFFERENT flag per backend (transformers
        --attn_impl vs Megatron --attention_backend) over disjoint value domains, and its Megatron
        path ignores attn_impl entirely. The same-name copy that made this look like a zero-gap field
        was reading the flag legacy does not use.

        1 -> 3 with MTP joint training, and both new entries are gaps in the same sense -- the two
        surfaces disagree about what "MTP is configured" means rather than about a name.
        ``enable_mtp_training`` is DERIVED because legacy needs no such flag: its trainer passes
        ``labels`` into the model, so --mtp_num_layers alone already trains the MTP heads, while dev
        computes its loss outside the model and must be told. ``mtp_freeze`` is ABSENT because the
        legacy SFT surface trains the heads whenever they exist and has no way to ask for the opposite.

        The TrainConfig/Distributed/Checkpoint/Dataset gaps also count the transformers
        TrainingArguments knobs (precision-eval, torch.compile, DDP/FSDP tuning, hub push,
        dataloader_drop_last) that have no Megatron surface -- see ABSENT in swift/dev/cli/megatron.py.
        """
        report = audit_coverage(_legacy_arg_names())
        gap = {
            cfg: len(b['renamed']) + len(b['derived']) + len(b['superseded']) + len(b['absent'])
            for cfg, b in report.items()
        }
        assert gap['TrainConfig'] == 53
        assert gap['DistributedConfig'] == 12
        assert gap['CheckpointConfig'] == 13
        # Every TemplateConfig field maps by name onto the Megatron surface.
        assert gap['TemplateConfig'] == 0
        assert gap['ModelConfig'] == 3
        assert report['ModelConfig']['renamed'] == ['attn_impl']
        assert report['ModelConfig']['derived'] == ['enable_mtp_training']
        assert report['ModelConfig']['absent'] == ['mtp_freeze']
        # dataloader_drop_last is the one DatasetConfig field with no Megatron dataloader flag.
        assert gap['DatasetConfig'] == 1

    def test_tables_only_name_real_config_fields(self):
        """A stale entry (renamed/derived/absent naming a field that no longer exists) would mask a
        real gap, so the tables are checked against the Configs themselves."""
        fields_by_cfg = {c.__name__: {f.name for f in dataclasses.fields(c)} for c in _CONFIGS}
        for table, label in ((RENAMES, 'RENAMES'), (DERIVED, 'DERIVED'), (SUPERSEDED, 'SUPERSEDED'), (ABSENT,
                                                                                                      'ABSENT')):
            for cfg_name, entries in table.items():
                assert cfg_name in fields_by_cfg, f'{label}: unknown Config {cfg_name}'
                unknown = set(entries) - fields_by_cfg[cfg_name]
                assert not unknown, f'{label}[{cfg_name}] names non-existent fields: {unknown}'


# === Renames and derivations =====================================================================


def test_lr_maps_to_learning_rate():
    """The break-change #1 trap: without this rename LoRA silently trains at TrainConfig's 1e-5."""
    _, _, _, train, *_ = megatron_args_to_configs(_args(lr=1e-4), world_size=1)
    assert train.learning_rate == 1e-4


def test_lr_unset_keeps_config_default():
    _, _, _, train, *_ = megatron_args_to_configs(_args(lr=None), world_size=1)
    assert train.learning_rate == TrainConfig().learning_rate


def test_core_renames():
    _, _, _, train, *_ = megatron_args_to_configs(
        _args(train_iters=42, micro_batch_size=2, adam_eps=1e-6, lr_warmup_fraction=0.03, global_batch_size=2),
        world_size=1)
    assert train.max_steps == 42
    assert train.per_device_train_batch_size == 2
    assert train.adam_epsilon == 1e-6
    assert train.warmup_ratio == 0.03


def test_train_iters_none_normalizes_to_minus_one():
    """`train_iters=None` means unset on the Megatron surface; max_steps spells that as -1, so a
    plain copy would leave None in an int field and break downstream `max_steps > 0` checks."""
    _, _, _, train, *_ = megatron_args_to_configs(_args(train_iters=None), world_size=1)
    assert train.max_steps == -1


class TestWeightDecayRampIsNotForwardedWhenDerived:
    """legacy post-init FILLS start/end_weight_decay from weight_decay when the style is 'constant'
    (megatron_args.py:1097-1099), where they are a no-op. Forwarding those derived values gives dev a
    start/end pair with a constant style, which validate_configs rejects -- so a plain
    `--weight_decay 0.1` run would die for a setting the user never made. Found by the slow CLI
    对拍, pinned here."""

    def test_constant_style_drops_derived_ends(self):
        _, _, _, train, *_ = megatron_args_to_configs(
            _args(weight_decay=0.1, weight_decay_incr_style='constant', start_weight_decay=0.1, end_weight_decay=0.1),
            world_size=1)
        assert train.start_weight_decay is None
        assert train.end_weight_decay is None
        assert train.weight_decay == 0.1

    def test_real_ramp_is_forwarded(self):
        """With a non-constant style the pair IS user intent and must survive."""
        _, _, _, train, *_ = megatron_args_to_configs(
            _args(weight_decay=0.1, weight_decay_incr_style='linear', start_weight_decay=0.0, end_weight_decay=0.2),
            world_size=1)
        assert train.weight_decay_incr_style == 'linear'
        assert train.start_weight_decay == 0.0
        assert train.end_weight_decay == 0.2

    def test_constant_style_passes_validate_configs(self):
        """End-to-end: the mapped Configs must survive the cross-config validator, which is where
        the original failure surfaced (it runs first thing in run_sft)."""
        from swift.dev.config import validate_configs

        configs = megatron_args_to_configs(
            _args(
                weight_decay=0.1,
                weight_decay_incr_style='constant',
                start_weight_decay=0.1,
                end_weight_decay=0.1,
                lr_decay_style='constant'),
            world_size=1)
        model, template, dataset, train, dist, checkpoint, tuner, megatron, moe = configs
        validate_configs(
            model, template, dataset, train, dist, checkpoint, tuner,
            megatron_config=megatron, moe_config=moe)


def test_name_hit_fields_still_copy():
    """The renames must not disturb the fields that already share a name on both surfaces."""
    configs = megatron_args_to_configs(
        _args(min_lr=1e-7, weight_decay=0.2, seed=7, save_steps=123), world_size=1)
    train, checkpoint = configs[3], configs[5]
    assert train.min_lr == 1e-7
    assert train.weight_decay == 0.2
    assert train.seed == 7
    assert checkpoint.save_steps == 123


def test_clip_grad_lands_on_max_grad_norm_and_clears_the_alias():
    """`--clip_grad` is the one Megatron arg that both name-hits and is renamed.

    It used to copy straight into TrainConfig.clip_grad. Now the two dev fields are one knob
    (max_grad_norm, with clip_grad as a deprecated alias), so the arg must land on max_grad_norm --
    and the alias must be cleared, or resolve_max_grad_norm would warn about a "both set" conflict
    on every run that passes --clip_grad.
    """
    from swift.dev.optimizer import resolve_max_grad_norm

    train = megatron_args_to_configs(_args(clip_grad=0.5), world_size=1)[3]
    assert train.max_grad_norm == 0.5
    assert train.clip_grad is None
    assert resolve_max_grad_norm(train) == 0.5


class TestGradientAccumulationDerivation:

    def test_ga_from_global_batch_size(self):
        """GA = GBS / (MBS * dp); dp = world / (tp*pp*cp)."""
        args = _args(global_batch_size=16, micro_batch_size=2)
        assert _derive_gradient_accumulation_steps(args, world_size=2) == 4  # 16/(2*2)

    def test_ga_accounts_for_model_parallel(self):
        """tp/pp/cp shrink dp, so the same GBS needs more accumulation."""
        args = _args(global_batch_size=8, micro_batch_size=1, tensor_model_parallel_size=2)
        assert _derive_gradient_accumulation_steps(args, world_size=4) == 4  # dp=2 -> 8/(1*2)

    def test_non_integral_ga_fails_fast(self):
        """Flooring would train on a different global batch than the flag asked for."""
        args = _args(global_batch_size=5, micro_batch_size=2)
        with pytest.raises(ValueError, match='not divisible'):
            _derive_gradient_accumulation_steps(args, world_size=1)

    def test_world_not_divisible_by_model_parallel_fails_fast(self):
        args = _args(global_batch_size=8, tensor_model_parallel_size=3)
        with pytest.raises(ValueError, match='not divisible'):
            _derive_gradient_accumulation_steps(args, world_size=2)

    def test_ga_reaches_train_config(self):
        _, _, _, train, *_ = megatron_args_to_configs(_args(global_batch_size=8, micro_batch_size=2), world_size=1)
        assert train.gradient_accumulation_steps == 4


class TestAttentionBackendRename:
    """The attention kernel must come from --attention_backend, not from the inherited attn_impl.

    MegatronSftArguments carries BOTH fields: attn_impl (from ModelArguments, the transformers
    surface) and attention_backend (from MegatronArguments). legacy's Megatron training path reads
    only the latter -- the sole attn_impl hit under swift/megatron/ is the unrelated vit_attn_impl.
    A same-name copy therefore filled dev's field from a value legacy IGNORES while dropping the flag
    the user actually set.

    The value domains differ, which is why this is a rename and not a lucky name match:
      transformers: sdpa / eager / flex_attention / flash_attn / flash_attention_2|_3|_4
      megatron:     flash / fused / unfused / local / auto (+ flash_2|_3|_4 pinning)
    (resolve_megatron_attn_backend translates the transformers names for callers that build a Config
    directly; on THIS surface --attention_backend wins because it is the flag legacy reads.)
    """

    def test_attention_backend_reaches_model_config(self):
        model, *_ = megatron_args_to_configs(_args(attention_backend='fused'), world_size=1)
        assert model.attn_impl == 'fused'

    def test_default_flash_is_carried_through(self):
        """legacy defaults --attention_backend to 'flash'; that must survive the mapping.

        Otherwise dev falls back to mcore's AttnBackend.auto, which TE resolves to the FUSED cuDNN
        kernel for this model shape -- a different kernel than legacy runs.
        """
        model, *_ = megatron_args_to_configs(_args(), world_size=1)
        assert model.attn_impl == 'flash'

    def test_transformers_side_attn_impl_does_not_leak_in(self):
        """A --attn_impl the user passed for the HF surface must NOT become the Megatron kernel.

        legacy ignores attn_impl on this path and reads --attention_backend, so honouring it would
        change which kernel runs relative to legacy. Note the resolver DOES understand transformers
        names (sdpa -> unfused etc.) -- that is for configs built directly in Python; here, where both
        flags exist side by side, the Megatron one is authoritative because that is the one legacy
        obeys.
        """
        model, *_ = megatron_args_to_configs(
            _args(attn_impl='flash_attention_4', attention_backend='fused'), world_size=1)
        assert model.attn_impl == 'fused', \
            'the transformers-surface attn_impl overrode the Megatron attention_backend'

    def test_enum_valued_attention_backend_is_normalised(self):
        """legacy's post-init converts the flag to an AttnBackend enum; dev's Config stores the name.

        Which of the two arrives depends on how far post-init has run, so both must map to the same
        string -- the enum is re-derived at build time by resolve_megatron_attn_backend.
        """
        from megatron.core.transformer.enums import AttnBackend

        model, *_ = megatron_args_to_configs(_args(attention_backend=AttnBackend.unfused), world_size=1)
        assert model.attn_impl == 'unfused'

    def test_mapped_value_is_accepted_by_the_resolver(self):
        """End-to-end: whatever the mapping produces must be resolvable, not just string-equal.

        Guards the seam between the two halves -- a mapping that emitted, say, 'AttnBackend.flash'
        would pass the assertions above if they only compared strings loosely, then fail at build.
        """
        from megatron.core.transformer.enums import AttnBackend

        from swift.dev.naming import resolve_megatron_attn_backend

        for flag, expected in (('flash', AttnBackend.flash), ('fused', AttnBackend.fused),
                               ('unfused', AttnBackend.unfused), ('local', AttnBackend.local), ('auto',
                                                                                                AttnBackend.auto)):
            model, *_ = megatron_args_to_configs(_args(attention_backend=flag), world_size=1)
            assert resolve_megatron_attn_backend(model.attn_impl) is expected, flag


class TestDecayStyleReverse:
    """The forward map is not injective, so the reverse needs a stated rule: prefer the swift name
    identical to the Megatron style; refuse when none matches rather than guessing."""

    @pytest.mark.parametrize('style', ['cosine', 'linear', 'constant'])
    def test_identical_name_wins(self, style):
        assert _decay_style_to_swift(style) == style

    def test_ambiguous_without_identical_name_is_refused(self):
        """'inverse-square-root' only reverses to 'inverse_sqrt' (no identical swift name), so the
        rule refuses -- picking one silently could select a schedule the user did not ask for."""
        with pytest.raises(NotImplementedError, match='ambiguous'):
            _decay_style_to_swift('inverse-square-root')

    def test_unknown_style_is_refused(self):
        with pytest.raises(NotImplementedError, match='lr_decay_style'):
            _decay_style_to_swift('WSD')

    def test_reaches_train_config(self):
        _, _, _, train, *_ = megatron_args_to_configs(_args(lr_decay_style='linear'), world_size=1)
        assert train.lr_scheduler_type == 'linear'


# === dev-only knobs and rejections ===============================================================


def test_dev_only_launch_knobs_are_set_by_the_entry_point():
    """backend/mode/nproc_per_node do not exist on the Megatron surface: copying would leave
    backend=None and the run would build a TransformersModel instead."""
    dist = megatron_args_to_configs(_args(), world_size=4)[4]
    assert dist.backend == 'megatron'
    assert dist.mode == 'local'
    assert dist.nproc_per_node == 4


def test_parallel_sizes_pass_through():
    dist = megatron_args_to_configs(
        _args(
            tensor_model_parallel_size=2,
            pipeline_model_parallel_size=1,
            context_parallel_size=1,
            global_batch_size=2,
            micro_batch_size=1),
        world_size=2)[4]
    assert dist.tensor_model_parallel_size == 2


def test_lr_warmup_iters_maps_to_warmup_steps():
    """An absolute warmup count is carried, not refused: legacy supports BOTH knobs.

    megatron_lm_utils uses lr_warmup_fraction when set and falls back to lr_warmup_iters otherwise,
    so dropping the latter lost a schedule legacy can express.
    """
    _, _, _, train, *_ = megatron_args_to_configs(_args(lr_warmup_iters=100), world_size=1)
    assert train.warmup_steps == 100
    assert train.warmup_ratio == 0.0


def test_lr_warmup_fraction_wins_over_iters_on_megatron():
    """legacy priority: the fraction wins when set. Both fields still reach the Config; the priority
    is applied by optimizer.warmup_budget, so the Config stays a faithful record of the flags."""
    from swift.dev.optimizer import warmup_budget

    _, _, _, train, *_ = megatron_args_to_configs(_args(lr_warmup_iters=100, lr_warmup_fraction=0.1), world_size=1)
    assert train.warmup_steps == 100 and train.warmup_ratio == 0.1
    assert warmup_budget(train, 50, is_megatron=True) == 0.1 * 50


def test_warmup_priority_is_inverted_on_the_transformers_backend():
    """The two backends resolve the pair with OPPOSITE priority; both are reproduced verbatim.

    transformers' TrainingArguments.get_warmup_steps takes warmup_steps when it is >= 1, so the
    ABSOLUTE count wins there -- unifying the rule would silently change one backend's warmup.
    """
    from swift.dev.config import TrainConfig
    from swift.dev.optimizer import warmup_budget

    cfg = TrainConfig(warmup_steps=100, warmup_ratio=0.1)
    assert warmup_budget(cfg, 50, is_megatron=False) == 100.0
    assert warmup_budget(cfg, 50, is_megatron=True) == 0.1 * 50


def test_non_adam_optimizer_reaches_train_config():
    _, _, _, train, *_ = megatron_args_to_configs(_args(optimizer='muon'), world_size=1)
    assert train.optimizer == 'muon'


def test_hf_surface_args_are_refused():
    """Symmetry with cli/sft.py's Megatron sentinel: HF args must not fall through this mapping."""
    hf_like = SimpleNamespace(model='m', dataset=['d'], tuner_type='full', torch_dtype=None)
    with pytest.raises(ValueError, match='train_iters'):
        megatron_args_to_configs(hf_like, world_size=1)


def test_tuner_dispatch():
    tuner = megatron_args_to_configs(_args(tuner_type='full'), world_size=1)[6]
    assert tuner is None
    tuner = megatron_args_to_configs(
        _args(tuner_type='lora', lora_rank=16, target_modules=['q_proj']), world_size=1)[6]
    assert isinstance(tuner, TunerConfig)
    assert tuner.tuner_type == 'lora' and tuner.lora_rank == 16
    with pytest.raises(NotImplementedError, match='tuner_type'):
        megatron_args_to_configs(_args(tuner_type='vera'), world_size=1)


def test_torch_dtype_object_is_normalized():
    import torch
    model, *_ = megatron_args_to_configs(_args(torch_dtype=torch.bfloat16), world_size=1)
    assert model.torch_dtype == 'bfloat16'


def test_returns_megatron_runtime_config_types():
    model, template, dataset, train, dist, checkpoint, _, megatron, moe = megatron_args_to_configs(
        _args(), world_size=1)
    assert isinstance(model, ModelConfig)
    assert isinstance(template, TemplateConfig)
    assert isinstance(dataset, DatasetConfig)
    assert isinstance(train, TrainConfig)
    assert isinstance(dist, DistributedConfig)
    assert isinstance(checkpoint, CheckpointConfig)
    assert isinstance(megatron, MegatronConfig)
    assert isinstance(moe, MoEConfig)


def test_export_maps_parallel_megatron_and_moe_configs(monkeypatch, tmp_path):
    import torch
    from swift.megatron.arguments import MegatronArguments

    from swift.dev.recipe.convert import _build_megatron_args

    monkeypatch.setattr(MegatronArguments, '__post_init__', lambda self: None)
    processor = SimpleNamespace(model_info=SimpleNamespace(is_moe_model=True, torch_dtype=torch.float16))
    args = _build_megatron_args(
        ModelConfig(model='m', torch_dtype='bfloat16'),
        ConvertConfig(to_mcore=True),
        DistributedConfig(
            tensor_model_parallel_size=2,
            pipeline_model_parallel_size=3,
            context_parallel_size=4,
            expert_model_parallel_size=5,
            sequence_parallel=True,
            bridge_backend='megatron-bridge'),
        processor=processor,
        output_dir=str(tmp_path),
        megatron_config=MegatronConfig(attention_softmax_in_fp32=False),
        moe_config=MoEConfig(moe_router_dtype='fp64'))

    assert args.tensor_model_parallel_size == 2
    assert args.pipeline_model_parallel_size == 3
    assert args.context_parallel_size == 4
    assert args.expert_model_parallel_size == 5
    assert args.sequence_parallel is True
    assert args.bridge_backend == 'megatron-bridge'
    assert args.attention_softmax_in_fp32 is False
    assert args.moe_router_dtype == 'fp64'
    assert args.moe_grouped_gemm is True
    assert args.torch_dtype is torch.bfloat16
