"""Unified name resolution: swift-style config names -> twinkle constructibles.

Single entry point ``resolve(category, swift_name)`` so the scattered name mappings
(loss / optimizer / scheduler / megatron_decay_style / strategy) live in ONE place, and
the twinkle string-parsing gotcha is fixed at the boundary instead of surfacing as a
runtime ``ValueError`` deep inside ``set_loss('cross_entropy')``.

Why this layer exists (twinkle construct_class gotcha):
  twinkle's ``construct_class(name, base, module)`` resolves a *string* via
  ``getattr(module, name)`` and, on miss, ``Plugin.load_plugin`` (which only
  accepts ``hf://`` / ``ms://`` ids, else raises). So a plain swift-style name
  like ``'cross_entropy'`` / ``'cosine'`` / ``'adamw_torch_fused'`` does NOT
  resolve: the module holds ``CrossEntropyLoss`` (a class), not an attribute
  named ``cross_entropy``. twinkle even defines ``torch_loss_mapping`` but
  ``construct_class`` never consults it. This layer bridges that gap.

Design (per-category target differs on purpose — do NOT pretend they were all broken):
  - loss:     swift name -> twinkle *type* (reuse twinkle ``torch_loss_mapping``,
              never fork twinkle's loss roster). configure_loss instantiates it.
  - optim:    swift name -> twinkle-resolvable *name* string ('AdamW' etc.);
              twinkle set_optimizer does ``getattr(torch.optim, name)`` fine.
  - sched:    swift name -> twinkle-resolvable *name* string
              ('CosineWarmupScheduler' etc.); resolves against
              ``[torch.optim.lr_scheduler, twinkle.module.scheduler]``.
              None means "constant lr / no scheduler".
  - megatron_decay_style: swift name -> Megatron ``lr_decay_style`` string. Megatron builds
              warmup and decay in ONE scheduler, so there is no None ("no scheduler") case.
  - strategy: swift name -> twinkle strategy string ('accelerate' etc.).
"""
from __future__ import annotations

from typing import Any, Optional, Type

# --- optimizer: swift optim name -> torch.optim class name (twinkle-resolvable) ---
# Value is either a name string resolved against torch.optim, or a (type, extra_kwargs) pair when
# the swift/HF name does NOT map to a plain torch.optim class with default kwargs.
_OPTIM_NAME_MAP = {
    'adamw_torch': 'AdamW',
    'adamw_torch_fused': 'AdamW',
    'adamw': 'AdamW',
    'adam': 'Adam',
    'sgd': 'SGD',
    'adafactor': 'Adafactor',
}

# Optim names needing extra constructor kwargs to match transformers' Trainer exactly.
# adamw_torch_fused: HF shares one handler with adamw_torch and only adds fused=True
# (trainer_optimizer.py _get_adamw_torch), so without this the two names are indistinguishable
# and the fused CUDA kernel is silently never used despite being the dev default.
_OPTIM_EXTRA_KWARGS = {
    'adamw_torch_fused': {
        'fused': True
    },
    # HF does NOT use torch.optim.Adafactor: it uses its own transformers.optimization.Adafactor
    # and forces these two off (trainer_optimizer.py _get_adafactor). torch's Adafactor defaults
    # differ, so resolving 'adafactor' to torch.optim would train DIFFERENTLY from legacy swift
    # without any error -- the one silent training-effect divergence in this map.
    'adafactor': {
        'scale_parameter': False,
        'relative_step': False
    },
}

# --- scheduler: swift lr_scheduler_type -> what to hand twinkle's set_lr_scheduler.
#     None (explicit key) => constant lr, no scheduler set.
#     A str is resolved by twinkle from [torch.optim.lr_scheduler, twinkle.module.scheduler]; a CLASS
#     is constructed directly, which is how the HF-backed schedules get in (see dev/scheduler.py --
#     HF ships factory functions, and neither a function nor an instance can cross set_lr_scheduler).
#     Still narrower than HF's SchedulerType: reduce_lr_on_plateau needs a metric rather than a step
#     count, and warmup_stable_decay needs stable/decay step counts dev has no field for. ---
_SCHED_NAME_MAP = {
    'cosine': 'CosineWarmupScheduler',
    'linear': 'LinearWarmupScheduler',
    'constant': None,
    'cosine_with_min_lr': 'CosineWithMinLRScheduler',
    'constant_with_warmup': 'ConstantWithWarmupScheduler',
    'cosine_with_restarts': 'CosineWithRestartsScheduler',
    'polynomial': 'PolynomialDecayScheduler',
    'inverse_sqrt': 'InverseSqrtScheduler',
}

# --- megatron scheduler: swift lr_scheduler_type -> Megatron OptimizerParamScheduler
#     lr_decay_style. Values must stay inside Megatron's legal set -- the same Literal legacy swift
#     accepts in MegatronArguments.lr_decay_style ('constant', 'linear', 'cosine',
#     'inverse-square-root', 'WSD') -- which test_optimizer_config pins.
#
#     The key set is NOT required to match _SCHED_NAME_MAP: the two backends run different
#     schedulers, so each maps what it can actually run and rejects the rest BY NAME (the error
#     says which backend refused, and whether the other one accepts it). Forcing the two sets equal
#     would cap dev at their intersection forever.
#
#     'constant' maps to a real style rather than None because Megatron has no "no scheduler" mode;
#     warmup is applied before any decay-style branch (optimizer_param_scheduler.py get_lr), which
#     is also why 'constant_with_warmup' is expressible HERE but not on the Transformers path.
#
#     'WSD' (== HF's 'warmup_stable_decay') is deliberately NOT mapped yet: Megatron asserts
#     wsd_decay_steps is not None when that style is selected, and legacy exposes two dedicated
#     knobs for it (lr_wsd_decay_iters / lr_wsd_decay_style) that dev has no field for. Adding the
#     key without them would only move the failure into the scheduler constructor. ---
_MEGATRON_DECAY_STYLE_MAP = {
    'cosine': 'cosine',
    'linear': 'linear',
    'constant': 'constant',
    'constant_with_warmup': 'constant',
    'inverse_sqrt': 'inverse-square-root',
    # Megatron's scheduler decays toward min_lr natively, so the "with min lr" variant is the plain
    # cosine style plus TrainConfig.min_lr -- no extra schedule. validate_configs requires min_lr to
    # be set for this name, since the name promises a floor and 0.0 would silently be plain cosine.
    'cosine_with_min_lr': 'cosine',
}

# --- attention kernel: ModelConfig.attn_impl -> Megatron AttnBackend ---
#     attn_impl is the unified name for this concept (HF attn_implementation / Megatron
#     attention_backend), so the Megatron side reuses the field rather than adding one.
#     Megatron's own five values are below; transformers-style names are TRANSLATED (not rejected) by
#     the resolver, since each has a real Megatron counterpart.
MEGATRON_ATTN_BACKEND_NAMES = ('flash', 'fused', 'unfused', 'local', 'auto')

# transformers-style attn_impl -> the Megatron kernel that means the same thing. attn_impl is ONE
# shared field, so a config written for the transformers backend can reach the Megatron surface
# verbatim; translating is strictly better than refusing, since every one of these names does have a
# Megatron counterpart:
#   flash_attn                      -> flash   (no version in the name, so nothing is pinned: TE
#                                              picks whichever FA build is installed. 'flash_attn' is
#                                              HF's back-compat alias for _2, model_args.py:34-38,
#                                              but carries no version of its own)
#   sdpa                            -> unfused (TE calls unfused "the native PyTorch implementation",
#                                              dot_product_attention.py:848 -- i.e. torch SDPA)
#   eager                           -> local   (mcore's own non-TE pytorch attention)
# flash_attention_2/_3/_4 are deliberately NOT here: they name a SPECIFIC FA version, which is a pin
# dev does not implement (see resolve_megatron_attn_backend). Mapping them to plain 'flash' would
# silently drop the pin and let TE choose, while the equivalent Megatron spellings (flash_2/_3/_4)
# fail fast -- so the same intent got two different outcomes depending on which name the user typed.
# Both spellings now take the same fail-fast path.
# flex_attention is deliberately absent: neither TE nor mcore implements it (no hit for 'flex' in
# either enums.py or TE's dot_product_attention), so there is nothing to map it to.
_HF_TO_MEGATRON_ATTN = {
    'flash_attn': 'flash',
    'sdpa': 'unfused',
    'eager': 'local',
}

# transformers-only kernels with no Megatron counterpart -> rejected rather than silently downgraded.
_HF_ONLY_ATTN_NAMES = ('flex_attention', )

# Both spellings of "pin FlashAttention to version N". Kept as prefixes rather than an explicit list
# so a future FA5 is refused too instead of falling through to the generic "unknown name" error.
_FLASH_VERSION_PIN_PREFIXES = ('flash_attention_', 'flash_')

# TE exposes one module-level availability flag per FlashAttention major version; pinning works by
# turning the others off (legacy does the same, megatron_args.py:905-915).
_FLASH_PIN_FLAGS = {2: 'is_installed', 3: 'v3_is_installed', 4: 'v4_is_installed'}
_FLASH_PIN_SUPPORTED_VERSIONS = frozenset(_FLASH_PIN_FLAGS)


def _is_version_pinned_flash(key: str) -> bool:
    """True for names that pin a FlashAttention VERSION (flash_3, flash_attention_4, ...).

    Excludes the unversioned aliases, which are a plain kernel choice: 'flash' (Megatron) and
    'flash_attn' (HF). Only a trailing integer counts as a pin.
    """
    for prefix in _FLASH_VERSION_PIN_PREFIXES:
        if key.startswith(prefix) and key[len(prefix):].isdigit():
            return True
    return False


def flash_version_pin(attn_impl: Optional[str]) -> Optional[int]:
    """The FlashAttention version ``attn_impl`` pins, or None when it pins nothing.

    Accepts both spellings (Megatron ``flash_3`` / transformers ``flash_attention_3``) so the same
    intent resolves identically. Unversioned names ('flash', 'flash_attn', 'sdpa', ...) -> None.
    """
    if attn_impl is None:
        return None
    key = str(attn_impl).lower()
    for prefix in _FLASH_VERSION_PIN_PREFIXES:
        if key.startswith(prefix) and key[len(prefix):].isdigit():
            return int(key[len(prefix):])
    return None


def apply_flash_version_pin(attn_impl: Optional[str]) -> Optional[int]:
    """Force transformer_engine to use exactly the FlashAttention version ``attn_impl`` pins.

    Reproduces legacy's MegatronArguments._init_attention_backend (megatron_args.py:898-918): TE has
    no per-call version selector, so the only lever is its module-level availability flags -- assert
    the requested build is installed, then switch the other two OFF so TE's dispatcher can only pick
    the requested one.

    Returns the pinned version, or None when nothing was pinned (then TE keeps its own choice).

    MUST run in the process that builds the model: these are transformer_engine module globals, so a
    driver-side call would not reach a Ray worker. Called from DevMegatronStrategy.__init__, which
    runs on the worker, rather than from build_model, which runs on the driver in Ray mode.
    """
    version = flash_version_pin(attn_impl)
    if version is None:
        return None
    from transformer_engine.pytorch.attention.dot_product_attention.utils import FlashAttentionUtils as fa_utils

    if version not in _FLASH_PIN_SUPPORTED_VERSIONS:
        raise NotImplementedError(
            f'attn_impl={attn_impl!r} pins FlashAttention v{version}, which transformer_engine has no '
            f'availability flag for. Supported: {sorted(_FLASH_PIN_SUPPORTED_VERSIONS)}.')
    # Only flags this TE build actually defines: v4_is_installed is absent on older versions, and a
    # bare getattr would surface as AttributeError instead of an actionable message.
    present = {v: flag for v, flag in _FLASH_PIN_FLAGS.items() if hasattr(fa_utils, flag)}
    if version not in present:
        raise ValueError(f'attn_impl={attn_impl!r} requests flash-attn v{version}, but the installed '
                         f'transformer_engine has no {_FLASH_PIN_FLAGS[version]!r} flag, so this version cannot be '
                         f'selected. Versions this TE build can pin: {sorted(present)}.')
    installed = {v: getattr(fa_utils, flag) for v, flag in present.items()}
    if not installed[version]:
        raise ValueError(f'attn_impl={attn_impl!r} requests flash-attn v{version}, which is not installed. '
                         f'Detected: ' + ', '.join(f'FA{v}={state}' for v, state in sorted(installed.items())))
    # Switch the other versions off so TE's dispatcher cannot fall back to one of them.
    for other, flag in present.items():
        if other != version:
            setattr(fa_utils, flag, False)
    return version


# --- strategy: swift/dev strategy name -> twinkle strategy string ---
_STRATEGY_NAME_MAP = {
    'accelerate': 'accelerate',
    'ddp': 'accelerate',
    'fsdp': 'native_fsdp',
    'native_fsdp': 'native_fsdp',
    'deepspeed': 'deepspeed',
}


def resolve_loss(swift_name: str) -> Type:
    """swift loss name -> twinkle Loss *type* (via twinkle torch_loss_mapping)."""
    from twinkle.loss import torch_loss_mapping

    key = swift_name.lower()
    cls = torch_loss_mapping.get(key)
    if cls is None:
        raise NotImplementedError(f'Unknown loss {swift_name!r}. Known: {sorted(torch_loss_mapping)}')
    return cls


def resolve_optim(swift_name: str) -> str:
    """swift optim name -> twinkle-resolvable torch.optim class name."""
    key = swift_name.lower()
    name = _OPTIM_NAME_MAP.get(key)
    if name is None:
        raise NotImplementedError(f'Unknown optimizer {swift_name!r}. Known: {sorted(_OPTIM_NAME_MAP)}')
    return name


def resolve_optim_target(swift_name: str) -> tuple:
    """swift optim name -> ``(target, extra_kwargs)`` matching transformers' Trainer construction.

    ``target`` is either a twinkle-resolvable class NAME (looked up on torch.optim) or a class TYPE
    when torch.optim has no equivalent -- twinkle's construct_class accepts both. ``extra_kwargs``
    carries the constructor arguments HF applies for that specific name (fused / Adafactor flags),
    which a name-only mapping silently drops.
    """
    key = swift_name.lower()
    name = _OPTIM_NAME_MAP.get(key)
    if name is None:
        raise NotImplementedError(f'Unknown optimizer {swift_name!r}. Known: {sorted(_OPTIM_NAME_MAP)}')
    extra = dict(_OPTIM_EXTRA_KWARGS.get(key, {}))
    if key == 'adafactor':
        # Import the HF implementation, not torch.optim.Adafactor (see _OPTIM_EXTRA_KWARGS).
        from transformers.optimization import Adafactor
        return Adafactor, extra
    return name, extra


def parse_optim_args(optim_args: Optional[str]) -> dict:
    """Parse a ``"k1=v1,k2=v2"`` optim_args string into kwargs, mirroring HF ``_parse_optim_args``.

    Values are coerced to bool/int/float when unambiguous (the CLI can only deliver strings, but
    optimizer constructors expect real types); anything else is passed through as a string.
    """
    if not optim_args:
        return {}
    parsed: dict = {}
    for mapping in optim_args.replace(' ', '').split(','):
        if '=' not in mapping:
            raise ValueError(f'Invalid optim_args entry {mapping!r}: expected "key=value" pairs separated by ",".')
        key, value = mapping.split('=', 1)
        parsed[key] = _coerce(value)
    return parsed


def parse_scheduler_kwargs(lr_scheduler_kwargs) -> dict:
    """TrainConfig.lr_scheduler_kwargs (dict, or a JSON string from the CLI) -> kwargs.

    Legacy swift accepts JSON here (``--lr_scheduler_kwargs '{"min_lr": 1e-6}'``), so a string must
    be parsed rather than forwarded, and a malformed one must say so instead of reaching the
    scheduler constructor as a str.
    """
    if not lr_scheduler_kwargs:
        return {}
    if isinstance(lr_scheduler_kwargs, dict):
        return dict(lr_scheduler_kwargs)
    import json
    try:
        parsed = json.loads(lr_scheduler_kwargs)
    except json.JSONDecodeError as e:
        raise ValueError(f'lr_scheduler_kwargs is not valid JSON: {lr_scheduler_kwargs!r} ({e})')
    if not isinstance(parsed, dict):
        raise ValueError(f'lr_scheduler_kwargs must be a JSON object, got {type(parsed).__name__}: '
                         f'{lr_scheduler_kwargs!r}')
    return parsed


def _coerce(value: str):
    lowered = value.lower()
    if lowered in ('true', 'false'):
        return lowered == 'true'
    if lowered in ('none', 'null'):
        return None
    for cast in (int, float):
        try:
            return cast(value)
        except ValueError:
            continue
    return value


def _unsupported_scheduler(swift_name: str, *, backend: str, supported: dict, other_backend: str,
                           other_supported: dict) -> NotImplementedError:
    """Build the rejection for an lr_scheduler_type the given backend cannot run.

    Names the backend that refused, because the two backends support different sets on purpose --
    otherwise "not supported" reads like a dev-wide limitation when the other backend would run it.
    """
    msg = (f'lr_scheduler_type {swift_name!r} is not supported on the {backend} backend. '
           f'Supported there: {sorted(k for k in supported)}.')
    if swift_name.lower() in other_supported:
        msg += (f' ({swift_name!r} IS supported on the {other_backend} backend -- the two run '
                f'different schedulers, so their sets differ by design.)')
    return NotImplementedError(msg)


def resolve_scheduler(swift_name: str):
    """swift lr_scheduler_type -> a name twinkle can resolve, a class, or None for constant lr.

    dev's own adapters (dev/scheduler.py) are returned as CLASSES: twinkle resolves a string only
    against [torch.optim.lr_scheduler, twinkle.module.scheduler], and dev's module is in neither, so
    a name would fall through to Plugin.load_plugin and raise.
    """
    key = swift_name.lower()
    if key not in _SCHED_NAME_MAP:
        raise _unsupported_scheduler(
            swift_name,
            backend='Transformers',
            supported=_SCHED_NAME_MAP,
            other_backend='Megatron',
            other_supported=_MEGATRON_DECAY_STYLE_MAP)
    name = _SCHED_NAME_MAP[key]
    if name is None:
        return None
    # Imported lazily so this module stays cheap for callers that only need name lookups.
    from swift.dev import scheduler as dev_scheduler
    return getattr(dev_scheduler, name, name)


def resolve_megatron_decay_style(swift_name: str) -> str:
    """swift lr_scheduler_type -> Megatron lr_decay_style.

    Fail-fast on unsupported names instead of falling back to 'cosine': a silent fallback would
    train a different schedule than requested -- the same class of bug as the old
    constant_with_warmup -> LinearWarmupScheduler mapping on the Transformers path.
    """
    key = swift_name.lower()
    if key not in _MEGATRON_DECAY_STYLE_MAP:
        raise _unsupported_scheduler(
            swift_name,
            backend='Megatron',
            supported=_MEGATRON_DECAY_STYLE_MAP,
            other_backend='Transformers',
            other_supported=_SCHED_NAME_MAP)
    return _MEGATRON_DECAY_STYLE_MAP[key]


def resolve_megatron_attn_backend(attn_impl: Optional[str]):
    """ModelConfig.attn_impl -> a Megatron AttnBackend ENUM member (never a string).

    Returns the ENUM deliberately. mcore stores this on TransformerConfig.attention_backend and tests
    it by identity -- e.g. ``self.attention_backend == AttnBackend.flash``
    (Megatron-LM transformer_config.py:2720). TransformerConfig is a plain dataclass with no type
    coercion, so handing it the string 'flash' type-checks fine and then compares unequal to every
    AttnBackend member: the run would silently take a different kernel than requested. That is the
    same silent-fallback class of bug as the warmup rounding and the min_lr override.

    Accepts BOTH naming conventions, because attn_impl is one shared field and a config written for
    the transformers backend can arrive here unchanged:
      - Megatron's own names pass through: flash / fused / unfused / local / auto.
      - transformers names are translated via _HF_TO_MEGATRON_ATTN (sdpa -> unfused, eager -> local,
        flash_attn|flash_attention_N -> flash). Each has a real Megatron counterpart, so translating
        beats refusing -- and refusing would have made `--attn_impl sdpa` unusable on this backend for
        no reason.
    Only names with NO counterpart are rejected (flex_attention), plus Megatron's flash_N version
    pinning, which legacy implements by mutating transformer_engine globals and dev does not
    reproduce.

    Default (attn_impl unset) is flash, matching legacy swift
    (MegatronArguments.attention_backend = 'flash', megatron_args.py:499) rather than mcore's own
    AttnBackend.auto (transformer_config.py:144). This is a deliberate divergence from the mcore
    default: under auto, TE picks per-shape, and for a Qwen2.5 bf16 causal THD forward it selects the
    FUSED cuDNN kernel, not flash -- so leaving auto in place means dev and legacy run different
    attention kernels on the same config. Measured with TE's own dispatcher
    (get_attention_backend): use_flash=False, use_fused=True (NVTE_F16_arbitrary_seqlen).
    """
    from megatron.core.transformer.enums import AttnBackend

    if attn_impl is None:
        return AttnBackend.flash
    key = str(attn_impl).lower()
    if key in _HF_ONLY_ATTN_NAMES:
        raise NotImplementedError(
            f'attn_impl={attn_impl!r} has no Megatron equivalent: neither transformer_engine nor '
            f'megatron-core implements it. Megatron kernels are '
            f'{list(MEGATRON_ATTN_BACKEND_NAMES)}; the transformers-style names that DO translate are '
            f'{sorted(_HF_TO_MEGATRON_ATTN)}.')
    if _is_version_pinned_flash(key):
        # A version pin selects the flash KERNEL; the version itself is enforced separately by
        # apply_flash_version_pin (TE module globals), exactly as legacy collapses its own flash_N to
        # 'flash' after mutating those globals (megatron_args.py:917). Both spellings land here so the
        # same intent behaves identically.
        if flash_version_pin(key) not in _FLASH_PIN_SUPPORTED_VERSIONS:
            raise NotImplementedError(
                f'attn_impl={attn_impl!r} pins a FlashAttention version transformer_engine has no '
                f'availability flag for. Supported pins: '
                f'{sorted("flash_%d" % v for v in _FLASH_PIN_SUPPORTED_VERSIONS)} (or the '
                f'transformers spelling flash_attention_N).')
        return AttnBackend.flash
    key = _HF_TO_MEGATRON_ATTN.get(key, key)
    if key not in MEGATRON_ATTN_BACKEND_NAMES:
        raise NotImplementedError(f'Unknown attn_impl={attn_impl!r} for the Megatron backend. '
                                  f'Supported: {list(MEGATRON_ATTN_BACKEND_NAMES)} '
                                  f'(transformers-style names also accepted: {sorted(_HF_TO_MEGATRON_ATTN)}).')
    return AttnBackend[key]


def resolve_strategy(swift_name: str) -> str:
    """swift/dev strategy name -> twinkle strategy string."""
    key = swift_name.lower()
    name = _STRATEGY_NAME_MAP.get(key)
    if name is None:
        raise NotImplementedError(f'Unknown strategy {swift_name!r}. Known: {sorted(_STRATEGY_NAME_MAP)}')
    return name


_RESOLVERS = {
    'loss': resolve_loss,
    'optim': resolve_optim,
    'scheduler': resolve_scheduler,
    'megatron_decay_style': resolve_megatron_decay_style,
    'strategy': resolve_strategy,
}


def resolve(category: str, swift_name: str) -> Any:
    """Unified entry: resolve a swift-style name for one of loss/optim/scheduler/strategy.

    Returns a twinkle *type* for 'loss', a twinkle-resolvable *name* string for
    'optim'/'scheduler'/'strategy' (scheduler may return None => constant lr).
    """
    fn = _RESOLVERS.get(category)
    if fn is None:
        raise ValueError(f'Unknown resolve category {category!r}. Known: {sorted(_RESOLVERS)}')
    return fn(swift_name)
