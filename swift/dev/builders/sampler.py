"""Builders for twinkle's Sampler: config -> sampler, GenerationConfig -> SamplingParams.

Peer of ``build_model`` / ``build_template``, and the single place dev maps its Configs onto
twinkle's sampling surface. ``run_infer`` / ``run_deploy`` / ``run_sampling`` all come through here,
so a backend quirk is fixed once.

Backends are vLLM, SGLang and transformers. The first two are the throughput engines; transformers is
the reach engine -- it loads whatever ``AutoModelForCausalLM`` loads, needs no extra install, and is
the only one that degrades per input instead of failing a batch. Anything needing a HF *forward* rather
than generation (embedding/seq_cls/reward scoring) still goes through the model side (``build_model`` +
``task=``), not through here.

The template contract is twinkle's: ``sample()`` calls ``encode`` / ``decode`` /
``get_vllm_input_ids`` / ``concat_input_feature`` on whatever ``set_template`` stored, and
twinkle's ``construct_class`` passes a ``twinkle.template.Template`` instance straight through. So
the template dev builds is handed over as an instance rather than re-resolved by name.
"""
from __future__ import annotations

import json
import logging
import os
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional

if TYPE_CHECKING:
    from swift.dev.config import GenerationConfig, ModelConfig

logger = logging.getLogger(__name__)

SamplerBackend = Literal['vllm', 'sglang', 'transformers']

#: ModelConfig knobs the engines accept, under the name each one uses for them. Everything else
#: an engine takes is passed verbatim through ``engine_args`` -- dev does not mirror engine flags.
_MODEL_KNOB_NAMES = {
    'vllm': {
        'torch_dtype': 'dtype',
        'max_model_len': 'max_model_len'
    },
    'sglang': {
        'torch_dtype': 'dtype',
        'max_model_len': 'context_length'
    },
    'transformers': {
        'torch_dtype': 'dtype',
        'max_model_len': 'max_model_len',
        'attn_impl': 'attn_implementation',
        'device_map': 'device_map'
    },
}

_SAMPLER_CLASSES = {'vllm': 'vLLMSampler', 'sglang': 'SGLangSampler', 'transformers': 'TransformersSampler'}


def build_sampler(
    model_config: ModelConfig,
    *,
    backend: SamplerBackend = 'vllm',
    engine_args: Optional[Dict[str, Any]] = None,
    device_mesh: Any = None,
    template: Any = None,
    adapters: Optional[List[str]] = None,
    remote_group: Optional[str] = None,
) -> Any:
    """ModelConfig -> a twinkle Sampler, with the template already set.

    Args:
        model_config: supplies ``model`` (the model id/path) plus the knobs in
            ``_MODEL_KNOB_NAMES``. ``task_type`` must be causal_lm or unset: the other task types
            need a pooling forward that no sampler has.
        backend: 'vllm', 'sglang' or 'transformers'.
        engine_args: passed verbatim to the engine, and wins over the ModelConfig knobs so a caller
            can always reach an engine flag dev does not model.
        device_mesh: twinkle DeviceMesh for data parallelism. ``sample`` is declared
            ``dispatch='slice_dp'``, so under Ray the inputs are sliced across DP ranks for free;
            leave it None for a single in-process engine.
        template: a twinkle Template instance, set on the sampler so Trajectory inputs (messages)
            can be encoded. Without it only pre-encoded InputFeature inputs work.
        adapters: LoRA paths this sampler will be asked to serve. Needed at CONSTRUCTION time, not
            just at ``sample()`` time: vLLM refuses ``LoRARequest`` unless the engine was created with
            ``enable_lora=True``, and it sizes its adapter slots from ``max_loras``. Passing the list
            up front is what makes ``sample(adapter_path=...)`` work at all, and what lets several
            adapters be resident at once. Ignored for transformers, which loads adapters on demand.
        remote_group: name of the twinkle ``DeviceGroup`` to place the engine in, under
            ``mode='ray'``. This is what keeps the sampler on its own GPUs instead of sharing the
            trainer's -- ``remote_class`` reads it off the constructor kwargs. Leave None in local
            mode, where there is only one process and nothing to place.

    Returns:
        The sampler. Callers own its lifetime: ``shutdown()`` is registered with atexit by the
        sampler itself, but a long-lived process should call it explicitly to free the GPU.
    """
    if model_config.model is None:
        raise ValueError('ModelConfig.model is required to build a sampler (it is the model id/path).')
    task_type = model_config.task_type or 'causal_lm'
    if task_type != 'causal_lm':
        raise ValueError(f'build_sampler supports task_type="causal_lm" only, got {task_type!r}. '
                         'Pooling tasks (seq_cls/embedding/reranker) have no sampler: they need a HF '
                         'forward, so build them with build_model(..) and pass task= instead.')
    if backend not in _SAMPLER_CLASSES:
        raise ValueError(f'Unknown sampler backend {backend!r}; expected one of {sorted(_SAMPLER_CLASSES)}. '
                         '(lmdeploy is deliberately not supported.)')

    kwargs = dict(engine_args or {})
    for cfg_name, engine_name in _MODEL_KNOB_NAMES[backend].items():
        value = getattr(model_config, cfg_name, None)
        # setdefault, not assignment: an explicit engine_args entry is the caller's override.
        if value is not None:
            kwargs.setdefault(engine_name, value)
    if adapters:
        _enable_lora(kwargs, backend, adapters)

    import twinkle.sampler as twinkle_sampler
    sampler_cls = getattr(twinkle_sampler, _SAMPLER_CLASSES[backend])

    logger.info(f'Building {backend} sampler for {model_config.model} with engine_args={kwargs}')
    extra: Dict[str, Any] = {'remote_group': remote_group} if remote_group else {}
    sampler = sampler_cls(model_config.model, engine_args=kwargs, device_mesh=device_mesh, **extra)
    if template is not None:
        sampler.set_template(template)
    return sampler


def _enable_lora(kwargs: Dict[str, Any], backend: str, adapters: List[str]) -> None:
    """Turn on the engine's LoRA machinery, sized for ``adapters``.

    vLLM allocates ``max_loras`` adapter slots and rejects ranks above ``max_lora_rank`` at request
    time, so both have to be right before the first request rather than discovered from it. The rank
    is read from each adapter's own ``adapter_config.json`` -- guessing low fails the request, and
    guessing high wastes memory on every slot.
    """
    if backend == 'transformers':
        return  # peft loads adapters into the live module; nothing to reserve.
    if backend == 'vllm':
        kwargs.setdefault('enable_lora', True)
        kwargs.setdefault('max_loras', len(adapters))
        kwargs.setdefault('max_lora_rank', _max_adapter_rank(adapters))
    elif backend == 'sglang':
        kwargs.setdefault('enable_lora', True)
        kwargs.setdefault('max_loras_per_batch', len(adapters))


def _max_adapter_rank(adapters: List[str], default: int = 16) -> int:
    """Largest ``r`` across the adapters, from their configs; ``default`` when none can be read."""
    ranks = []
    for adapter in adapters:
        config_path = os.path.join(adapter, 'adapter_config.json')
        if not os.path.isfile(config_path):
            # Hub ids are only resolved later by the sampler, so the rank is not knowable here.
            logger.warning(f'No adapter_config.json under {adapter}; falling back to max_lora_rank={default}.')
            continue
        with open(config_path, encoding='utf-8') as f:
            ranks.append(int(json.load(f).get('r', default)))
    return max(ranks) if ranks else default


def to_sampling_params(generation_config: Optional[GenerationConfig] = None, **overrides) -> Any:
    """GenerationConfig -> twinkle SamplingParams.

    Only the fields the config actually sets are carried over, so twinkle's own defaults stand for
    the rest (temperature 1.0, top_p 1.0, top_k -1, repetition_penalty 1.0) rather than being
    overwritten with None.

    Name differences worth stating, since a silent mismatch is a silently different distribution:
    ``max_new_tokens`` -> ``max_tokens``, ``stop_words`` -> ``stop``, and the ``logprobs`` bool +
    ``top_logprobs`` int pair collapses into twinkle's single int, whose meaning is vLLM's: 0 means
    the sampled token's logprob only, k means the top k alongside it.

    ``overrides`` win over the config; ``num_samples`` (the n-best-of width) has no GenerationConfig
    field and is expected to arrive that way.
    """
    from twinkle.data_format import SamplingParams

    params: Dict[str, Any] = {}
    if generation_config is not None:
        _reject_unsupported(generation_config)
        if generation_config.max_new_tokens is not None:
            params['max_tokens'] = generation_config.max_new_tokens
        for name in ('temperature', 'top_k', 'top_p', 'repetition_penalty'):
            value = getattr(generation_config, name)
            if value is not None:
                params[name] = value
        if generation_config.stop_words:
            params['stop'] = list(generation_config.stop_words)
        if generation_config.logprobs:
            params['logprobs'] = generation_config.top_logprobs or 0
    params.update(overrides)
    return SamplingParams(**params)


def _reject_unsupported(generation_config: GenerationConfig) -> None:
    """Fail loudly on GenerationConfig knobs twinkle's SamplingParams has no field for.

    Dropping them silently would return a plausible-looking sample generated under different rules
    than the caller asked for, which is worse than not running.
    """
    if generation_config.num_beams and generation_config.num_beams > 1:
        raise ValueError(f'num_beams={generation_config.num_beams} is not supported: twinkle SamplingParams '
                         'has no beam-search field. Use num_samples for parallel sampling instead.')
    if generation_config.structured_outputs_regex:
        raise ValueError('structured_outputs_regex is not supported: twinkle SamplingParams has no '
                         'guided-decoding field. Pass the engine its own guided-decoding args via '
                         'engine_args if the backend supports them.')


def sampled_texts(responses: List[Any]) -> List[List[str]]:
    """SampleResponse list -> the decoded text of each sequence, grouped per input.

    Both the infer and sampling recipes need exactly this projection (one inner list per input,
    ``num_samples`` entries long), so it lives here rather than being written twice.
    """
    return [[seq.decoded for seq in response.sequences] for response in responses]
