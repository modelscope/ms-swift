"""Builders for twinkle's Sampler: config -> sampler, GenerationConfig -> SamplingParams.

Peer of ``build_model`` / ``build_template``, and the single place dev maps its Configs onto
twinkle's sampling surface. ``run_infer`` / ``run_deploy`` all come through here,
so a backend quirk is fixed once.

Backends are vLLM, SGLang and transformers. The first two are the throughput engines; transformers is
the reach engine -- it loads whatever ``AutoModelForCausalLM`` loads, needs no extra install, and is
the only one that degrades per input instead of failing a batch. vLLM and SGLang also serve the
pooling tasks (embedding/seq_cls/reranker) through ``sampler.encode``; transformers has no pooling
head, so anything needing a HF *forward* on that backend still goes through the model side
(``build_model`` + ``task=``). A generative reranker is not pooling -- it is a decoder-only model
scored off generation, so it builds here as a generation sampler (see ``_GENERATIVE_TASK_TYPES``).

The template contract is twinkle's: ``sample()`` calls ``encode`` / ``decode`` /
``get_vllm_input_ids`` / ``concat_input_feature`` on whatever ``set_template`` stored, and
twinkle's ``construct_class`` passes a ``twinkle.template.Template`` instance straight through. So
the template dev builds is handed over as an instance rather than re-resolved by name.
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional

if TYPE_CHECKING:
    from swift.dev.config import GenerationConfig, InferConfig, ModelConfig, QuantizeConfig, RolloutConfig

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

#: dev ``task_type`` values that are a pooling forward rather than generation, and the twinkle/vLLM
#: pooling head each one runs. ``embedding``->``embed`` (a sentence vector), ``seq_cls``->``classify``
#: (per-class logits/probs), ``reranker``->``classify`` (a cross-encoder relevance score is what a
#: ``classify`` scoring model returns; vLLM has no separate ``score`` pooling task). These need a
#: backend with a pooling head, so only vLLM/SGLang serve them; transformers has none and still goes
#: through ``build_model`` + ``task=``. A *generative* reranker is deliberately absent -- see
#: ``_GENERATIVE_TASK_TYPES``.
_POOLING_TASK_MAP = {
    'embedding': 'embed',
    'seq_cls': 'classify',
    'reranker': 'classify',
}

#: dev ``task_type`` values served by the *generation* path rather than a pooling head. ``causal_lm`` is
#: plain generation; ``generative_reranker`` is a decoder-only reranker (e.g. Qwen3-Reranker) scored by
#: the yes/no logprob difference of its first generated token, so it needs a generation engine, not a
#: pooling runner -- neither vLLM nor sglang serves it through ``encode`` (sglang's docs are explicit:
#: launch it *without* ``--is-embedding``). ``run_infer`` builds the sampler here and does the yes/no
#: scoring itself.
_GENERATIVE_TASK_TYPES = ('causal_lm', 'generative_reranker')


def build_engine_args(backend: str, infer_config: 'InferConfig', rollout_config: 'RolloutConfig') -> Dict[str, Any]:
    """Map inference and rollout Config fields to sampler engine arguments."""
    if backend == 'pt':
        backend = 'transformers'
    if backend == 'transformers':
        return {'max_batch_size': infer_config.max_batch_size}
    prefix = f'{backend}_'
    result = {}
    excluded = {
        'engine_kwargs', 'mode', 'server_base_url', 'server_host', 'server_port', 'server_timeout',
        'server_group_port', 'server_pass_dataset'
    }
    for name, value in asdict(rollout_config).items():
        if name.startswith(prefix) and value is not None:
            key = name[len(prefix):]
            if key not in excluded:
                result[key] = value
    result.update(getattr(rollout_config, f'{prefix}engine_kwargs', None) or {})
    return result


def build_sampler(
    model_config: ModelConfig,
    *,
    backend: SamplerBackend = 'vllm',
    engine_args: Optional[Dict[str, Any]] = None,
    device_mesh: Any = None,
    template: Any = None,
    adapters: Optional[List[str]] = None,
    remote_group: Optional[str] = None,
    quantize_config: Optional[QuantizeConfig] = None,
) -> Any:
    """ModelConfig -> a twinkle Sampler, with the template already set.

    Args:
        model_config: supplies ``model`` (the model id/path) plus the knobs in
            ``_MODEL_KNOB_NAMES``. ``task_type`` may be a generation task in ``_GENERATIVE_TASK_TYPES``
            (``causal_lm``, or ``generative_reranker``, a decoder-only reranker scored off generation)
            or one of the pooling tasks in ``_POOLING_TASK_MAP`` (embedding/seq_cls/reranker); a pooling
            task builds the engine as a pooling model and is served by ``sampler.encode``. Pooling needs
            a backend with a pooling head, so it is vLLM/SGLang only -- transformers still goes through
            ``build_model`` + ``task=``.
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
    is_pooling = task_type in _POOLING_TASK_MAP
    if task_type not in _GENERATIVE_TASK_TYPES and not is_pooling:
        raise ValueError(f'build_sampler got task_type={task_type!r}; expected one of the generation tasks '
                         f'{list(_GENERATIVE_TASK_TYPES)} or the pooling tasks {sorted(_POOLING_TASK_MAP)}.')
    if is_pooling and backend == 'transformers':
        raise ValueError(f'backend="transformers" cannot serve the pooling task_type={task_type!r}: it has '
                         'no pooling head. Build it with build_model(..) and pass task= instead, or use '
                         'backend="vllm"/"sglang".')
    if backend not in _SAMPLER_CLASSES:
        raise ValueError(f'Unknown sampler backend {backend!r}; expected one of {sorted(_SAMPLER_CLASSES)}. '
                         '(lmdeploy is deliberately not supported.)')

    kwargs = dict(engine_args or {})
    quant_method = getattr(quantize_config, 'quant_method', None)
    if quant_method is not None:
        if backend != 'transformers':
            engine_option = 'vllm_quantization' if backend == 'vllm' else 'sglang_quantization'
            raise ValueError(
                f'quant_method={quant_method!r} is a Transformers load-time setting and cannot be translated to '
                f'backend={backend!r}. Use --{engine_option} (or backend-specific engine_args), or rely on the '
                'quantization metadata embedded in an AWQ/GPTQ checkpoint.')
        from swift.dev.builders.quantization import build_load_quantization_config
        quantization_config = build_load_quantization_config(
            quantize_config, torch_dtype=model_config.torch_dtype)
        if quantization_config is not None:
            kwargs.setdefault('quantization_config', quantization_config)

    for cfg_name, engine_name in _MODEL_KNOB_NAMES[backend].items():
        value = getattr(model_config, cfg_name, None)
        # setdefault, not assignment: an explicit engine_args entry is the caller's override.
        if value is not None:
            kwargs.setdefault(engine_name, value)
    if is_pooling:
        # Route the model to the backend's pooling forward. vLLM spells it ``runner='pooling'`` (which
        # swaps the LM head for the pooling head and serves requests through ``encode``); sglang spells
        # it ``is_embedding=True``. setdefault so an explicit engine_args entry still wins.
        if backend == 'vllm':
            kwargs.setdefault('runner', 'pooling')
        elif backend == 'sglang':
            kwargs.setdefault('is_embedding', True)
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


def pooling_task_for(task_type: Optional[str]) -> str:
    """dev ``task_type`` -> the twinkle/vLLM pooling head name, or ``None`` for a generation task.

    The single place the dev-vocabulary to pooling-vocabulary mapping is applied, so ``run_infer`` and
    ``run_deploy`` do not each carry their own copy of it.
    """
    return _POOLING_TASK_MAP.get(task_type or 'causal_lm')


def is_pooling_task(task_type: Optional[str]) -> bool:
    """Whether ``task_type`` is a pooling forward rather than generation."""
    return (task_type or 'causal_lm') in _POOLING_TASK_MAP


def to_pooling_params(task_type: Optional[str] = None, **overrides) -> Any:
    """dev ``task_type`` (+ overrides) -> twinkle ``PoolingParams``.

    The pooling counterpart of :func:`to_sampling_params`. ``task_type`` selects the head via
    :func:`pooling_task_for` (defaulting to ``embed``), and a ``reranker`` also sets ``is_cross_encoder``
    (see below); ``overrides`` win and carry the post-processing knobs ``PoolingParams`` exposes
    (``use_activation``/``dimensions``/``normalize``), which have no dev Config field and are expected to
    arrive that way.
    """
    from twinkle.data_format import PoolingParams

    task_type = task_type or 'causal_lm'
    params: Dict[str, Any] = {'task': pooling_task_for(task_type) or 'embed'}
    # A dev ``reranker`` is a cross-encoder: it scores a (query, document) pair rather than classifying
    # a single sequence. vLLM infers this from the scoring model, so the flag is a no-op there, but
    # sglang routes on it (and then wants the raw text pair, not token ids).
    if task_type == 'reranker':
        params['is_cross_encoder'] = True
    params.update(overrides)
    return PoolingParams(**params)


def pooled_data(responses: List[Any]) -> List[List[float]]:
    """PoolingResponse list -> the pooled floats of each, one entry per input.

    The pooling counterpart of :func:`sampled_texts`: ``data`` is already a flat ``List[float]`` (see
    ``pooling_to_list``), so this is just the projection, kept here so the recipes do not each reach
    into the response.
    """
    return [list(response.data) for response in responses]
