"""run_infer: offline inference over a dataset, on twinkle's Sampler.

dev counterpart of legacy ``swift infer`` (``swift/pipelines/infer/infer.py::SwiftInfer`` plus
``infer/utils.py``). It covers the same ground as legacy:

- three generation backends (vllm / sglang / transformers; lmdeploy is deliberately dropped),
- LoRA, either applied at request time or merged in first,
- the pooling task types (seq_cls / embedding / reranker), which do not go through a sampler at all
  because they need a forward pass rather than generation,
- streaming to the terminal, an interactive REPL (:func:`infer_cli`), incremental result writing with
  cross-process gathering, and the acc/rouge metrics.

What is structured differently from legacy, and why:

- generation and pooling are two functions with one dispatcher, instead of ``task_type`` branches
  threaded through a single ``_batch_infer``. They share almost nothing -- one wants a Sampler, the
  other a model forward -- so keeping them apart is what stops each from carrying the other's cases.
- there is no ``__getattr__`` proxy onto the engine. Legacy's ``SwiftInfer.infer`` was actually the
  engine's method, which made the public surface depend on the backend; here the recipe owns it.
- the sampler is built once and shut down in a ``finally``, so a crash mid-run still frees the GPU.
"""
from __future__ import annotations

import json
import logging
import os
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Sequence, Tuple

if TYPE_CHECKING:
    from swift.dev.config import (DatasetConfig, DistributedConfig, GenerationConfig, ModelConfig, TemplateConfig,
                                  TunerConfig)

logger = logging.getLogger(__name__)

#: Task types that need a forward pass, not generation. They have no sampler: the model produces a
#: pooled vector or a class score in one shot, and there is nothing to decode.
POOLING_TASKS = ('seq_cls', 'embedding', 'reranker', 'generative_reranker')


def run_infer(
    model_config: ModelConfig,
    template_config: TemplateConfig,
    dataset_config: DatasetConfig,
    generation_config: Optional[GenerationConfig] = None,
    *,
    backend: Literal['vllm', 'sglang', 'transformers'] = 'vllm',
    engine_args: Optional[Dict[str, Any]] = None,
    distributed_config: Optional[DistributedConfig] = None,
    tuner_config: Optional[TunerConfig] = None,
    adapters: Optional[List[str]] = None,
    merge_lora: bool = False,
    num_samples: int = 1,
    max_rows: Optional[int] = None,
    split_dataset_ratio: float = 0.01,
    output_path: Optional[str] = None,
    write_batch_size: Optional[int] = None,
    metric: Optional[Literal['acc', 'rouge']] = None,
    strict: bool = True,
    _shutdown: bool = True,
) -> List[Dict[str, Any]]:
    """Infer over a dataset and return one result row per input.

    Args:
        model_config: model id/path, dtype, ``task_type``. A pooling ``task_type`` (see
            :data:`POOLING_TASKS`) switches to the forward path and ignores ``backend``.
        template_config: chat template, and the ``system`` that overrides the dataset's.
        dataset_config: what to infer over. See ``split_dataset_ratio`` for how the split is chosen.
        generation_config: decoding knobs. ``stream=True`` prints tokens as they arrive.
        backend: generation engine. ``transformers`` is the one that runs anything and the only one
            that can degrade per row (see ``strict``).
        engine_args: forwarded verbatim to the engine.
        distributed_config: when it declares DP > 1, a DeviceMesh is built and ``sample`` slices the
            inputs across ranks by itself.
        tuner_config: source of ``adapters`` when not given explicitly.
        adapters: LoRA checkpoints. One adapter is applied to the whole run; the engine is configured
            for LoRA at construction because vLLM cannot enable it later.
        merge_lora: fold the adapters into the base weights first and infer on the merged model.
            Slower to start and needs disk, but then costs nothing per request -- which is the right
            trade for a long offline run, and the reason legacy defaulted to it for export.
        num_samples: completions per prompt. The first one is what ``response`` and the metric see.
        max_rows: stop after this many rows, for smoke tests.
        split_dataset_ratio: how much of ``dataset`` becomes the eval split when
            ``DatasetConfig.val_dataset`` is not set. Legacy's semantics, kept because a config that
            names one ``--dataset`` means "infer over the eval slice", not "over everything".
        output_path: jsonl destination.
        write_batch_size: rows per incremental flush. Without it results are written once at the end,
            so a crash loses the run; with it the file grows as inference proceeds. Under DP the
            batches are gathered across ranks before writing, so the file stays whole.
        metric: 'acc' or 'rouge', computed against the reference answers.
        strict: transformers backend only -- when False a row that fails to encode or generate is
            recorded with an empty response instead of aborting the run.
        _shutdown: leave True. False keeps the engine alive for tests that reuse it.

    Returns:
        Result rows, each with ``response`` / ``responses`` / ``labels`` / ``messages`` plus every
        column the dataset row already had.
    """
    from swift.dev.recipe.run_sft import _initialize_twinkle

    _initialize_twinkle(distributed_config)
    adapters = _resolve_adapters(adapters, tuner_config)
    if merge_lora and adapters:
        model_config, adapters = _merge_adapters(model_config, template_config, adapters)

    rows = _load_prompt_rows(dataset_config, max_rows, split_dataset_ratio)
    if not rows:
        raise ValueError('run_infer got an empty dataset. Set DatasetConfig.dataset or .val_dataset.')

    task_type = model_config.task_type or 'causal_lm'
    if task_type in POOLING_TASKS:
        return _run_pooling(model_config, template_config, distributed_config, rows, adapters, output_path, metric)

    return _run_generative(
        model_config,
        template_config,
        generation_config,
        rows,
        backend=backend,
        engine_args=engine_args,
        distributed_config=distributed_config,
        adapters=adapters,
        num_samples=num_samples,
        output_path=output_path,
        write_batch_size=write_batch_size,
        metric=metric,
        strict=strict,
        shutdown=_shutdown,
    )


def _run_generative(
    model_config: ModelConfig,
    template_config: TemplateConfig,
    generation_config: Optional[GenerationConfig],
    rows: List[Dict[str, Any]],
    *,
    backend: str,
    engine_args: Optional[Dict[str, Any]],
    distributed_config: Optional[DistributedConfig],
    adapters: Optional[List[str]],
    num_samples: int,
    output_path: Optional[str],
    write_batch_size: Optional[int],
    metric: Optional[str],
    strict: bool,
    shutdown: bool,
) -> List[Dict[str, Any]]:
    """The causal-LM path: encode prompts, sample, write."""
    from swift.dev.builders import build_sampler, build_template, to_sampling_params
    from swift.model import get_model_processor

    logger.info(f'run_infer: {len(rows)} prompts, backend={backend}, num_samples={num_samples}')
    _, processor = get_model_processor(model_config.model, model_type=model_config.model_type, load_model=False)
    template = build_template(template_config, processor)
    device_mesh = _build_device_mesh_if_dp(distributed_config)
    adapter_path = adapters[0] if adapters else None

    sampler = build_sampler(
        model_config,
        backend=backend,
        engine_args=engine_args,
        device_mesh=device_mesh,
        template=template,
        adapters=adapters)
    try:
        params = to_sampling_params(generation_config, num_samples=num_samples)
        streaming = bool(generation_config is not None and generation_config.stream)
        writer = _IncrementalWriter(output_path, write_batch_size)
        results: List[Dict[str, Any]] = []

        for batch in _batches(rows, write_batch_size or len(rows)):
            trajectories, labels = split_prompt_and_reference(batch, template_config)
            if streaming:
                texts = _sample_streaming(sampler, trajectories, params, adapter_path)
            else:
                texts = _sample_batch(sampler, trajectories, params, adapter_path, backend, strict)
            batch_results = _assemble_results(batch, trajectories, labels, texts)
            results.extend(batch_results)
            writer.write(batch_results)
    finally:
        if shutdown:
            sampler.shutdown()

    writer.finish(results)
    if metric:
        logger.info(f'run_infer metric: {compute_metric(results, metric)}')
    return results


def _sample_batch(sampler, trajectories: List[Dict[str, Any]], params: Any, adapter_path: Optional[str], backend: str,
                  strict: bool) -> List[List[str]]:
    """One ``sample()`` call for the batch. ``strict`` only reaches the transformers backend."""
    from swift.dev.builders import sampled_texts

    kwargs: Dict[str, Any] = {}
    if adapter_path is not None:
        kwargs['adapter_path'] = adapter_path
    if backend == 'transformers':
        kwargs['strict'] = strict
    elif not strict:
        logger.warning(f'strict=False is only honoured by the transformers backend; the {backend} engine fails '
                       'the whole batch on a bad row. Switch backend if per-row tolerance matters.')
    return sampled_texts(sampler.sample(trajectories, params, **kwargs))


def _sample_streaming(sampler, trajectories: List[Dict[str, Any]], params: Any,
                      adapter_path: Optional[str]) -> List[List[str]]:
    """Stream each prompt to stdout, and return the accumulated texts.

    One prompt at a time by necessity: ``sample_stream`` is a single-request API, and interleaving
    several streams onto one terminal would produce unreadable output. This is a display mode, not a
    throughput mode -- for many rows leave ``stream`` off.
    """
    texts: List[List[str]] = []
    for index, trajectory in enumerate(trajectories):
        print(f'[{index}] ', end='', flush=True)
        pieces: List[str] = []
        for delta, _finish_reason in sampler.sample_stream(trajectory, params, adapter_path=adapter_path):
            if delta:
                print(delta, end='', flush=True)
                pieces.append(delta)
        print(flush=True)
        texts.append([''.join(pieces)])
    return texts


def _run_pooling(
    model_config: ModelConfig,
    template_config: TemplateConfig,
    distributed_config: Optional[DistributedConfig],
    rows: List[Dict[str, Any]],
    adapters: Optional[List[str]],
    output_path: Optional[str],
    metric: Optional[str],
) -> List[Dict[str, Any]]:
    """The forward-pass path for seq_cls / embedding / reranker.

    Deliberately not a sampler: these produce a vector or a score from one forward, so a generation
    engine has nothing to contribute and (for vLLM/sglang) would refuse to load the pooling head at
    all. The label comes from the row's own ``label`` column rather than from a trailing assistant
    turn, because there is no completion to strip.
    """
    from swift.dev.builders import build_model, build_template
    from swift.dev.config import DistributedConfig
    from swift.model import get_model_processor

    task_type = model_config.task_type
    logger.info(f'run_infer: {len(rows)} rows, task_type={task_type} (forward pass, no sampler)')
    _, processor = get_model_processor(model_config.model, model_type=model_config.model_type, load_model=False)
    template = build_template(template_config, processor)
    model = build_model(model_config, distributed_config or DistributedConfig())
    if adapters:
        # add_adapter_to_model takes a checkpoint directory as well as a PeftConfig, so a trained
        # adapter is loaded here rather than built -- apply_tuner would create a fresh, untrained one.
        for index, adapter in enumerate(adapters):
            model.add_adapter_to_model(f'adapter_{index}' if index else 'default', adapter)

    encoded = [template.encode(_to_trajectory(row, list(row['messages']), template_config)) for row in rows]
    outputs = model.forward_only(inputs=encoded, task=task_type, return_logits=True)

    results = []
    for row, output in zip(rows, _per_row_outputs(outputs, len(rows))):
        result = {key: value for key, value in row.items() if key != 'messages'}
        results.append({
            'response': output,
            'responses': [output],
            'labels': row.get('label'),
            'messages': list(row['messages']),
            **result
        })
    if output_path:
        _write_jsonl(output_path, results)
        logger.info(f'run_infer: wrote {len(results)} rows to {output_path}')
    if metric:
        logger.info(f'run_infer metric: {compute_metric(results, metric)}')
    return results


def _per_row_outputs(outputs: Any, num_rows: int) -> List[Any]:
    """Unpack ``forward_only``'s output into one plain-Python value per row.

    The shape depends on the head: seq_cls gives logits per row, embedding gives a vector. Both are
    returned as lists so the result is jsonl-serialisable, which is the whole point of this path.
    """
    tensor = outputs
    if isinstance(outputs, dict):
        for key in ('logits', 'embedding', 'last_hidden_state'):
            if key in outputs:
                tensor = outputs[key]
                break
    if hasattr(tensor, 'tolist'):
        listed = tensor.tolist()
        if isinstance(listed, list) and len(listed) == num_rows:
            return listed
        return [listed]
    return list(tensor) if isinstance(tensor, (list, tuple)) else [tensor] * num_rows


def infer_cli(
    model_config: ModelConfig,
    template_config: TemplateConfig,
    generation_config: Optional[GenerationConfig] = None,
    *,
    backend: Literal['vllm', 'sglang', 'transformers'] = 'vllm',
    engine_args: Optional[Dict[str, Any]] = None,
    adapters: Optional[List[str]] = None,
    multi_round: bool = True,
) -> None:
    """Interactive REPL, the dev counterpart of legacy ``--eval_human true``.

    Commands (legacy's, unchanged, because muscle memory is the point of a REPL):
        ``clear`` / ``reset-system`` / ``multi-line`` / ``single-line`` / ``quit``.

    Multimodal inputs are prompted for by path when the template asks for them, matching legacy's
    ``input_mm_data``. History is kept across turns unless ``multi_round`` is False.
    """
    from swift.dev.builders import build_sampler, build_template, to_sampling_params
    from swift.model import get_model_processor

    _, processor = get_model_processor(model_config.model, model_type=model_config.model_type, load_model=False)
    template = build_template(template_config, processor)
    sampler = build_sampler(
        model_config, backend=backend, engine_args=engine_args, template=template, adapters=adapters)
    adapter_path = adapters[0] if adapters else None
    params = to_sampling_params(generation_config)
    stream = bool(generation_config is not None and generation_config.stream)

    state = _CliState(system=template_config.system)
    print('Interactive inference. Commands: clear | reset-system | multi-line | single-line | quit')
    try:
        while True:
            try:
                query = state.read_query()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if query is None:
                continue
            if query is _QUIT:
                break

            state.add_query(query)
            trajectory = state.to_trajectory()
            if stream:
                pieces = []
                for delta, _ in sampler.sample_stream(trajectory, params, adapter_path=adapter_path):
                    if delta:
                        print(delta, end='', flush=True)
                        pieces.append(delta)
                print(flush=True)
                response = ''.join(pieces)
            else:
                from swift.dev.builders import sampled_texts
                texts = sampled_texts(sampler.sample([trajectory], params, adapter_path=adapter_path))
                response = texts[0][0] if texts and texts[0] else ''
                print(response, flush=True)
            if multi_round:
                state.add_response(response)
            else:
                state.clear()
    finally:
        sampler.shutdown()


#: Sentinel returned by ``_CliState.read_query`` for 'quit', kept distinct from an empty line (which
#: means "reprompt") and from None (a command that was already handled).
_QUIT = object()


class _CliState:
    """Conversation state for :func:`infer_cli`, i.e. legacy's ``InferCliState``."""

    def __init__(self, system: Optional[str] = None):
        self.system = system
        self.messages: List[Dict[str, Any]] = []
        self.media: Dict[str, List[str]] = {'images': [], 'audios': [], 'videos': []}
        self.multiline = False

    def clear(self) -> None:
        self.messages = []
        self.media = {key: [] for key in self.media}

    def add_query(self, query: str) -> None:
        self.messages.append({'role': 'user', 'content': query})

    def add_response(self, response: str) -> None:
        self.messages.append({'role': 'assistant', 'content': response})

    def to_trajectory(self) -> Dict[str, Any]:
        trajectory: Dict[str, Any] = {'messages': list(self.messages)}
        if self.system:
            trajectory['messages'] = [{'role': 'system', 'content': self.system}] + trajectory['messages']
        for key, values in self.media.items():
            if values:
                trajectory[key] = list(values)
        return trajectory

    def read_query(self):
        """Read one turn, handling the commands. Returns the query, None (handled), or ``_QUIT``."""
        raw = self._read_raw()
        stripped = raw.strip()
        if not stripped:
            return None
        lowered = stripped.lower()
        if lowered in ('quit', 'exit'):
            return _QUIT
        if lowered == 'clear':
            self.clear()
            print('History cleared.')
            return None
        if lowered == 'reset-system':
            self.system = input('Enter the new system prompt: ').strip() or None
            self.clear()
            print(f'System set to {self.system!r}; history cleared (a mid-conversation system swap '
                  'would leave turns answered under the old one).')
            return None
        if lowered in ('multi-line', 'single-line'):
            self.multiline = lowered == 'multi-line'
            print(f'multi-line mode: {self.multiline}')
            return None
        return stripped

    def _read_raw(self) -> str:
        if not self.multiline:
            return input('<<< ')
        print('<<< (multi-line; end with a single "#" on its own line)')
        lines = []
        while True:
            line = input()
            if line.strip() == '#':
                break
            lines.append(line)
        return '\n'.join(lines)

    def prompt_media(self, kinds: Sequence[str]) -> None:
        """Ask for media paths, blank line to stop -- legacy's ``input_mm_data``."""
        for kind in kinds:
            while True:
                path = input(f'Input a {kind[:-1]} path/url (blank to finish): ').strip()
                if not path:
                    break
                self.media.setdefault(kind, []).append(path)


class _IncrementalWriter:
    """Append results to jsonl as batches complete, gathering across DP ranks first.

    Why gather: with DP > 1 each rank holds only its own slice, so a naive per-rank write either
    interleaves partial files or has every rank overwrite the same path. Legacy solved this with
    ``JsonlWriter(gather_obj=True)``; twinkle's ``gather_object`` is the same idea.

    When ``batch_size`` is None this does nothing until :meth:`finish`, which writes everything at
    once -- the simple case stays simple.
    """

    def __init__(self, output_path: Optional[str], batch_size: Optional[int]):
        self.output_path = output_path
        self.incremental = bool(output_path and batch_size)
        self._started = False

    def write(self, batch: List[Dict[str, Any]]) -> None:
        if not self.incremental:
            return
        rows = _gather_rows(batch)
        if rows is None:
            return  # not the writing rank
        _write_jsonl(self.output_path, rows, append=self._started)
        self._started = True
        logger.info(f'run_infer: flushed {len(rows)} rows to {self.output_path}')

    def finish(self, results: List[Dict[str, Any]]) -> None:
        if self.incremental or not self.output_path:
            return
        rows = _gather_rows(results)
        if rows is None:
            return
        _write_jsonl(self.output_path, rows)
        logger.info(f'run_infer: wrote {len(rows)} rows to {self.output_path}')


def _gather_rows(rows: List[Dict[str, Any]]) -> Optional[List[Dict[str, Any]]]:
    """All ranks' rows on the writing rank, None elsewhere. A no-op without torch.distributed."""
    import torch.distributed as dist

    if not (dist.is_available() and dist.is_initialized()):
        return rows

    from twinkle.utils import framework_util, is_master

    gathered = framework_util.gather_object(rows, device_mesh=None)
    return gathered if is_master() else None


def split_prompt_and_reference(rows: List[Dict[str, Any]],
                               template_config: TemplateConfig) -> Tuple[List[Dict[str, Any]], List[Optional[str]]]:
    """rows -> ``(trajectories, references)``, with each row's trailing assistant turn moved aside.

    Shared with ``run_sampling``: both have to hand the model a prompt that stops before the reference
    answer, and both need that answer afterwards (as a metric label / as ground_truth). Doing it in one
    place is what keeps "what the model saw" identical between the two recipes.
    """
    trajectories, references = [], []
    for row in rows:
        messages = list(row['messages'])
        reference = messages.pop()['content'] if messages and messages[-1]['role'] == 'assistant' else None
        references.append(reference)
        trajectories.append(_to_trajectory(row, messages, template_config))
    return trajectories, references


def _assemble_results(
    rows: List[Dict[str, Any]],
    trajectories: List[Dict[str, Any]],
    labels_list: List[Optional[str]],
    texts: List[List[str]],
) -> List[Dict[str, Any]]:
    """Zip prompts, labels and completions into the output rows.

    ``messages`` comes from the trajectory actually sampled -- prompt-only (the reference answer was
    popped) and carrying any system substitution -- so appending the response yields one assistant
    turn, not the reference followed by the model's.
    """
    results = []
    for row, trajectory, label, candidates in zip(rows, trajectories, labels_list, texts):
        messages = list(trajectory['messages'])
        if candidates:
            messages = messages + [{'role': 'assistant', 'content': candidates[0]}]
        passthrough = {key: value for key, value in row.items() if key != 'messages'}
        results.append({
            'response': candidates[0] if candidates else None,
            'responses': candidates,
            'labels': label,
            'messages': messages,
            **passthrough
        })
    return results


def _resolve_adapters(adapters: Optional[List[str]], tuner_config: Optional[TunerConfig]) -> Optional[List[str]]:
    if adapters:
        return list(adapters)
    if tuner_config is not None and getattr(tuner_config, 'adapters', None):
        return list(tuner_config.adapters)
    return None


def _merge_adapters(model_config: ModelConfig, template_config: TemplateConfig,
                    adapters: List[str]) -> Tuple[ModelConfig, None]:
    """Merge the adapters into the base weights and return a ModelConfig pointing at the result.

    The returned config is a copy: mutating the caller's would make a second run with the same object
    silently infer on the merged directory. Adapters come back as None because they are now part of
    the weights -- passing them again would apply the same delta twice.
    """
    import dataclasses

    from swift.dev.config import TunerConfig
    from swift.dev.recipe.merge_lora import run_merge_lora

    merged = run_merge_lora(
        model_config, TunerConfig(adapters=list(adapters)), template_config=template_config, device_map='cpu')
    logger.info(f'run_infer: merged {len(adapters)} adapter(s) into {merged}')
    return dataclasses.replace(model_config, model=merged), None


def _build_device_mesh_if_dp(distributed_config: Optional[DistributedConfig]) -> Any:
    """A DeviceMesh only when DP > 1: a single-process run wants a plain in-process engine."""
    if distributed_config is None:
        return None
    from swift.dev.builders import build_device_mesh

    mesh = build_device_mesh(distributed_config)
    return mesh if mesh is not None and getattr(mesh, 'data_world_size', 1) > 1 else None


def _to_trajectory(row: Dict[str, Any], messages: List[Dict[str, Any]],
                   template_config: TemplateConfig) -> Dict[str, Any]:
    """Build the twinkle Trajectory for one row.

    ``TemplateConfig.system`` REPLACES a system turn the row already has rather than stacking a second
    one, matching legacy: two system messages is not a supported prompt shape for most templates.
    """
    trajectory: Dict[str, Any] = {'messages': messages}
    system = getattr(template_config, 'system', None)
    if system:
        without_system = [message for message in messages if message.get('role') != 'system']
        trajectory['messages'] = [{'role': 'system', 'content': system}] + without_system
    for key in ('images', 'audios', 'videos', 'objects', 'tools'):
        if row.get(key):
            trajectory[key] = row[key]
    return trajectory


def _load_prompt_rows(dataset_config: DatasetConfig, max_rows: Optional[int],
                      split_dataset_ratio: float = 0.01) -> List[Dict[str, Any]]:
    """Load the rows to infer over, restoring legacy's split semantics.

    ``val_dataset`` wins outright. Otherwise ``dataset`` is split and only the eval slice is used --
    which is what a single ``--dataset`` meant in legacy. Set ``split_dataset_ratio=0`` to infer over
    the whole thing.
    """
    from swift.dev.builders.dataset import _load_kwargs
    from swift.dev.dataset import load_dataset

    kwargs = _load_kwargs(dataset_config)
    if dataset_config.val_dataset:
        _, rows = load_dataset(datasets=list(dataset_config.val_dataset), split_dataset_ratio=0.0, **kwargs)
        rows = rows if rows is not None else []
    elif split_dataset_ratio:
        _, rows = load_dataset(
            datasets=list(dataset_config.dataset), split_dataset_ratio=split_dataset_ratio, **kwargs)
        rows = rows if rows is not None else []
        logger.info(f'run_infer: using the eval split of --dataset (split_dataset_ratio='
                    f'{split_dataset_ratio}); pass split_dataset_ratio=0 to infer over all of it.')
    else:
        rows, _ = load_dataset(datasets=list(dataset_config.dataset), split_dataset_ratio=0.0, **kwargs)

    rows = list(rows)
    if max_rows is not None:
        rows = rows[:max_rows]
    return rows


def _batches(rows: List[Dict[str, Any]], size: int):
    for start in range(0, len(rows), max(1, size)):
        yield rows[start:start + max(1, size)]


def _write_jsonl(path: str, rows: List[Dict[str, Any]], append: bool = False) -> None:
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, 'a' if append else 'w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')


def compute_metric(results: List[Dict[str, Any]], metric: Literal['acc', 'rouge']) -> Dict[str, float]:
    """Score the first completion of each row against its reference answer.

    ``acc`` is exact string equality, which is what legacy's ``--metric acc`` measured -- NOT the
    token-level ``twinkle.metric.Accuracy``, which scores logits against label ids and would report a
    different (and much higher) number for the same run.

    Only the first completion is scored: with ``num_samples > 1`` a best-of-n score is a different
    measurement (pass@n) and reporting it as accuracy would overstate the model. Rows without a
    reference are skipped rather than counted as wrong.
    """
    from twinkle.metric import ExactMatch, RougeBleu

    pairs = [(r['response'], r['labels']) for r in results if r.get('labels') is not None and r.get('response')]
    if not pairs:
        logger.warning('metric requested but no row has both a response and a reference answer.')
        return {}
    predictions, references = zip(*pairs)
    scorer = ExactMatch() if metric == 'acc' else RougeBleu()
    scorer.accumulate(predictions=list(predictions), references=list(references))
    return scorer.calculate()
