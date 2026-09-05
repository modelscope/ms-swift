"""Best-of-n data synthesis: run_sampling orchestration, on twinkle's Sampler.

The simplified successor to ``swift.pipelines.sampling.SwiftSampling`` + ``VanillaSampler``. Job
unchanged: generate n candidates per prompt, score them, keep the best as positives and the worst as
the rejected response -- i.e. produce DPO-shaped training data, not just completions. That
produce-vs-consume split is why this is a separate recipe from ``run_infer`` rather than a flag on it.

The checkpointed resume is kept verbatim in spirit, because a multi-hour synthesis run that cannot
resume is unusable: ``output_file.tmp`` is the live write, ``output_file.resume`` is the snapshot
taken after each completed batch, ``ckpt_state.json`` records the last finished batch index, and the
final move to ``output_file`` is what marks a run complete.

Scoring is NOT a sampler concern and needs no engine: ``SamplingConfig.reward_funcs`` names entries
in swift's ``orms`` registry (all eight are pure Python) or passes callables, resolved through
``swift.dev.reward``. So the sampling backend and the reward path are fully independent -- switching
vLLM/SGLang cannot change a score.

Two deliberate departures from legacy's scoring:
- one weighted reward list instead of separate orm/prm channels combined as ``prm + orm*10``. That
  10x was an unnamed hard-coded priority; ``reward_weights`` says the same thing explicitly.
- the ground-truth answer is no longer scored alongside the candidates. Legacy needed it as an anchor
  because it min-max normalised within each group; a plain weighted sum has no such need, and scoring
  the reference told us nothing we then used.

Also dropped: the md5 response cache (``cache_files``), and the ``client``/OpenAI teacher backend
that ``DistillSampler`` provided.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    from swift.dev.config import (DatasetConfig, DistributedConfig, GenerationConfig, ModelConfig, SamplingConfig,
                                  TemplateConfig)

logger = logging.getLogger(__name__)


def run_sampling(
    model_config: ModelConfig,
    template_config: TemplateConfig,
    dataset_config: DatasetConfig,
    sampling_config: SamplingConfig,
    generation_config: Optional[GenerationConfig] = None,
    *,
    backend: str = 'vllm',
    engine_args: Optional[Dict[str, Any]] = None,
    distributed_config: Optional[DistributedConfig] = None,
    adapters: Optional[List[str]] = None,
    output_dir: str = 'output',
    _shutdown: bool = True,
) -> str:
    """Sample, score, filter, and write DPO-shaped rows. Returns the output file path.

    Returns early without touching the engine when the output file already exists and
    ``override_exist_file`` is False -- so re-running a finished job is a no-op, not a re-do.

    Args:
        backend: 'vllm' / 'sglang' / 'transformers' to sample locally, or 'client' to distil from a
            remote OpenAI-compatible teacher (legacy's ``sampler_type=distill``). With 'client' no
            model is loaded here at all; put the endpoint in ``engine_args``.
        adapters: LoRA to sample with. Ignored by the 'client' backend, which has no local weights.
    """
    from swift.dev.builders import build_sampler, build_template, to_sampling_params
    from swift.dev.recipe.run_infer import _build_device_mesh_if_dp, _load_prompt_rows
    from swift.dev.recipe.run_sft import _initialize_twinkle
    from swift.model import get_model_processor

    if sampling_config.n_best_to_keep >= sampling_config.num_return_sequences:
        raise ValueError(f'n_best_to_keep={sampling_config.n_best_to_keep} must be < '
                         f'num_return_sequences={sampling_config.num_return_sequences}: the lowest-scoring '
                         'candidate becomes the rejected_response, so it cannot also be a positive.')

    os.makedirs(output_dir, exist_ok=True)
    paths = _CheckpointPaths(output_dir, sampling_config.output_file)
    if os.path.exists(paths.final) and not sampling_config.override_exist_file:
        logger.info(f'run_sampling: {paths.final} exists and override_exist_file is False; nothing to do.')
        return paths.final

    if distributed_config is not None:
        _initialize_for_sampling(distributed_config)

    rows = _load_prompt_rows(dataset_config, None, split_dataset_ratio=0.0)
    rows = _select_piece(rows, sampling_config.data_range)
    if not rows:
        raise ValueError('run_sampling got an empty dataset. Set DatasetConfig.dataset or .val_dataset.')

    channels = _build_channels(sampling_config)
    cache = _CandidateCache(sampling_config.cache_files)

    if backend == 'client':
        sampler = _ClientSampler(**(engine_args or {}))
    else:
        _, processor = get_model_processor(model_config.model, model_type=model_config.model_type, load_model=False)
        template = build_template(template_config, processor)
        sampler = build_sampler(
            model_config,
            backend=backend,
            engine_args=engine_args,
            device_mesh=_build_device_mesh_if_dp(distributed_config),
            template=template,
            adapters=adapters,
            remote_group=_SAMPLER_GROUP if distributed_config is not None
            and distributed_config.mode == 'ray' else None)

    batches = _plan_batches(len(rows), sampling_config.batch_size, sampling_config.max_batches)
    resume_from, write_mode = paths.prepare(sampling_config.resume)
    params = to_sampling_params(generation_config, num_samples=sampling_config.num_return_sequences)

    try:
        with open(paths.tmp, write_mode, encoding='utf-8') as f:
            for index, (start, end) in enumerate(batches):
                if index <= resume_from:
                    continue
                batch = rows[start:end]
                logger.info(f'run_sampling: batch {index + 1}/{len(batches)} ({len(batch)} prompts)')
                for line in _sample_batch(sampler, batch, params, template_config, channels, sampling_config, cache,
                                          backend):
                    f.write(line)
                f.flush()
                paths.checkpoint(index)
    finally:
        if _shutdown:
            sampler.shutdown()

    paths.finalize()
    logger.info(f'run_sampling: wrote {paths.final}')
    return paths.final


#: Name of the twinkle DeviceGroup the sampling engine is placed in under mode='ray'.
_SAMPLER_GROUP = 'sampler'


def _initialize_for_sampling(distributed_config: DistributedConfig) -> None:
    """Initialize twinkle, giving the sampler its own Ray DeviceGroup.

    Legacy declared three groups -- ``['sampler', 'prm', 'orm']`` -- because its reward models could be
    locally-loaded ``TransformersEngine`` instances that needed GPUs of their own. dev's scoring path
    has no such thing: every registered ORM is pure Python and every PRM is a remote call, so a 'prm'
    or 'orm' GPU group would reserve devices that nothing ever runs on. Only the sampler group exists,
    and it exists for the reason that matters -- keeping the engine off the trainer's devices.
    """
    import twinkle
    from twinkle import DeviceGroup

    if distributed_config.mode != 'ray':
        twinkle.initialize(mode='local')
        return
    nproc = distributed_config.nproc_per_node
    if nproc is None:
        raise ValueError("DistributedConfig.nproc_per_node is required in mode='ray' (it sizes the sampler's "
                         'Ray DeviceGroup). Pass it explicitly -- there is no default.')
    twinkle.initialize(
        mode='ray',
        nproc_per_node=nproc,
        groups=[DeviceGroup(name=_SAMPLER_GROUP, ranks=list(range(nproc)), device_type='GPU', gpus_per_worker=1)])


def _build_channels(sampling_config: SamplingConfig) -> '_RewardChannels':
    """Resolve both reward channels and log what will actually score."""
    from swift.dev.reward import get_reward_funcs

    orm_funcs, orm_names = get_reward_funcs(sampling_config.reward_funcs, sampling_config.reward_config)
    prm_funcs, prm_names = get_reward_funcs(sampling_config.prm_funcs, sampling_config.reward_config)
    channels = _RewardChannels(orm_funcs, orm_names, prm_funcs, prm_names)
    if channels.empty:
        logger.info('run_sampling: no reward funcs -- every candidate is emitted as a positive.')
    else:
        logger.info(f'run_sampling: orm={orm_names} (x{sampling_config.orm_channel_weight}) prm={prm_names}, '
                    f'normalize={sampling_config.normalize_rewards}')
    return channels


class _ClientSampler:
    """Sample from a remote OpenAI-compatible teacher, i.e. legacy's ``DistillSampler``.

    Presents the slice of twinkle's Sampler interface that ``run_sampling`` uses -- ``sample`` returning
    objects with ``.sequences[].decoded``, plus ``shutdown`` -- so distillation is a backend choice
    rather than a second code path through the recipe. It is NOT a twinkle Sampler: there is no local
    model, no device mesh, and no template (the server owns the chat formatting).

    Reasoning models are handled the way legacy did: when the response carries ``reasoning_content``
    separately from ``content``, the two are recombined as
    ``<think>{reasoning}</think>\\n\\n<answer>{content}</answer>``. Dropping the reasoning would train
    the student on conclusions without the derivation, which is the opposite of the point.
    """

    def __init__(self,
                 base_url: Optional[str] = None,
                 api_key: Optional[str] = None,
                 model: Optional[str] = None,
                 max_workers: int = 8,
                 timeout: float = 600.0,
                 **client_kwargs: Any):
        from twinkle import requires
        requires('openai')
        from openai import OpenAI

        if not model:
            raise ValueError("backend='client' needs the teacher's model name in "
                             "engine_args, e.g. engine_args={'model': 'deepseek-reasoner', "
                             "'base_url': ..., 'api_key': ...}.")
        self.model = model
        self.client = OpenAI(
            base_url=base_url, api_key=api_key or os.environ.get('OPENAI_API_KEY'), timeout=timeout, **client_kwargs)
        self.max_workers = max_workers

    def sample(self, inputs: List[Dict[str, Any]], sampling_params: Any = None, **kwargs) -> List[Any]:
        """Query the teacher once per prompt, concurrently.

        Threads rather than asyncio: the calls are network-bound and the recipe's loop is synchronous,
        so a pool keeps the concurrency without turning the caller inside out. Failures degrade to an
        empty group -- a remote teacher rate-limiting one prompt must not end a long distillation run.
        """
        from concurrent.futures import ThreadPoolExecutor

        num_samples = getattr(sampling_params, 'num_samples', 1) or 1
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            groups = list(pool.map(lambda item: self._one(item, sampling_params, num_samples), inputs))
        return [_ClientResponse(group) for group in groups]

    def _one(self, trajectory: Dict[str, Any], sampling_params: Any, num_samples: int) -> List[str]:
        request: Dict[str, Any] = {'model': self.model, 'messages': trajectory['messages'], 'n': num_samples}
        for attr, name in (('max_tokens', 'max_tokens'), ('temperature', 'temperature'), ('top_p', 'top_p'),
                           ('stop', 'stop'), ('seed', 'seed')):
            value = getattr(sampling_params, attr, None)
            if value is not None:
                request[name] = value
        try:
            completion = self.client.chat.completions.create(**request)
        except Exception as exc:  # noqa: BLE001
            logger.warning(f'teacher request failed, dropping this prompt: {exc}')
            return []
        return [_join_reasoning(choice.message) for choice in completion.choices]

    def shutdown(self) -> None:
        """Nothing to release: the OpenAI client holds no GPU and closes its own connections."""


class _ClientResponse:
    """The ``.sequences[].decoded`` shape ``sampled_texts`` expects, over plain strings."""

    def __init__(self, texts: List[str]):
        self.sequences = [_ClientSequence(text) for text in texts]
        self.prompt_token_ids = None


class _ClientSequence:

    def __init__(self, text: str):
        self.decoded = text
        self.tokens = []
        self.stop_reason = 'stop'
        self.logprobs = None


def _join_reasoning(message: Any) -> str:
    """Recombine a reasoning model's split output into one trainable string."""
    content = getattr(message, 'content', None) or ''
    reasoning = getattr(message, 'reasoning_content', None)
    if not reasoning:
        return content
    return f'<think>{reasoning}</think>\n\n<answer>{content}</answer>'


def _sample_batch(
    sampler: Any,
    batch: List[Dict[str, Any]],
    params: Any,
    template_config: TemplateConfig,
    channels: '_RewardChannels',
    sampling_config: SamplingConfig,
    cache: '_CandidateCache',
    backend: str,
) -> List[str]:
    """One batch of prompts -> the jsonl lines it contributes.

    Kept whole (sample + score + emit) so the caller's loop stays purely about checkpointing: a batch
    is either fully written and snapshotted or not written at all.

    Prompts already covered by ``cache_files`` skip generation entirely -- they are the expensive part,
    and their candidates are reused verbatim.
    """
    from swift.dev.builders import sampled_texts
    from swift.dev.recipe.run_infer import split_prompt_and_reference

    trajectories, ground_truths = split_prompt_and_reference(batch, template_config)
    wanted = sampling_config.num_return_sequences
    cached = [cache.lookup(trajectory, wanted) for trajectory in trajectories]

    to_sample = [index for index, hit in enumerate(cached) if hit is None]
    candidates: List[List[str]] = [hit or [] for hit in cached]
    if to_sample:
        kwargs: Dict[str, Any] = {'strict': sampling_config.strict} if backend == 'transformers' else {}
        fresh = sampled_texts(sampler.sample([trajectories[i] for i in to_sample], params, **kwargs))
        for index, group in zip(to_sample, fresh):
            candidates[index] = group
    if len(to_sample) < len(batch):
        logger.info(f'run_sampling: reused cached candidates for {len(batch) - len(to_sample)}/{len(batch)} prompts')

    lines: List[str] = []
    for row, trajectory, ground_truth, group in zip(batch, trajectories, ground_truths, candidates):
        lines.extend(_emit_rows(row, trajectory, ground_truth, group, channels, sampling_config))
    return lines


def _emit_rows(
    row: Dict[str, Any],
    trajectory: Dict[str, Any],
    ground_truth: Optional[str],
    candidates: List[str],
    channels: '_RewardChannels',
    sampling_config: SamplingConfig,
) -> List[str]:
    """One prompt's candidates -> the jsonl lines it contributes (possibly none).

    Without any reward func every candidate is a positive and no ``rejected_response`` is written:
    there is no ranking, so naming one candidate worse than another would be a fabrication.
    """
    candidates = [c for c in candidates if c]
    if not candidates:
        return []

    if channels.empty:
        return [_dpo_line(row, trajectory, positive, None, ground_truth) for positive in candidates]

    # The ground truth is scored in the same group as the candidates, so normalisation sees it as the
    # anchor -- but it is never emitted as a positive, since it is not something the model produced.
    scored = list(candidates)
    if sampling_config.score_ground_truth and ground_truth:
        scored = scored + [ground_truth]
    scores = channels.score(scored, row, sampling_config)[:len(candidates)]

    threshold = sampling_config.reward_threshold
    keep = [i for i in range(len(candidates)) if threshold is None or scores[i] > threshold]
    if not keep:
        return []
    if _is_too_easy(len(keep), len(candidates), sampling_config.easy_query_threshold):
        return []

    ranked = sorted(range(len(candidates)), key=lambda i: scores[i], reverse=True)
    negative_index = ranked[-1]
    positives = [i for i in ranked[:sampling_config.n_best_to_keep] if i in keep and i != negative_index]
    if not positives:
        return []
    negative = candidates[negative_index]
    logger.debug(f'scores={[round(s, 4) for s in scores]} positives={positives} negative={negative_index}')
    return [_dpo_line(row, trajectory, candidates[i], negative, ground_truth) for i in positives]


class _RewardChannels:
    """The ORM and PRM scoring channels, and how their two scores become one.

    Legacy kept them apart and combined them as ``prm + orm * 10``. That 10 was an unnamed priority --
    it made any ORM difference dominate every PRM difference. Here it is ``orm_channel_weight``, so the
    legacy ranking is reproducible by setting it to 10 and the default (1.0) is an honest sum.
    """

    def __init__(self, orm_funcs: List[Any], orm_names: List[str], prm_funcs: List[Any], prm_names: List[str]):
        self.orm_funcs = orm_funcs
        self.orm_names = orm_names
        self.prm_funcs = prm_funcs
        self.prm_names = prm_names

    @property
    def empty(self) -> bool:
        return not self.orm_funcs and not self.prm_funcs

    def score(self, candidates: List[str], row: Dict[str, Any], sampling_config: SamplingConfig) -> List[float]:
        """Score one prompt's candidates -> one float each.

        The row's own columns are broadcast across the candidates, because that is what an ORM's
        contract wants: ``MathAccuracy(completions, solution=[...])`` needs a ``solution`` entry per
        completion, and every candidate of one prompt shares the prompt's reference.
        """
        total = [0.0] * len(candidates)
        if self.orm_funcs:
            orm = self._channel(candidates, row, self.orm_funcs, sampling_config.reward_weights, sampling_config)
            total = [t + sampling_config.orm_channel_weight * value for t, value in zip(total, orm)]
        if self.prm_funcs:
            prm = self._channel(candidates, row, self.prm_funcs, sampling_config.prm_weights, sampling_config)
            total = [t + value for t, value in zip(total, prm)]
        return total

    @staticmethod
    def _channel(candidates: List[str], row: Dict[str, Any], funcs: List[Any], weights: Optional[List[float]],
                 sampling_config: SamplingConfig) -> List[float]:
        from swift.dev.reward import compute_rewards_per_func, weight_rewards

        columns = {key: [value] * len(candidates) for key, value in row.items() if key != 'messages'}
        rewards_per_func = compute_rewards_per_func(candidates, funcs, columns)
        scores = weight_rewards(rewards_per_func, weights).tolist()
        # Normalise per channel, before the channels are added: doing it after would let the channel
        # with the larger raw range decide the ranking regardless of the weights.
        return _normalize(scores) if sampling_config.normalize_rewards else scores


def _normalize(scores: List[float]) -> List[float]:
    """Min-max the group into [0, 1], as legacy's ``normalize`` did.

    A degenerate group (every score equal) has no spread to stretch, so it collapses to a constant --
    ``min(1.0, value)`` for a positive score, 0.0 otherwise. Returning the raw values instead would
    make the threshold behave differently for degenerate and non-degenerate groups.
    """
    if not scores:
        return scores
    low, high = min(scores), max(scores)
    if low == high:
        return [min(1.0, low) if low > 0 else 0.0] * len(scores)
    return [(value - low) / (high - low + 1e-5) for value in scores]


class _CandidateCache:
    """Candidates from earlier runs, keyed by prompt, i.e. legacy's ``cache_files``.

    Keyed on the prompt's messages rather than on row order, because the cache files come from other
    runs whose dataset order and slicing need not match this one's. A prompt is only served from cache
    when it has at least as many candidates as this run asks for -- fewer would silently shrink the
    group that the reward ranking then works on.
    """

    def __init__(self, cache_files: List[str]):
        self.by_prompt: Dict[str, List[str]] = {}
        for path in cache_files or []:
            if not os.path.isfile(path):
                logger.warning(f'cache_files entry {path} does not exist; ignoring it.')
                continue
            self._load(path)
        if self.by_prompt:
            logger.info(f'run_sampling: cache covers {len(self.by_prompt)} prompts from {len(cache_files)} file(s)')

    def _load(self, path: str) -> None:
        with open(path, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    # A crashed producer leaves a truncated final line; skipping it beats refusing the
                    # whole cache file.
                    logger.warning(f'skipping an unparseable line in {path}')
                    continue
                messages = entry.get('messages')
                if not messages or messages[-1].get('role') != 'assistant':
                    continue
                key = _prompt_key(messages[:-1])
                self.by_prompt.setdefault(key, []).append(messages[-1].get('content') or '')

    def lookup(self, trajectory: Dict[str, Any], wanted: int) -> Optional[List[str]]:
        if not self.by_prompt:
            return None
        cached = self.by_prompt.get(_prompt_key(trajectory['messages']))
        if cached is None or len(cached) < wanted:
            return None
        return cached[:wanted]


def _prompt_key(messages: List[Dict[str, Any]]) -> str:
    return hashlib.md5(
        json.dumps(messages, sort_keys=True, ensure_ascii=False, default=str).encode('utf-8')).hexdigest()


def _is_too_easy(n_passing: int, n_candidates: int, easy_query_threshold: Optional[float]) -> bool:
    """Whether a prompt is too easy to be worth keeping.

    A prompt almost every candidate already answers correctly carries no learning signal, and keeping
    it biases the dataset toward what the model can do. None disables the filter.
    """
    if easy_query_threshold is None:
        return False
    return n_passing / n_candidates >= easy_query_threshold


def _dpo_line(row: Dict[str, Any], trajectory: Dict[str, Any], positive: str, negative: Optional[str],
              ground_truth: Optional[str]) -> str:
    """Build one output row: the prompt with ``positive`` as its assistant turn, plus the rejection.

    The messages come from the trajectory actually sampled, not from the source row, so the emitted
    prompt is the one the model saw (system substitution included) rather than an assumed-equal
    reconstruction.
    """
    out: Dict[str, Any] = {key: value for key, value in row.items() if key != 'messages'}
    out['messages'] = list(trajectory['messages']) + [{'role': 'assistant', 'content': positive}]
    if negative is not None:
        out['rejected_response'] = negative
    if ground_truth is not None:
        out['ground_truth'] = ground_truth
    # Group id: every row from one prompt shares it, so positives can be traced back to their group.
    prompt_repr = json.dumps(trajectory['messages'], sort_keys=True, ensure_ascii=False, default=str)
    out['id'] = hashlib.md5(prompt_repr.encode('utf-8')).hexdigest()
    return json.dumps(out, ensure_ascii=False, default=str) + '\n'


def _select_piece(rows: List[Dict[str, Any]], data_range: Optional[tuple]) -> List[Dict[str, Any]]:
    """Take piece ``index`` of ``total`` -- the manual split for running several processes at once.

    The last piece absorbs the remainder, so no row is silently dropped by integer division (legacy
    truncated to ``piece_len * total``).
    """
    if not data_range:
        return rows
    index, total = data_range
    if not 0 <= index < total:
        raise ValueError(f'data_range index {index} is out of range for total {total}.')
    piece_len = len(rows) // total
    start = piece_len * index
    end = len(rows) if index == total - 1 else piece_len * (index + 1)
    return rows[start:end]


def _plan_batches(n_rows: int, batch_size: int, max_batches: Optional[int]) -> List[Tuple[int, int]]:
    """Fixed ``(start, end)`` slices, computed up front so the resume index means the same thing
    across runs. A trailing partial batch is kept, unlike legacy's ``n // batch_size`` truncation."""
    if batch_size < 1:
        raise ValueError(f'SamplingConfig.batch_size must be >= 1, got {batch_size}.')
    batches = [(start, min(start + batch_size, n_rows)) for start in range(0, n_rows, batch_size)]
    return batches[:max_batches] if max_batches else batches


class _CheckpointPaths:
    """The four-file resume scheme, kept in one place so the ordering cannot be got wrong.

    ``final`` only ever appears via the closing move, which is what makes its existence mean "this run
    finished" -- the check ``run_sampling`` opens with depends on that.
    """

    def __init__(self, output_dir: str, output_file: str):
        self.final = os.path.join(output_dir, output_file)
        self.tmp = self.final + '.tmp'
        self.resume = self.final + '.resume'
        self.state = os.path.join(output_dir, 'sampling_state.json')

    def prepare(self, resume: bool) -> Tuple[int, str]:
        """Returns ``(last_finished_batch_index, open_mode)``; -1 means start from the first batch.

        Resuming copies the snapshot back over ``tmp`` first: ``tmp`` may hold a half-written batch
        from the crash, and appending to that would emit a truncated row.
        """
        if not resume:
            for path in (self.tmp, self.resume, self.state):
                if os.path.exists(path):
                    os.remove(path)
            return -1, 'w'

        if os.path.exists(self.resume):
            shutil.copyfile(self.resume, self.tmp)
        last = -1
        if os.path.exists(self.state):
            with open(self.state, 'r', encoding='utf-8') as f:
                last = json.load(f).get('batch_index', -1)
            logger.info(f'run_sampling: resuming after batch index {last}')
        return last, 'a'

    def checkpoint(self, batch_index: int) -> None:
        """Snapshot, then record. In this order: a snapshot without a state file re-does one batch,
        whereas a state file without its snapshot would skip a batch whose rows were never saved."""
        shutil.copyfile(self.tmp, self.resume)
        with open(self.state, 'w', encoding='utf-8') as f:
            json.dump({'batch_index': batch_index}, f)

    def finalize(self) -> None:
        """Publish the snapshot as the output file, keeping any previous one under a timestamp."""
        if os.path.exists(self.final):
            shutil.move(self.final, f'{self.final}.{int(time.time())}')
        source = self.resume if os.path.exists(self.resume) else self.tmp
        shutil.move(source, self.final)
        for path in (self.tmp, self.state):
            if os.path.exists(path):
                os.remove(path)
