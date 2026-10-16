# Copyright (c) ModelScope Contributors. All rights reserved.
"""The shared reward-channel layer: resolve, build, place and score a model reward channel.

One reward channel (``--orm`` / ``--prm``) names either a rule (a ``reward`` extension-point plugin) or a
reward MODEL. This module owns everything about the model case, so that BOTH businesses that score with a
reward model -- inference/synthesis (``recipe/run_infer.py``) and online RL (``recipe/grpo.py`` /
``run_ppo.py``) -- drive the SAME code instead of a second copy (basic principle: a capability two
businesses use lives in a common file, and neither business imports the other's recipe):

* :class:`RewardModelSpec` / :class:`RewardSlot` -- how one channel is served, resolved before anything is
  built (a generative judge's reward LoRA and DeviceGroup must be known up front).
* :func:`resolve_reward_model` / :func:`resolve_reward_channel` -- the scalar / generative-reuse /
  generative-independent / api decision, driven by the reward model's real ``task_type``.
* :class:`RewardScorer` and its three concrete scorers (:class:`ModelReward`, :class:`GenerativeJudgeReward`,
  :class:`ApiJudgeReward`) -- the ``func(completions, **columns)`` callables a channel scores with.
* :func:`build_model_reward` / :func:`build_judge_sampler` -- spec -> scorer construction.
* :func:`exclusive_reward_groups` / :func:`plan_sampling_device_groups` -- the ray DeviceGroup placement a
  GPU-resident reward needs, derived from the channel's own ``parallel_spec``.
* :func:`normalize_scores` and the judge prompt/parse helpers.

Backend- and sampler-agnostic: a scalar RM scores by ``forward_only`` logits, a generative judge by
prompting a twinkle sampler (reused or owned), an api judge through an OpenAI-compatible client. The
judge template is a plain ``str`` argument here (NOT read off ``InferConfig``) so RL -- which carries no
``InferConfig`` -- reuses the same builder.
"""
from __future__ import annotations
import copy
import os
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Protocol, Tuple

from swift.dev.utils.logger import get_logger

if TYPE_CHECKING:
    from swift.dev.config import (
        DistributedConfig,
        ModelConfig,
        QuantizeConfig,
        TemplateConfig,
    )

logger = get_logger()

#: The DeviceGroup name the sampling engine itself is placed in. A GPU-resident reward channel gets its own
#: disjoint group (see :func:`plan_sampling_device_groups`), so the name only has to be stable between the
#: planner and the sampler build that targets it via ``remote_group``.
SAMPLER_GROUP = 'sampler'

#: Reward-model ``task_type``s that score by a pooling/classification forward (a scalar per candidate).
SCALAR_TASK_TYPES = frozenset({'seq_cls', 'reranker'})
#: Reward-model ``task_type``s that are causal_lm-shaped and can be prompted as an LLM-as-judge.
GENERATIVE_TASK_TYPES = frozenset({'causal_lm', 'generative_reranker'})

#: Upper bound on a judge's generation. Small on purpose: the verdict is a short analysis plus a
#: ``Reward: <score>`` line, and capping it keeps a chatty judge from dominating the run's cost.
JUDGE_MAX_NEW_TOKENS = 512

#: Built-in LLM-as-judge prompt, overridable via the caller's ``judge_template``. A ``str.format`` template
#: rendered with ``prompt=`` (the conversation that produced the candidate) and ``completion=`` (the
#: candidate itself). It follows swift's ``GenRMPlugin`` convention -- the judge is asked to end with
#: ``Reward: <score>`` in [0, 1] -- so :func:`parse_judge_score` reads it back the same way. It deliberately
#: carries no literal braces beyond the two placeholders, since ``str.format`` would treat any other
#: ``{...}`` as a field.
DEFAULT_JUDGE_TEMPLATE = (
    'You are a strict reward model. Based on the dialogue history, analyze in detail whether the '
    "candidate response is accurate, complete, and relevant.\n"
    'Assign a reward score between 0 and 1, where 0 indicates completely incorrect and 1 indicates '
    'fully correct. Before finishing your response, assign the reward using the following format:\n\n'
    'Reward: <score>\n\n'
    'For example:\nReward: 0.85\n\n'
    '=== Dialogue history ===\n{prompt}\n\n'
    '=== Candidate response ===\n{completion}\n\n'
    'Now analyze the candidate response and end with a single line "Reward: <score>".')


@dataclass
class RewardModelSpec:
    """How one model reward channel (``orm_model`` / ``prm_model``) will be served.

    Resolved before the sampler is built (see :func:`resolve_reward_model`) so a generative judge's reward
    LoRA can be merged into the sampler's adapter list at construction time.

    ``kind`` is one of:
    - ``scalar``: a seq_cls/reranker RM, built frozen and scored by forward logits (:class:`ModelReward`).
    - ``generative_reuse``: a causal_lm judge sharing the sampler's model, scored through that sampler.
    - ``generative_independent``: a causal_lm judge on its own model, scored through its own engine.
    - ``api``: an http(s) endpoint judged through an OpenAI-compatible client (:class:`ApiJudgeReward`).

    ``parallel_spec`` is this reward model's OWN parallel layout (``--orm_parallel_spec`` /
    ``--prm_parallel_spec``), distinct from the run's ``DistributedConfig.parallel_spec`` (the policy /
    sampler's). None means the channel inherits the sampler's width as a pure data-parallel replica set.
    """

    kind: str
    model_id: Optional[str] = None
    adapter: Optional[str] = None
    task_type: Optional[str] = None
    #: Explicit identity overrides for a SCALAR reward model, positional with the channel's model
    #: subsequence (``--orm_model_type`` / ``--orm_model_revision`` / ``--orm_template`` and the prm twins
    #: on RewardConfig). ``None`` lets the checkpoint's own metadata decide (see
    #: :func:`~swift.dev.rewards.reward.build_frozen_reward_model`); a generative / api judge ignores them
    #: (its identity is the judge model itself).
    model_type: Optional[str] = None
    revision: Optional[str] = None
    template_name: Optional[str] = None
    parallel_spec: Optional[str] = None

    @property
    def display_name(self) -> str:
        return self.model_id or self.adapter or 'judge'

    @property
    def needs_device_group(self) -> bool:
        """True when this reward keeps its own weights resident on GPU and so needs a dedicated DeviceGroup.

        A scalar RM and an independent generative judge own their weights; a sampler-reusing judge scores
        through the sampler's engine and an API judge has no local model -- neither needs its own cards.
        """
        return self.kind in ('scalar', 'generative_independent')

    def group_world_size(self, distributed_config: Optional['DistributedConfig']) -> int:
        """The GPU width this reward's DeviceGroup needs.

        With a ``parallel_spec`` it is the spec's own world size (the product of its mesh dims); without
        one the channel inherits the sampler's width (``nproc_per_node``). ``distributed_config`` may be
        None when the caller is only probing whether a group is needed, so the no-spec fallback degrades
        to 1 rather than failing -- a GPU-resident reward with no ray config is rejected upstream anyway.
        """
        if self.parallel_spec is not None:
            from twinkle import DeviceMesh
            return DeviceMesh.from_spec(self.parallel_spec).world_size
        return getattr(distributed_config, 'nproc_per_node', None) or 1

    def group_gpus_per_worker(self) -> int:
        """GPUs each worker in this reward's DeviceGroup sees: the spec's tp width, else one.

        An independent judge's engine is told ``tensor_parallel_size`` = the spec's tp (see
        :func:`build_judge_sampler`), so each of its workers must see exactly that many GPUs -- this is
        what makes the mesh's tp dim real instead of dead info. A scalar RM has no tp (the transformers
        backend does not shard weights by tp), so this is one and its group is a pure data-parallel set.
        """
        if self.parallel_spec is None:
            return 1
        from twinkle import DeviceMesh
        return DeviceMesh.from_spec(self.parallel_spec).tp_world_size or 1


@dataclass
class RewardSlot:
    """One item of a merged ``--orm`` / ``--prm`` list, resolved but not yet built.

    A slot is EITHER a rule (``rule`` holds the registered name / plugin class / callable the ``reward``
    extension point resolves) OR a reward model (``spec`` holds how it is served). Keeping the two in one
    ordered list is what lets ``orm_weights`` / ``prm_weights`` align to the order the user typed: a rule
    is built after the sampler, a model spec is resolved before it (its LoRA and DeviceGroup must be known
    up front), yet each stays at the position it was given.
    """

    rule: Optional[Any] = None
    spec: Optional[RewardModelSpec] = None

    @property
    def is_model(self) -> bool:
        return self.spec is not None


def resolve_reward_model(value: Optional[str], adapter: Optional[str], sampler_model: Optional[str],
                         backend: str) -> Optional[RewardModelSpec]:
    """Decide HOW a model reward channel is served, from its value plus the sampling backend.

    Three facts drive the choice, and their order matters:
    - An http(s) URL is an API judge. This is checked FIRST because ``get_model_info_meta`` would try to
      resolve (and download) the value as a model id, which a URL is not.
    - A value's real ``task_type`` comes from ``get_model_info_meta``: seq_cls/reranker -> a scalar RM;
      causal_lm/generative_reranker -> a generative judge. This is what tells a scalar RM (which cannot
      reuse a generation sampler -- different model class, no LM head to prompt) from a generative one.
    - ``value is None`` with an ``adapter`` means "reuse the sampler's own model as a generative judge,
      plus this reward LoRA"; with neither, there is no model channel at all.

    A generative judge reuses the sampler only when its base is the sampler's own model AND the sampler is
    local (``backend`` not 'client'/'no', which load no local engine); otherwise it builds its own. A scalar
    RM and an independent generative judge are always built locally, regardless of backend -- the 'client'
    sampling backend only means the SAMPLER is remote, not that no local model may exist.
    """
    local_backend = backend not in {'client', 'no'}
    if value is None:
        if adapter is None:
            return None
        if not local_backend:
            raise ValueError(f'orm_adapters/prm_adapters={adapter!r} without a model id in the channel means "reuse '
                             f"the sampler's model as a generative judge\", but backend={backend!r} loads no local "
                             'sampler to reuse. Point the channel at a local generative model or an http(s) endpoint.')
        return RewardModelSpec('generative_reuse', model_id=sampler_model, adapter=adapter, task_type='causal_lm')
    if value.startswith('http://') or value.startswith('https://'):
        return RewardModelSpec('api', model_id=value, adapter=adapter)
    from swift.model import get_model_info_meta

    model_info, _ = get_model_info_meta(value)
    task_type = model_info.task_type or 'causal_lm'
    if task_type in SCALAR_TASK_TYPES:
        return RewardModelSpec('scalar', model_id=value, adapter=adapter, task_type=task_type)
    if task_type in GENERATIVE_TASK_TYPES:
        if local_backend and value == sampler_model:
            return RewardModelSpec('generative_reuse', model_id=value, adapter=adapter, task_type=task_type)
        return RewardModelSpec('generative_independent', model_id=value, adapter=adapter, task_type=task_type)
    raise ValueError(f'reward model {value!r} resolved to task_type={task_type!r}, which is neither a scalar '
                     f'RM {sorted(SCALAR_TASK_TYPES)} nor a generative judge {sorted(GENERATIVE_TASK_TYPES)}.')


def resolve_reward_channel(items: List[Any], adapters: Optional[List[str]], sampler_model: Optional[str],
                           backend: str, *, parallel_spec: Optional[str] = None, channel: str,
                           model_types: Optional[List[str]] = None,
                           model_revisions: Optional[List[str]] = None,
                           template_names: Optional[List[str]] = None) -> List[RewardSlot]:
    """Resolve one merged ``--orm`` / ``--prm`` list into ordered rule/model slots.

    Registered-name-first: an item that names a registered reward rule (or is a plugin class/callable) is a
    rule slot; any other item is a reward-model id, resolved through :func:`resolve_reward_model`. Order is
    preserved so the channel's ``*_weights`` align to the list the user typed.

    A channel carries at most ONE reward model -- one DeviceGroup and one ``*_parallel_spec`` pair with one
    model -- so a second model id is rejected rather than silently sharing the channel's group. The
    companion lists (``orm_adapters`` / ``orm_model_type`` / ``orm_model_revision`` / ``orm_template`` and
    the prm twins, all on RewardConfig) align positionally with that model subsequence: one entry per
    model id, or ``None``. The one legacy exception: ``adapters`` with a single entry and NO model id in
    the channel keeps its old meaning -- reuse the sampler's own model as a generative judge wearing that
    LoRA -- and is appended after the rules.

    ``parallel_spec`` (``--orm_parallel_spec`` / ``--prm_parallel_spec``) is that model's own parallel
    layout; it is attached to the channel's model spec so the DeviceGroup width, the scalar RM's DeviceMesh
    and an independent judge's engine TP all derive from one place. A rule slot holds no model and ignores
    it; a sampler-reusing / API judge needs no device group, so the spec is inert there.
    """
    from swift.dev.rewards.reward import is_registered_reward

    is_rule = [is_registered_reward(item) for item in items]
    model_ids = [item for item, rule in zip(items, is_rule) if not rule]
    if len(model_ids) > 1:
        raise ValueError(
            f'--{channel} resolves at most one reward model alongside any number of rule names, but got '
            f'{len(model_ids)} model ids: {model_ids}. Keep one model id per channel.')

    def _aligned(values: Optional[List[Any]], field: str, *, solo_ok: bool = False) -> Optional[List[Any]]:
        """One companion entry per model id, or None; ``solo_ok`` keeps the sampler-reuse LoRA exception."""
        if values is None:
            return None
        values = list(values)
        if len(values) == len(model_ids):
            return values
        if solo_ok and not model_ids and len(values) == 1:
            return values
        raise ValueError(
            f'--{channel}_{field} must carry exactly one entry per model id in the channel '
            f'({len(model_ids)} model id(s)), but got {len(values)}: {values}.')

    adapter_list = _aligned(adapters, 'adapters', solo_ok=True)
    model_type_list = _aligned(model_types, 'model_type')
    revision_list = _aligned(model_revisions, 'model_revision')
    template_list = _aligned(template_names, 'template')
    slots: List[RewardSlot] = []
    for item, rule in zip(items, is_rule):
        if rule:
            slots.append(RewardSlot(rule=item))
        else:
            index = model_ids.index(item)
            spec = resolve_reward_model(
                item, adapter_list[index] if adapter_list else None, sampler_model, backend)
            if spec is not None:
                spec.parallel_spec = parallel_spec
                spec.model_type = model_type_list[index] if model_type_list else None
                spec.revision = revision_list[index] if revision_list else None
                spec.template_name = template_list[index] if template_list else None
            slots.append(RewardSlot(spec=spec))
    if adapter_list and not model_ids:
        # No model id in the list: the lone adapter entry means "reuse the sampler as a generative judge".
        spec = resolve_reward_model(None, adapter_list[0], sampler_model, backend)
        if spec is not None:
            spec.parallel_spec = parallel_spec
            slots.append(RewardSlot(spec=spec))
    return slots


class RewardScorer(Protocol):
    """The one contract every reward-model scorer meets: score a batch of completions, then release.

    Structural (a :class:`typing.Protocol`), so it names and documents the shape :func:`build_model_reward`
    returns without forcing the concrete scorers (:class:`ModelReward`, :class:`GenerativeJudgeReward`,
    :class:`ApiJudgeReward`) to inherit anything -- each already has a ``__call__(completions, **columns)``
    and a ``shutdown()``. A plain rule callable meets ``__call__`` too but has no ``shutdown`` (it holds no
    GPU); rules are resolved elsewhere and are not annotated here, and a channel's shutdown calls
    ``shutdown`` only when a func has one.
    """

    #: The placement (DeviceGroup) identity this scorer forwards on, read by
    #: :func:`~swift.dev.rewards.reward.score_completions` to overlap scorers on disjoint placements and
    #: serialize scorers that contend for one. ``None`` means "no dedicated device" -- a rule, a
    #: sampler-reusing judge, or an API judge -- so such scorers share one group. A scorer without the
    #: attribute at all is read as ``None`` too, so a plain rule callable needs not define it.
    placement_key: Optional[Any]

    def __call__(self, completions: List[str], **columns: Any) -> List[Optional[float]]:
        ...

    def shutdown(self) -> None:
        ...


def render_conversation(messages: Optional[List[Dict[str, Any]]]) -> str:
    """Flatten a message list into ``Role: content`` lines, matching ``GenRMPlugin.messages_to_query``."""
    lines = []
    for message in messages or []:
        if not isinstance(message, dict):
            continue
        role = message.get('role')
        content = message.get('content')
        if not role or not content:
            continue
        lines.append(f'{str(role).capitalize()}: {content}')
    return '\n'.join(lines)


def parse_judge_score(text: Optional[str]) -> Optional[float]:
    """Read a judge's numeric verdict out of its free-form output.

    Prefers the ``Reward: <score>`` convention the judge is prompted for; falls back to the first bare
    number so a judge that answers ``0.7`` without the label still scores. Returns ``None`` (which
    ``compute_rewards_per_func`` records as ``nan``) when nothing numeric is found, rather than guessing 0
    -- a silent 0 would rank an unparseable verdict as a genuinely bad candidate.
    """
    if not text:
        return None
    match = re.search(r'Reward:\s*([0-1](?:\.\d+)?)', text, flags=re.IGNORECASE)
    if match:
        return float(match.group(1))
    match = re.search(r'-?\d+(?:\.\d+)?', text)
    if match:
        return float(match.group(0))
    return None


def judge_query(template: str, prompt_messages: Optional[List[Dict[str, Any]]], completion: str) -> str:
    """Render one judge prompt: the conversation that produced a candidate, plus the candidate itself."""
    messages = list(prompt_messages or [])
    # The candidate arrives as the trailing assistant turn of ``messages``; strip it so the prompt is the
    # conversation alone and the completion is supplied verbatim through the template's ``completion=``.
    if messages and messages[-1].get('role') == 'assistant':
        messages = messages[:-1]
    return template.format(prompt=render_conversation(messages), completion=completion)


class GenerativeJudgeReward:
    """LLM-as-judge reward: prompt a generative model to score each candidate, parse the number out.

    A generative RM is a causal_lm and shares the sampler's role, so when its base is the sampler's own
    model it REUSES that engine -- the reward LoRA (if any) is resident in the sampler's adapter list at
    construction and selected per request via ``adapter_path``. Otherwise it owns a standalone engine
    (``owns_sampler``), released by :meth:`shutdown` (which the channel's shutdown calls).

    Scoring runs through a twinkle sampler, NOT a legacy engine: this is the dev-native counterpart of
    swift's ``GenRMPlugin``, reusing its judge-prompt and ``Reward: <score>`` parsing conventions.
    """

    def __init__(self,
                 sampler: Any,
                 *,
                 judge_template: Optional[str] = None,
                 adapter_path: Optional[str] = None,
                 params: Any = None,
                 owns_sampler: bool = False):
        self.sampler = sampler
        self.template = judge_template or DEFAULT_JUDGE_TEMPLATE
        self.adapter_path = adapter_path
        self.params = params
        self.owns_sampler = owns_sampler
        #: An independent judge owns a standalone engine, so it is its own placement and two such judges
        #: overlap; a sampler-reusing judge scores through the rollout engine it shares with generation and
        #: with any other reusing judge, so it declares no dedicated placement (``None``) and groups with the
        #: rules -- serializing rather than issuing concurrent generations against one engine.
        self.placement_key = id(sampler) if owns_sampler else None

    def __call__(self, completions: List[str], messages=None, **kwargs) -> List[Optional[float]]:
        from swift.dev.builders import sampled_texts

        messages = messages or [[] for _ in completions]
        trajectories = [
            {
                'messages': [{
                    'role': 'user',
                    'content': judge_query(self.template, prompt, completion)
                }]
            } for prompt, completion in zip(messages, completions)
        ]
        sample_kwargs = {'adapter_path': self.adapter_path} if self.adapter_path is not None else {}
        texts = sampled_texts(self.sampler.sample(trajectories, self.params, **sample_kwargs))
        return [parse_judge_score(group[0] if group else '') for group in texts]

    def shutdown(self) -> None:
        if self.owns_sampler and self.sampler is not None:
            self.sampler.shutdown()


class ApiJudgeReward:
    """LLM-as-judge reward served by a remote OpenAI-compatible endpoint.

    The API counterpart of :class:`GenerativeJudgeReward` -- same judge template and score parser, but
    generation happens on a remote server through the OpenAI client (the pattern a 'client' sampler already
    uses, including its thread-pool concurrency). Chosen when ``orm_model`` / ``prm_model`` is an http(s)
    URL. The judge model name is taken from ``engine_args['model']``, as the 'client' sampler backend does;
    there is deliberately no separate field for it.
    """

    def __init__(self,
                 base_url: str,
                 *,
                 model: Optional[str] = None,
                 api_key: Optional[str] = None,
                 judge_template: Optional[str] = None,
                 max_workers: int = 8,
                 timeout: float = 600.0):
        from twinkle import requires
        requires('openai')
        from openai import OpenAI

        if not model:
            raise ValueError('an API judge reward needs the judge model name; put it in engine_args '
                             "({'model': ..., 'base_url': ...}) the way the 'client' sampler backend does.")
        self.model = model
        self.client = OpenAI(
            base_url=base_url, api_key=api_key or os.environ.get('OPENAI_API_KEY'), timeout=timeout)
        self.template = judge_template or DEFAULT_JUDGE_TEMPLATE
        self.max_workers = max_workers
        #: No GPU and no local engine -- generation happens on a remote server, and this scorer already
        #: overlaps its own requests on an internal thread pool -- so it declares no dedicated placement.
        self.placement_key = None

    def __call__(self, completions: List[str], messages=None, **kwargs) -> List[Optional[float]]:
        from concurrent.futures import ThreadPoolExecutor

        messages = messages or [[] for _ in completions]
        queries = [judge_query(self.template, prompt, completion)
                   for prompt, completion in zip(messages, completions)]
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            texts = list(pool.map(self._one, queries))
        return [parse_judge_score(text) for text in texts]

    def _one(self, query: str) -> str:
        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[{
                    'role': 'user',
                    'content': query
                }],
                temperature=0.0,
                max_tokens=JUDGE_MAX_NEW_TOKENS)
        except Exception as exc:  # noqa: BLE001
            # Degrade to an unparseable (nan) score rather than ending a long run: a rate-limited judge on
            # one candidate must not abort the whole synthesis.
            logger.warning(f'API judge request failed, scoring this candidate as unparseable: {exc}')
            return ''
        return completion.choices[0].message.content or ''

    def shutdown(self) -> None:
        """Nothing to release: the OpenAI client holds no GPU and closes its own connections."""


class ModelReward:
    """Score candidates with a frozen scalar (seq_cls/reranker) reward model, dev-native.

    Built through the SAME shared builder every consumer uses
    (:func:`swift.dev.rewards.reward.build_frozen_reward_model`), so a scalar RM here and in RL are
    identical: its real ``task_type`` / ``num_labels`` come from the checkpoint's metadata rather than a
    hard-coded head, it is built with ``build_model`` (the run's ``distributed_config``, a dedicated
    ``remote_group`` and its own ``parallel_spec`` threaded through), and an optional frozen LoRA is
    attached the same way. It scores by ``forward_only`` logits (via the scalar seq-cls RM plugin), NOT by
    prompting -- that is the generative judge's job.
    """

    def __init__(self,
                 model_id: str,
                 model_config: ModelConfig,
                 template_config: TemplateConfig,
                 *,
                 adapter: Optional[str] = None,
                 model_type: Optional[str] = None,
                 revision: Optional[str] = None,
                 template_name: Optional[str] = None,
                 distributed_config: Optional[DistributedConfig] = None,
                 remote_group: Optional[str] = None,
                 parallel_spec: Optional[str] = None):
        from swift.dev.rewards.reward import build_frozen_reward_model, build_reward_model_plugins

        model, template = build_frozen_reward_model(
            model_id,
            model_config,
            template_config,
            model_type=model_type,
            revision=revision,
            template_name=template_name,
            adapter=adapter,
            distributed_config=distributed_config,
            remote_group=remote_group,
            parallel_spec=parallel_spec)
        plugins, _ = build_reward_model_plugins([model], [template])
        self.model = model
        self.plugin = plugins[0]
        #: This reward model's own actor identity: each scalar RM is a distinct model on its own DeviceGroup,
        #: so separate reward models score concurrently while a scorer never contends with itself. Matches
        #: the ``placement_key`` the scalar plugin carries (see ``_ScalarRewardModelPlugin``).
        self.placement_key = id(model)

    def __call__(self, completions: List[str], messages=None, **kwargs) -> List[float]:
        messages = messages or [[] for _ in completions]
        rows = []
        for prompt, completion in zip(messages, completions):
            prompt = list(prompt or [])
            if not prompt or prompt[-1].get('role') != 'assistant' or prompt[-1].get('content') != completion:
                prompt.append({'role': 'assistant', 'content': completion})
            rows.append({'messages': prompt})
        scores = self.plugin(inputs=rows)
        return [float(value) for value in scores.tolist()]

    def shutdown(self) -> None:
        shutdown = getattr(self.model, 'shutdown', None)
        if shutdown is not None:
            shutdown()


def build_judge_sampler(spec: RewardModelSpec, model_config: ModelConfig, template_config: TemplateConfig,
                        backend: str, engine_args: Optional[Dict[str, Any]],
                        distributed_config: Optional[DistributedConfig],
                        quantize_config: Optional[QuantizeConfig],
                        remote_group: Optional[str] = None) -> Any:
    """Build a standalone generative judge engine for a judge that cannot reuse the sampler.

    A copy of the run's ModelConfig repointed at the judge's own model, with a generation ``task_type``
    (never a pooling one -- that would build an embedding/classify engine with no LM head to prompt). The
    'client'/'no' sampling backends have no local engine to inherit a backend from, so a real sampler
    backend is substituted. Under ray it is placed in its channel's own reward DeviceGroup (``remote_group``)
    so it does not collide with the sampler; in local mode there is no group to target and it shares the
    process's devices.

    The judge's own ``parallel_spec`` (``--orm_parallel_spec`` / ``--prm_parallel_spec``), when given, drives
    BOTH its DP mesh and its engine tensor-parallel width, so the mesh's tp dim is real rather than dead
    info: ``tensor_parallel_size`` is set to the spec's tp (matching the ``gpus_per_worker`` the DeviceGroup
    was planned with), and the DP mesh comes from the spec instead of the run's config. With no spec it
    inherits the run's ``engine_args`` and DP-only mesh, exactly as before.
    """
    from swift.dev.builders import build_parallel_mesh, build_sampler, build_template, load_model_processor

    judge_backend = backend if backend in {'vllm', 'sglang', 'transformers'} else 'vllm'
    judge_model_config = copy.copy(model_config)
    judge_model_config.model = spec.model_id
    judge_model_config.task_type = spec.task_type or 'causal_lm'
    _, processor = load_model_processor(judge_model_config)
    judge_template = build_template(template_config, processor)
    if spec.parallel_spec is not None:
        from twinkle import DeviceMesh
        judge_mesh = DeviceMesh.from_spec(spec.parallel_spec)
        # Both vLLM and SGLang read the engine's TP from engine_args['tensor_parallel_size']; setting it from
        # the same spec that sized the DeviceGroup's gpus_per_worker keeps the two in agreement.
        judge_engine_args = dict(engine_args or {})
        judge_engine_args['tensor_parallel_size'] = judge_mesh.tp_world_size or 1
    else:
        judge_mesh = build_parallel_mesh(distributed_config, none_if_single_rank=True)
        judge_engine_args = engine_args
    return build_sampler(
        judge_model_config,
        backend=judge_backend,
        engine_args=judge_engine_args,
        device_mesh=judge_mesh,
        template=judge_template,
        adapters=[spec.adapter] if spec.adapter else None,
        remote_group=remote_group,
        quantize_config=quantize_config)


def build_model_reward(spec: RewardModelSpec,
                       *,
                       judge_template: Optional[str] = None,
                       sampler: Any = None,
                       model_config: Optional[ModelConfig] = None,
                       template_config: Optional[TemplateConfig] = None,
                       backend: str = 'vllm',
                       engine_args: Optional[Dict[str, Any]] = None,
                       distributed_config: Optional[DistributedConfig] = None,
                       quantize_config: Optional[QuantizeConfig] = None,
                       remote_group: Optional[str] = None) -> RewardScorer:
    """Turn a resolved :class:`RewardModelSpec` into a ``func(completions, **columns)`` callable.

    ``judge_template`` is the caller's own LLM-as-judge prompt (``InferConfig.judge_template`` for
    inference, ``RLConfig.judge_template`` for RL); it is a plain string rather than a config object so
    both businesses -- which carry different configs -- reuse this one builder.
    """
    from swift.dev.builders import to_sampling_params

    if spec.kind == 'scalar':
        return ModelReward(spec.model_id,
                           model_config,
                           template_config,
                           adapter=spec.adapter,
                           model_type=spec.model_type,
                           revision=spec.revision,
                           template_name=spec.template_name,
                           distributed_config=distributed_config,
                           remote_group=remote_group,
                           parallel_spec=spec.parallel_spec)
    if spec.kind == 'api':
        return ApiJudgeReward(
            spec.model_id, model=(engine_args or {}).get('model'), judge_template=judge_template)
    # Generative judge: greedy decoding so the verdict is deterministic across a run.
    judge_params = to_sampling_params(None, temperature=0.0, max_tokens=JUDGE_MAX_NEW_TOKENS)
    if spec.kind == 'generative_reuse':
        return GenerativeJudgeReward(
            sampler, judge_template=judge_template, adapter_path=spec.adapter, params=judge_params)
    judge_sampler = build_judge_sampler(
        spec, model_config, template_config, backend, engine_args, distributed_config, quantize_config,
        remote_group)
    return GenerativeJudgeReward(
        judge_sampler,
        judge_template=judge_template,
        adapter_path=spec.adapter,
        params=judge_params,
        owns_sampler=True)


def normalize_scores(scores: List[float]) -> List[float]:
    """Min-max the group into [0, 1], as legacy's ``normalize`` did.

    A degenerate group (every score equal) has no spread to stretch, so it collapses to a constant --
    ``min(1.0, value)`` for a positive score, 0.0 otherwise. Returning the raw values instead would make the
    threshold behave differently for degenerate and non-degenerate groups.
    """
    if not scores:
        return scores
    # A judge/API that returned nothing numeric scores nan. It must not poison the group's min/max -- a
    # single nan would make both nan and flatten every real score to nan too. Normalise over the real scores
    # and pass nan through untouched, so the caller still sees it as "unscored" (and can drop or null it).
    valid = [value for value in scores if value == value]
    if not valid:
        return scores
    low, high = min(valid), max(valid)
    if low == high:
        constant = min(1.0, low) if low > 0 else 0.0
        return [value if value != value else constant for value in scores]
    return [value if value != value else (value - low) / (high - low + 1e-5) for value in scores]


def exclusive_reward_groups(orm_slots: List[RewardSlot], prm_slots: List[RewardSlot],
                            distributed_config: Optional[DistributedConfig]) -> List[Tuple[str, int, int]]:
    """The DeviceGroups the GPU-resident reward channels need, in declaration order ('orm', 'prm').

    A channel needs a group when its reward model keeps weights resident (see
    :attr:`RewardModelSpec.needs_device_group`); a sampler-reusing judge and an API judge need none. Each
    channel resolves at most one model (see :func:`resolve_reward_channel`), so this yields at most one
    ``(name, world_size, gpus_per_worker)`` per channel -- the group's width and per-worker GPU count derived
    from that model's own ``parallel_spec``, or the sampler width / one GPU when it has none.
    """
    groups: List[Tuple[str, int, int]] = []
    for slots, name in ((orm_slots, 'orm'), (prm_slots, 'prm')):
        spec = next((slot.spec for slot in slots if slot.is_model and slot.spec.needs_device_group), None)
        if spec is not None:
            groups.append((name, spec.group_world_size(distributed_config), spec.group_gpus_per_worker()))
    return groups


def plan_sampling_device_groups(sampler_ranks: Optional[int],
                                reward_groups: List[Tuple[str, int, int]]
                                ) -> Tuple[List[Tuple[str, List[int], int]], int]:
    """Plan the twinkle DeviceGroups for a sampling run. Pure function (no twinkle import).

    The sampler and every GPU-resident reward channel each get a disjoint block of ranks. The sampler's block
    is ``sampler_ranks`` wide (``DistributedConfig.nproc_per_node``, NOT twinkle's ``nproc_per_node`` -- that
    is the total rank count, returned as ``total_ranks``); each reward group carries its OWN width and
    per-worker GPU count, derived from that channel's ``parallel_spec``. So the total is ``sampler_ranks +
    sum(group widths)`` rather than a uniform ``ranks_per_group * (1 + R)`` -- a channel may be wider or
    narrower than the sampler.

    ``reward_groups`` is ``(name, world_size, gpus_per_worker)`` per channel. Returns ``(groups,
    total_ranks)`` where ``groups`` is ``(name, ranks, gpus_per_worker)`` to hand twinkle.initialize. The
    sampler keeps ``gpus_per_worker=1`` (its engine's own TP is a separate, pre-existing concern).
    """
    if sampler_ranks is None or sampler_ranks < 1:
        raise ValueError(f'the sampler GPU count must be >= 1, got {sampler_ranks!r}.')
    groups: List[Tuple[str, List[int], int]] = [(SAMPLER_GROUP, list(range(sampler_ranks)), 1)]
    start = sampler_ranks
    for name, world_size, gpus_per_worker in reward_groups:
        groups.append((name, list(range(start, start + world_size)), gpus_per_worker))
        start += world_size
    return groups, start
