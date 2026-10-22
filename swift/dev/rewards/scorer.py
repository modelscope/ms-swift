# Copyright (c) ModelScope Contributors. All rights reserved.
from __future__ import annotations
import copy
import os
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import torch

from swift.dev.utils.logger import get_logger

if TYPE_CHECKING:
    from swift.dev.config import ModelConfig, QuantizeConfig, TemplateConfig

logger = get_logger()


#: Reward-model ``task_type``s that score by a pooling/classification forward (a scalar per candidate).
SCALAR_TASK_TYPES = frozenset({'seq_cls', 'reranker'})
#: Reward-model ``task_type``s that are causal_lm-shaped and can be prompted as an LLM-as-judge.
GENERATIVE_TASK_TYPES = frozenset({'causal_lm', 'generative_reranker'})

#: Upper bound on a judge's generation: the verdict is a short analysis plus a ``Reward: <score>`` line.
JUDGE_MAX_NEW_TOKENS = 512

#: Built-in LLM-as-judge prompt, overridable via ``judge_template``. A ``str.format`` template rendered
#: with ``prompt=`` and ``completion=``; the ``Reward: <score>`` line is what parse_judge_score reads.
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
class RewardSpec:
    """How one model reward (the ``--orm`` / ``--prm`` model id) will be served.

    ``kind`` is one of ``scalar`` (a seq_cls/reranker RM, scored by forward logits), ``generative_reuse``
    (a causal_lm judge sharing the sampler's engine), ``generative_independent`` (a judge on its own
    engine), ``api`` (an http(s) endpoint through an OpenAI-compatible client). ``parallel_spec`` is
    this model's OWN parallel layout, distinct from the run's.
    """

    kind: str
    model_id: Optional[str] = None
    adapter: Optional[str] = None
    task_type: Optional[str] = None
    #: Identity overrides for a SCALAR reward model, aligned with the channel's model subsequence.
    model_type: Optional[str] = None
    revision: Optional[str] = None
    template_name: Optional[str] = None
    parallel_spec: Optional[str] = None

    @property
    def display_name(self) -> str:
        return self.model_id or self.adapter or 'judge'


@dataclass
class ChannelCompanions:
    """The per-channel companion lists a reward model's identity is aligned with (all optional).

    Each list carries exactly one entry per model id in the channel, or ``None`` for all-default;
    ``parallel_spec`` is the channel model's own parallel layout (``--orm_parallel_spec`` / prm twin).
    """

    adapters: Optional[List[str]] = None
    model_types: Optional[List[str]] = None
    model_revisions: Optional[List[str]] = None
    template_names: Optional[List[str]] = None
    parallel_spec: Optional[str] = None


@dataclass
class ChannelPlan:
    """One merged ``--orm`` / ``--prm`` list resolved, order preserved.

    ``entries`` holds, at the position the user typed each item, either a rule spec (a registered
    name / plugin class / callable -- anything is_registered_reward accepts) or a :class:`RewardSpec`.
    Order preservation is what lets the channel's ``*_weights`` align with the typed list.
    """

    entries: List[Any]

    @property
    def rules(self) -> List[Any]:
        from .reward import is_registered_reward
        return [entry for entry in self.entries if is_registered_reward(entry)]

    @property
    def specs(self) -> List[RewardSpec]:
        return [entry for entry in self.entries if isinstance(entry, RewardSpec)]


def split_channel(items: List[Any]) -> Tuple[List[Any], List[str]]:
    """Split a merged channel list into ``(rule_items, model_ids)`` -- the device-planning truth source."""
    from .reward import is_registered_reward
    rules = [item for item in items if is_registered_reward(item)]
    model_ids = [item for item in items if not is_registered_reward(item)]
    return rules, model_ids


def resolve_reward_model(value: Optional[str], adapter: Optional[str], sampler_model: Optional[str],
                         backend: str) -> Optional[RewardSpec]:
    """Decide HOW a model reward is served, from its value plus the sampling backend.

    An http(s) URL is an API judge (checked FIRST -- get_model_info_meta would try to download it);
    otherwise the checkpoint's real ``task_type`` decides scalar vs generative; ``value=None`` with an
    ``adapter`` reuses the sampler's own model as a generative judge wearing that LoRA (only when its
    base IS the sampler's model and the sampler is local); a scalar RM always builds locally.
    """
    local_backend = backend not in {'client', 'no'}
    if value is None:
        if adapter is None:
            return None
        if not local_backend:
            raise ValueError(f'orm_adapters/prm_adapters={adapter!r} without a model id in the channel means '
                             f'"reuse the sampler\'s model as a generative judge", but backend={backend!r} '
                             'loads no local sampler to reuse. Point the channel at a local generative '
                             'model or an http(s) endpoint.')
        return RewardSpec('generative_reuse', model_id=sampler_model, adapter=adapter, task_type='causal_lm')
    if value.startswith('http://') or value.startswith('https://'):
        return RewardSpec('api', model_id=value, adapter=adapter)
    from swift.model import get_model_info_meta

    model_info, _ = get_model_info_meta(value)
    task_type = model_info.task_type or 'causal_lm'
    if task_type in SCALAR_TASK_TYPES:
        return RewardSpec('scalar', model_id=value, adapter=adapter, task_type=task_type)
    if task_type in GENERATIVE_TASK_TYPES:
        if local_backend and value == sampler_model:
            return RewardSpec('generative_reuse', model_id=value, adapter=adapter, task_type=task_type)
        return RewardSpec('generative_independent', model_id=value, adapter=adapter, task_type=task_type)
    raise ValueError(f'reward model {value!r} resolved to task_type={task_type!r}, which is neither a scalar '
                     f'RM {sorted(SCALAR_TASK_TYPES)} nor a generative judge {sorted(GENERATIVE_TASK_TYPES)}.')


def resolve_channel(items: List[Any], companions: ChannelCompanions, *, channel: str,
                    sampler_model: Optional[str], backend: str) -> ChannelPlan:
    """Resolve one merged ``--orm`` / ``--prm`` list into an ordered :class:`ChannelPlan`.

    Registered-name-first: an item naming a registered rule (or a class/callable) is a rule entry; any
    other item is a reward-model id resolved through :func:`resolve_reward_model`. A channel carries
    at most ONE reward model; the companion lists align positionally with that model subsequence. The
    legacy exception: ``adapters`` with one entry and NO model id reuses the sampler as a judge.
    """
    rules, model_ids = split_channel(list(items))
    if len(model_ids) > 1:
        raise ValueError(
            f'--{channel} resolves at most one reward model alongside any number of rule names, but got '
            f'{len(model_ids)} model ids: {model_ids}. Keep one model id per channel.')

    def aligned(values: Optional[List[Any]], field: str, *, solo_ok: bool = False) -> Optional[List[Any]]:
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

    adapter_list = aligned(companions.adapters, 'adapters', solo_ok=True)
    model_type_list = aligned(companions.model_types, 'model_type')
    revision_list = aligned(companions.model_revisions, 'model_revision')
    template_list = aligned(companions.template_names, 'template')

    def model_spec(model_id: str, adapter: Optional[str]) -> Optional[RewardSpec]:
        spec = resolve_reward_model(model_id, adapter, sampler_model, backend)
        if spec is not None:
            spec.parallel_spec = companions.parallel_spec
            if model_id in model_ids:
                index = model_ids.index(model_id)
                spec.model_type = model_type_list[index] if model_type_list else None
                spec.revision = revision_list[index] if revision_list else None
                spec.template_name = template_list[index] if template_list else None
        return spec

    entries: List[Any] = []
    for item in items:
        if item in rules:
            entries.append(item)
        else:
            entries.append(model_spec(item, adapter_list[model_ids.index(item)] if adapter_list else None))
    if adapter_list and not model_ids:
        # No model id in the list: the lone adapter entry means "reuse the sampler as a judge".
        spec = model_spec(sampler_model, adapter_list[0])
        if spec is not None:
            entries.append(spec)
    return ChannelPlan(entries=entries)


class RewardModel:
    """One frozen forward-only model handle: a scalar RM scorer, or a reference/teacher logp scorer.

    ``score`` runs the seq-cls head over message rows (call it from inside a remote boundary, never
    from the driver); ``logps`` runs ``forward_only(output_key='logps')`` over feature refs (TODO
    (swift->twinkle): slice_dp must shard ref-valued list kwargs with the inputs shard).
    ``disable_adapter`` marks a policy-actor owner whose ``logps`` forwards ``disable_lora=True``.
    """

    def __init__(self, model: Any, template: Any = None, *, placement_key: Any = None,
                 disable_adapter: bool = False, display_name: Optional[str] = None):
        self.model = model
        self.template = template
        #: The DeviceGroup identity this handle forwards on; None means the policy actor itself.
        self.placement_key = placement_key
        #: The metric-column label (the RewardSpec's display name); rules carry the registry's.
        self.display_name = display_name or 'reward'
        self.disable_adapter = disable_adapter

    def score(self, rows: List[Dict[str, Any]]) -> List[float]:
        """Score message rows with the scalar head -> one float per row.

        task='seq_cls' routes both backends through the score head: forwarding the default
        'causal_lm' would gather label logps out of a pooled [B, 1] score (a shape error on
        transformers, and megatron would miss the seq_cls pooling branch entirely).
        """
        features = [self.template.encode(copy.deepcopy(row)) for row in rows]
        outputs = self.model.forward_only(inputs=features, return_logits=True, task='seq_cls')
        logits = outputs.get('logits')
        if logits is None:
            raise RuntimeError('reward model forward returned no logits.')
        return torch.as_tensor(logits).reshape(-1).tolist()

    def logps(self, features: List[Any], **kwargs: Any) -> Any:
        """Per-token response logps over ``features`` (per-sample refs or plain features)."""
        forward_kwargs = {'output_key': 'logps', **kwargs}
        if self.disable_adapter:
            forward_kwargs['disable_lora'] = True
        return self.model.forward_only(inputs=features, **forward_kwargs)

    def shutdown(self) -> None:
        shutdown = getattr(self.model, 'shutdown', None)
        if shutdown is not None:
            shutdown()


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

    Prefers the prompted ``Reward: <score>`` convention; falls back to the first bare number.
    Returns ``None`` (recorded as ``nan``) when nothing numeric is found, rather than guessing 0 --
    a silent 0 would rank an unparseable verdict as a genuinely bad candidate.
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
    # The candidate arrives as the trailing assistant turn of ``messages``; strip it so the prompt is
    # the conversation alone and the completion is supplied verbatim through the template.
    if messages and messages[-1].get('role') == 'assistant':
        messages = messages[:-1]
    return template.format(prompt=render_conversation(messages), completion=completion)


class GenerativeJudgeReward:
    """LLM-as-judge reward: prompt a generative model to score each candidate, parse the number out.

    When the judge's base is the sampler's own model it REUSES that engine (the reward LoRA is
    resident in the sampler's adapter list and selected per request via ``adapter_path``); otherwise
    it owns a standalone engine (``owns_sampler=True``), released by :meth:`shutdown`.
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
        #: An independent judge is its own placement (two such judges overlap); a sampler-reusing judge
        #: shares the rollout engine, so it declares no dedicated placement and serializes with rules.
        self.placement_key = id(sampler) if owns_sampler else None

    def __call__(self, rows: List[Dict[str, Any]], **kwargs) -> List[Optional[float]]:
        from swift.dev.builders import sampled_texts

        # A row is a trajectory: the trailing assistant turn is the candidate, the rest the
        # conversation that produced it (the same view every scorer kind sees).
        trajectories = [
            {
                'messages': [{
                    'role': 'user',
                    'content': judge_query(self.template, row.get('messages')[:-1], row['messages'][-1]['content'])
                }]
            } for row in rows
        ]
        sample_kwargs = {'adapter_path': self.adapter_path} if self.adapter_path is not None else {}
        texts = sampled_texts(self.sampler.sample(trajectories, self.params, **sample_kwargs))
        return [parse_judge_score(group[0] if group else '') for group in texts]

    def shutdown(self) -> None:
        if self.owns_sampler and self.sampler is not None:
            self.sampler.shutdown()


class ApiJudgeReward:
    """LLM-as-judge reward served by a remote OpenAI-compatible endpoint.

    Same judge template and score parser as :class:`GenerativeJudgeReward`, but generation happens on
    a remote server. Chosen when the channel's model id is an http(s) URL; the judge model name is
    taken from ``engine_args['model']``, as the 'client' sampler backend does.
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
        #: No GPU and no local engine -- this scorer already overlaps its own requests internally.
        self.placement_key = None

    def __call__(self, rows: List[Dict[str, Any]], **kwargs) -> List[Optional[float]]:
        from concurrent.futures import ThreadPoolExecutor

        queries = [judge_query(self.template, row.get('messages')[:-1], row['messages'][-1]['content'])
                   for row in rows]
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
            # Degrade to an unparseable (nan) score rather than ending a long run: a rate-limited
            # judge on one candidate must not abort the whole run.
            logger.warning(f'API judge request failed, scoring this candidate as unparseable: {exc}')
            return ''
        return completion.choices[0].message.content or ''

    def shutdown(self) -> None:
        """Nothing to release: the OpenAI client holds no GPU and closes its own connections."""


