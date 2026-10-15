from __future__ import annotations
import copy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, NamedTuple, Optional, Tuple

from swift.dev.recipe._scoring import (
    batched_response_scores,
    copy_inputs,
    encode_privileged_view,
    split_mini_batches,
)
from swift.dev.recipe.rl_auxiliary import AuxiliarySpec, FrozenAuxiliary, build_auxiliary
from swift.dev.utils import get_logger

if TYPE_CHECKING:
    from swift.dev.config import RLConfig

logger = get_logger()


class DistillRow(NamedTuple):
    """One distillation training row: the student feature plus the context a teacher view needs.

    Uniform across both ``lmbda`` sources -- an on-policy rollout sample and an off-policy dataset
    completion both reduce to ``(feature, messages, extra)`` -- so the loop's mini-batch split and the
    teacher scoring never branch on where a row came from.

    Fields:
        feature: the student's trainable feature (``input_ids`` + next-token-shifted, response-only
            ``labels``); fed verbatim to ``forward_backward`` and, for GKD, to the teacher forward.
        messages: the conversation the teacher's privileged view is rebuilt from -- prompt-only for an
            on-policy row, the full dataset conversation for an off-policy one. GKD ignores it (its
            teacher sees ``feature`` directly); OPSD replaces its user turn with ``teacher_prompt``.
        extra: dataset passthrough columns (``teacher_prompt``, multimodal inputs, routing tags).
    """
    feature: Dict[str, Any]
    messages: List[dict]
    extra: Dict[str, Any]


@dataclass
class DistillBatch:
    """One distillation round's rows plus the flag recording whether they came from the student.

    The distillation counterpart of :class:`TrainingBatch`: :func:`split_mini_batches` slices it by ``len`` +
    ``rows[i:j]`` exactly the same way, so the shared mini-batch planner handles both value types without
    branching. ``use_student`` survives the slice because each mini-batch still needs it -- an off-policy
    round (``use_student=False``, the dataset completions a ``lmbda < 1`` coin-flip picked) turns on the SFT
    cross-entropy term in the forward kwargs, and that choice is per-round, not per-row. ``rows`` are
    :class:`DistillRow` (a feature + its messages/extra).
    """
    rows: List[DistillRow]
    use_student: bool = True

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, key: slice) -> 'DistillBatch':
        if not isinstance(key, slice):
            raise TypeError(f'DistillBatch supports slice indexing only, got {type(key).__name__}.')
        return DistillBatch(rows=self.rows[key], use_student=self.use_student)


def build_frozen_teacher(rl_config: Any,
                         template: Any,
                         model_id: str,
                         *,
                         distributed_config: Any,
                         remote_group: str,
                         adapters: Optional[List[str]] = None,
                         role: str = 'teacher',
                         offload: bool = False,
                         instance_id: Optional[str] = None) -> FrozenAuxiliary:
    """Build one frozen twinkle teacher as a Ray actor on its own DeviceGroup (a ``forward_only`` scorer).

    A distillation teacher is always a frozen model actor -- never an HTTP server (no-server / all-Ray
    premise, RL_PLAN §2.H/§3.3). This is a thin teacher-shaped front over the shared
    :func:`~swift.dev.recipe.rl_auxiliary.build_auxiliary`: it packs ``rl_config``'s teacher knobs into an
    :class:`~swift.dev.recipe.rl_auxiliary.AuxiliarySpec` and delegates the build, so the teacher inherits
    the run's ``backend`` and ``mode='ray'`` placement exactly like the reference does. A teacher is a
    DISTINCT model (often larger than the student), so the spec is built from a bare ``ModelConfig(model_id)``
    (``base_model_config=False``) rather than a copy of the policy config, and ``teacher_model_type`` /
    ``teacher_model_revision`` / ``teacher_deepspeed`` are read off ``rl_config`` so it may differ in
    architecture from the student. Its DeviceGroup rank count comes from ``teacher_parallel_spec`` (the SAME
    spec ``plan_rl_device_groups`` sized the ``remote_group`` with), so the plan and the build cannot drift.
    ``offload`` (GKD's full-vocab memory hook) keeps the teacher on CPU between forwards. ``instance_id``
    disambiguates the K teachers MOPD builds into the ONE shared ``'teacher'`` group -- without it their Ray
    actor names (group+class+caller+rank) collide; a single-teacher GKD/OPSD build leaves it None.
    """
    spec = AuxiliarySpec(
        role=role,
        model_id=model_id,
        name=remote_group,
        parallel_spec=rl_config.teacher_parallel_spec,
        adapters=tuple(adapters or ()),
        model_type=rl_config.teacher_model_type,
        model_revision=rl_config.teacher_model_revision,
        deepspeed=rl_config.teacher_deepspeed,
        offload=offload,
        base_model_config=False,
        instance_id=instance_id)
    return build_auxiliary(spec, template=template, student=None, distributed_config=distributed_config)


def response_ids_from_feature(feature: Dict[str, Any]) -> List[int]:
    """The response token ids of an encoded feature, recovered from its (next-token shifted) labels.

    Labels hold the response token ids at trainable positions and ``-100`` elsewhere, so filtering
    ``-100`` yields exactly the sampled response -- the tokens the teacher must re-score.
    """
    labels = feature.get('labels')
    if labels is None:
        raise ValueError('OPSD/MOPD teacher scoring requires a labelled feature to recover the response token ids.')
    response_ids = [int(label) for label in labels if int(label) != -100]
    if not response_ids:
        raise ValueError('OPSD/MOPD feature contains no response tokens to score.')
    return response_ids


def teacher_view_messages(prompt_messages: List[dict], teacher_prompt: Any, response_ids: List[int]) -> List[dict]:
    """Build the privileged teacher message list: ``teacher_prompt`` in the user turn + shared response.

    Replaces the FIRST user turn's content with the privileged ``teacher_prompt`` (the dev-layer OPSD
    convention; for a single-turn sample first == last) and appends the assistant turn carrying the
    student's response token ids verbatim, so the teacher scores exactly the same response tokens.
    """
    messages = [dict(message) for message in prompt_messages]
    if messages and messages[-1].get('role') == 'assistant':
        messages = messages[:-1]
    for message in messages:
        if message.get('role') == 'user':
            message['content'] = teacher_prompt
            break
    else:
        raise ValueError('teacher_prompt requires a user turn in the prompt to carry the privileged context.')
    messages.append({'role': 'assistant', 'content': list(response_ids)})
    return messages


def build_privileged_teacher_feature(
    template: Any,
    prompt_messages: List[dict],
    teacher_prompt: Any,
    response_ids: List[int],
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Encode the OPSD privileged teacher view (``teacher_prompt`` in the user turn + the shared response).

    Builds the privileged message list (``teacher_view_messages``) then delegates the encode-and-align-check
    tail to :func:`~swift.dev.recipe._scoring.encode_privileged_view`, which GRPO's RLSD/SDAR teacher view
    shares -- the two differ only in how the messages are assembled, not in the encoding contract.
    """
    messages = teacher_view_messages(prompt_messages, teacher_prompt, response_ids)
    return encode_privileged_view(template, messages, response_ids, extra)


def distill_sampling_params(rl_config: Any, generation_config: Any) -> Dict[str, Any]:
    """SamplingParams dict for the on-policy distillation rollout (shared by GKD/OPSD/MOPD).

    The rollout uses ``sampling_temperature``, NOT the distillation ``distill_temperature``: those are
    separate roles. ``sampling_temperature=None`` falls back to ``distill_temperature`` so pre-split runs
    are byte-identical; setting it decouples how hot the student samples from how softly the divergence
    scores.
    """
    sampling_temperature = rl_config.sampling_temperature
    if sampling_temperature is None:
        sampling_temperature = rl_config.distill_temperature
    params: Dict[str, Any] = {
        'max_tokens': rl_config.max_completion_length,
        'temperature': sampling_temperature,
    }
    if generation_config is not None:
        if generation_config.top_p is not None:
            params['top_p'] = generation_config.top_p
        if generation_config.top_k is not None:
            params['top_k'] = generation_config.top_k
    return params


def distill_rows_from_dataset(
    dataset_config: Any,
    template: Any,
) -> Tuple[List[List[dict]], List[dict], List[DistillRow]]:
    """Load rollout prompts, their passthrough extras, and the off-policy dataset completion rows.

    Shared by GKD/OPSD/MOPD. ``prompts`` / ``prompt_extras`` are parallel (one entry per dataset row) and
    drive the on-policy rollout; ``dataset_rows`` holds only the rows that carry an assistant completion,
    encoded as training features, for the off-policy rounds ``lmbda < 1`` selects.
    """
    from swift.dev.builders import load_prompt_only_rows

    rows = load_prompt_only_rows(dataset_config, None, split_dataset_ratio=0.0)
    prompts: List[List[dict]] = []
    prompt_extras: List[dict] = []
    dataset_rows: List[DistillRow] = []
    for row in rows:
        messages = copy.deepcopy(row.get('messages') or [])
        if not messages:
            continue
        extra = {key: copy.deepcopy(value) for key, value in row.items() if key != 'messages'}
        has_completion = messages[-1].get('role') == 'assistant'
        prompts.append(copy.deepcopy(messages[:-1] if has_completion else messages))
        prompt_extras.append(extra)
        if has_completion:
            dataset_rows.append(DistillRow(template.encode(row), messages, extra))
    if not prompts:
        raise ValueError('the distillation dataset provided no prompt messages.')
    return prompts, prompt_extras, dataset_rows


class DistillAssembler:
    """The teacher-distillation score + assemble + encode component for the flat distill recipes.

    One component, THREE teacher-signal forms selected by the ``signal`` constructor parameter (not
    subclasses):

    * ``'logits'`` (GKD) -- the shared prompt+response full-vocab ``teacher_logits``, scored lazily one
      mini-batch at a time (a full-vocab ``[rows, seq, vocab]`` tensor over a whole round would not fit),
      consumed by the JSD loss with the ``topk`` / ``apply_sft_loss`` knobs;
    * ``'logps'`` (OPSD) -- RESPONSE-ONLY ``teacher_logps`` under each row's privileged view
      (``teacher_prompt`` in the user turn + the shared response tokens); the teacher and student prompts
      differ in length, so GKD's full-sequence form must not be used;
    * ``'multi_logps'`` (MOPD) -- K response-only ``teacher_logps`` channels (one per teacher in the
      :class:`~swift.dev.recipe.rl_auxiliary.AuxiliaryBundle`), overlapped across disjoint DeviceGroups via
      ``overlap_by_placement``, plus the bundle's mixture weights.

    Teacher discovery: every teacher entering this component is an
    :class:`~swift.dev.recipe.rl_auxiliary.AuxiliaryModel` built by the SHARED primitives --
    :func:`~swift.dev.recipe.rl_auxiliary.build_auxiliary` + :class:`AuxiliarySpec` (see
    :func:`build_frozen_teacher` for the teacher-shaped front), which resolves to
    :class:`~swift.dev.recipe.rl_auxiliary.PolicyAuxiliary` (adapter-disabled base / dynamic self),
    :class:`~swift.dev.recipe.rl_auxiliary.FrozenAuxiliary` (separate Ray-actor model) or an
    :class:`~swift.dev.recipe.rl_auxiliary.AuxiliaryBundle` (MOPD's K). There is no HTTP teacher server and
    no per-algorithm teacher class.

    The teacher forward runs in :meth:`forward_kwargs` (per mini-batch, ``train_batch_size`` rows) so its
    output is never materialised for a whole round. The component also owns the per-sample stream seams
    (:meth:`assemble` / :meth:`assembly_ready`) for a purely on-policy run (``lmbda == 1.0``).
    """

    def __init__(self,
                 model: Any,
                 rollout: Any,
                 *,
                 teacher: Any,
                 template: Any = None,
                 prompts: Optional[List[Any]] = None,
                 signal: str = 'logits',
                 gkd_logits_topk: Optional[int] = None,
                 sft_alpha: float = 0.0,
                 train_batch_size: int = 1,
                 gradient_accumulation_steps: int = 1):
        if signal not in ('logits', 'logps', 'multi_logps'):
            raise ValueError(f"distillation signal must be one of 'logits'/'logps'/'multi_logps', got {signal!r}.")
        self.model = model
        self.rollout = rollout
        #: The frozen teacher -- never ``None`` (the no-separate-teacher case is a ``PolicyAuxiliary``, the
        #: K-teacher case an ``AuxiliaryBundle``), so scoring is one uniform ``forward_only`` call.
        self.teacher = teacher
        self.template = template
        #: The rollout prompts, for :meth:`assemble`'s prompt lookup when a streamed sample carries no
        #: ``messages`` of its own.
        self.prompts = prompts
        self.signal = signal
        #: GKD knobs: score the JSD against only the teacher's top-k logits (None = full vocab); the weight
        #: on the supervised CE term GKD adds to dataset-sourced (off-policy) rounds.
        self.gkd_logits_topk = gkd_logits_topk
        self.sft_alpha = sft_alpha
        self.train_batch_size = max(1, train_batch_size)
        self.gradient_accumulation_steps = max(1, gradient_accumulation_steps)

    # --- the per-sample stream seams (a purely on-policy run, lmbda == 1.0) -------------------------------

    def assemble(self, records: List[Any]) -> DistillBatch:
        """Lift a streamed pull's samples into a :class:`DistillBatch` (always student-generated).

        The stream runs only under ``lmbda == 1.0``, so every row comes from the student: each trajectory's
        ``input_feature`` is the student row, and its conversation (``sample.messages``, else the prompt it
        was admitted for) plus ``extra`` ride along for the teacher view.
        """
        rows = [
            DistillRow(sample.input_feature,
                       sample.messages if sample.messages is not None else self.prompts[int(sample.prompt_id)],
                       sample.extra or {}) for sample in (record.sample for record in records)
        ]
        return DistillBatch(rows, True)

    def assembly_ready(self, buffer: List[Any]) -> Optional[List[Any]]:
        """The per-sample streaming pull rule: the buffer prefix, an exact multiple of ``train_batch_size * ga``.

        Each distill row is scored on its own (no group to wait for -- the same rule PPO uses), so the
        completed prefix up to the largest whole-unit multiple is pulled; the remainder stays buffered.
        """
        need_rows = self.train_batch_size * self.gradient_accumulation_steps
        if len(buffer) < need_rows:
            return None
        return buffer[:(len(buffer) // need_rows) * need_rows]

    # --- the train half -----------------------------------------------------------------------------------

    def plan_mini_batches(self, batch: DistillBatch) -> List[DistillBatch]:
        """Split one round's rows into full ``train_batch_size`` mini-batches, warning on a dropped tail.

        :func:`~swift.dev.recipe._scoring.split_mini_batches` slices ``DistillBatch`` by ``len`` +
        ``rows[i:j]``, so each mini-batch keeps the per-round ``use_student`` flag. Raises when the batch
        cannot fill even one mini-batch, so a caller never silently runs a no-op training round.
        """
        mini_batches = split_mini_batches(batch, self.train_batch_size)
        if not mini_batches:
            raise RuntimeError(
                f'rollout produced {len(batch)} distill rows but train_batch_size={self.train_batch_size} '
                '(per_device_train_batch_size * dp_size), so no full mini-batch fits and no optimizer step '
                'is possible. Enlarge the prompt dataset or num_generations, or lower '
                'per_device_train_batch_size / the DP world size.')
        dropped = len(batch) - len(mini_batches) * self.train_batch_size
        if dropped:
            logger.warning('Dropping %d of %d distill rows that do not fill a train_batch_size=%d mini-batch.',
                           dropped, len(batch), self.train_batch_size)
        return mini_batches

    def forward_kwargs(self, mini_batch: DistillBatch) -> Dict[str, Any]:
        """One mini-batch's ``forward_backward`` kwargs: student inputs + the lazily-scored teacher signal.

        The teacher forward runs HERE (per mini-batch, ``train_batch_size`` rows) so its full-vocab output is
        never materialised for a whole round. The signal form is the ``signal`` constructor parameter, so
        this shared assembly never branches on the algorithm beyond that switch.
        """
        teacher_kwargs = self._teacher_kwargs(mini_batch.rows, mini_batch.use_student)
        return {
            'inputs': copy_inputs([row.feature for row in mini_batch.rows]),
            'gradient_accumulation_steps': self.gradient_accumulation_steps,
            **self._distill_kwargs(teacher_kwargs, mini_batch.use_student),
        }

    # --- the three teacher-signal forms --------------------------------------------------------------------

    def _response_logps(self, scorer: Any, features: List[dict]) -> List[List[float]]:
        """Score ``features`` with chunked ``forward_only`` -> one response-token logp list per feature.

        A thin wrapper over :func:`~swift.dev.recipe._scoring.batched_response_scores` (one chunked
        ``forward_only`` per ``train_batch_size`` block, DP>1-correct, right-padding aware) -- the shared
        "score a frozen model over the student's response tokens" primitive OPSD/MOPD use.
        """
        return batched_response_scores(
            scorer, features, train_batch_size=self.train_batch_size, output_key='logps')

    def _teacher_kwargs(self, rows: List[DistillRow], use_student: bool) -> Dict[str, Any]:
        """The teacher signal for one mini-batch, in the form ``signal`` selected."""
        if self.signal == 'logits':
            # GKD: the shared prompt+response full-vocab logits.
            features = [row.feature for row in rows]
            return {'teacher_logits': self.teacher.forward_only(inputs=copy_inputs(features), return_logits=True)['logits']}
        if self.signal == 'logps':
            # OPSD: response-only logps under each row's privileged view.
            return {'teacher_logps': self._response_logps(self.teacher, [self._privileged_feature(row) for row in rows])}
        # MOPD: K response-only channels, overlapped across disjoint DeviceGroups, plus the mixture weights.
        features = [row.feature for row in rows]
        from swift.dev.recipe._scoring import overlap_by_placement
        tasks = [(teacher.placement_key, lambda teacher=teacher: self._response_logps(teacher, features))
                 for teacher in self.teacher.auxiliaries]
        return {'teacher_logps': overlap_by_placement(tasks), 'teacher_weights': self.teacher.weights}

    def _privileged_feature(self, row: DistillRow) -> dict:
        """Encode this row's privileged teacher view (``teacher_prompt`` + the shared response tokens)."""
        teacher_prompt = (row.extra or {}).get('teacher_prompt')
        if not teacher_prompt:
            raise ValueError('OPSD requires a per-sample `teacher_prompt` carrying the privileged context; '
                             'a row in this batch has none.')
        response_ids = response_ids_from_feature(row.feature)
        return build_privileged_teacher_feature(self.template, row.messages, teacher_prompt, response_ids, row.extra)

    def _distill_kwargs(self, teacher_kwargs: Dict[str, Any], use_student: bool) -> Dict[str, Any]:
        """The loss kwargs consuming ``teacher_kwargs`` (GKD adds its top-k / supervised-CE knobs)."""
        if self.signal == 'logits':
            return {
                **teacher_kwargs,
                'topk': self.gkd_logits_topk,
                'apply_sft_loss': bool(self.sft_alpha > 0 and not use_student),
            }
        return dict(teacher_kwargs)
