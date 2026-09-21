"""On-policy GKD assembly: run_gkd orchestration (generalized knowledge distillation).

Peer of ``run_sft`` for GKD. GKD trains a student to match a frozen teacher on sequences the STUDENT
itself generates -- so the student learns to correct its own mistakes rather than only imitating a
fixed corpus. The key simplification that keeps this recipe free of weight-sync machinery: the student
generates from its OWN live weights via ``model.generate`` (twinkle stands a sampler over the resident
weights, no second copy, no ``CheckpointEngineManager``), so the behaviour policy is trivially the
current policy. run_grpo needs weight-sync only because its rollout runs in a SEPARATE vLLM process;
GKD's on-policy generation is in-process, so it does not.

Per step:
  1. take a batch of prompts, ``model.generate`` on-policy completions (student's current weights);
  2. rebuild the training features (prompt+response, response-only labels, next-token shifted);
  3. teacher ``forward_only(return_logits=True)`` -> full-vocab ``teacher_logits`` (a frozen separate
     model, or -- for a LoRA student whose teacher IS its base -- the adapter-disabled student);
  4. student ``forward_backward(teacher_logits=...)`` -> GKDLoss (β-JSD, optional top-k).

NOTE ON MODE: on-policy generation and the teacher forward both run on the driver's in-process model,
so this recipe targets mode='local'. ``lmbda`` selects student-generated versus dataset completions,
``sft_alpha`` adds supervised CE on dataset batches, and a separate local teacher can be offloaded
between scoring calls.
"""
from __future__ import annotations
import logging
import math
import random
from copy import deepcopy
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    from swift.dev.config import (
        CheckpointConfig,
        DatasetConfig,
        DistributedConfig,
        GenerationConfig,
        LoggingConfig,
        ModelConfig,
        RLHFConfig,
        RolloutConfig,
        TemplateConfig,
        TrainConfig,
        TunerConfig,
    )
    from swift.dev.model import TrainableModel

logger = logging.getLogger(__name__)


def run_gkd(
    model_config: ModelConfig,
    template_config: TemplateConfig,
    dataset_config: DatasetConfig,
    train_config: TrainConfig,
    distributed_config: DistributedConfig,
    checkpoint_config: CheckpointConfig,
    rlhf_config: RLHFConfig,
    tuner_config: Optional[TunerConfig] = None,
    generation_config: Optional[GenerationConfig] = None,
    logging_config: Optional[LoggingConfig] = None,
    rollout_config: Optional['RolloutConfig'] = None,
    *,
    output_dir: str = 'output',
    _save_final: bool = True,
) -> List[dict]:
    """Assemble and run on-policy GKD from atomic Configs. Returns the loss history.

    The build order is :class:`~swift.dev.recipe.assembly.TrainAssembly`'s, shared with every other
    recipe; GKD then adds the frozen teacher and an on-policy generation loop, and drives the stages one
    by one because it has no dataloader (it generates its own data). The student is the trained model;
    the teacher is a separate frozen model (rlhf_config.teacher_model) or, for a LoRA student whose
    teacher is its own base, the adapter-disabled student (rlhf_config._teacher_use_disable_adapter).
    """
    from swift.dev.loss import configure_rlhf_loss
    from swift.dev.optimizer import configure_optimizer, resolve_max_grad_norm
    from swift.dev.recipe.assembly import TrainAssembly

    if rlhf_config.rlhf_type != 'gkd':
        raise ValueError(f'run_gkd requires rlhf_type="gkd", got {rlhf_config.rlhf_type!r}.')
    assembly = TrainAssembly(
        'run_gkd',
        model_config,
        template_config,
        dataset_config,
        train_config,
        distributed_config,
        checkpoint_config,
        tuner_config,
        rlhf_config=rlhf_config,
        output_dir=output_dir,
        logging_config=logging_config)
    assembly.prepare()
    TrainAssembly.initialize_twinkle(distributed_config)

    assembly.build_template()
    assembly.build_model()
    configure_rlhf_loss(assembly.model, rlhf_config)
    # No dataloader to derive a step budget from -- the prompts are sampled, not iterated -- so
    # max_steps IS the budget.
    max_steps = train_config.max_steps or 1
    configure_optimizer(assembly.model, train_config, num_training_steps=max_steps)

    prompts, dataset_features, prompt_extras, dataset_messages = _gkd_rows_from_dataset(
        dataset_config, assembly.template)
    if rlhf_config.lmbda < 1.0 and any(feature is None for feature in dataset_features):
        raise ValueError('GKD with lmbda < 1 requires every dataset row to contain an assistant completion.')
    assembly.loop = GKDLoop(
        assembly.model,
        _build_teacher(assembly.model, rlhf_config, tuner_config),
        assembly.template,
        prompts,
        dataset_features=dataset_features,
        prompt_extras=prompt_extras,
        dataset_messages=dataset_messages,
        lmbda=rlhf_config.lmbda,
        sft_alpha=rlhf_config.sft_alpha,
        gkd_logits_topk=rlhf_config.gkd_logits_topk,
        max_steps=max_steps,
        batch_size=train_config.per_device_train_batch_size,
        gradient_accumulation_steps=assembly.ga,
        max_grad_norm=resolve_max_grad_norm(train_config),
        seed=train_config.seed,
        offload_teacher_model=rlhf_config.offload_teacher_model,
        teacher_tag_key=(rollout_config.teacher_tag_key if rollout_config is not None else 'dataset'),
        output_dir=output_dir,
        sampling_params=_gkd_sampling_params(rlhf_config, generation_config),
        logging_config=logging_config)
    history = assembly.loop.fit()
    if _save_final:
        assembly.save_final()
    return history


def _build_teacher(model: TrainableModel, rlhf_config: RLHFConfig, tuner_config: Optional[TunerConfig]) -> Any:
    """The teacher the student distils from: 'disable_lora', or a frozen separate model.

    Returns 'disable_lora' when the teacher is exactly the LoRA student's own base (no second model is
    loaded -- the loop runs ``forward_only(disable_lora=True)``); otherwise a frozen model built from
    ``teacher_model`` sharing the student's processor/template so both encode a batch identically.
    """
    if rlhf_config._teacher_use_disable_adapter:
        if tuner_config is None:
            raise ValueError('rlhf_config._teacher_use_disable_adapter=True requires a LoRA student (tuner_config), '
                             'since it distils from the adapter-disabled base of that same model.')
        return 'disable_lora'
    if rlhf_config.teacher_model_server is not None:
        return _RemoteGKDTeacher(rlhf_config.teacher_model_server, rlhf_config.gkd_logits_topk)
    if rlhf_config.teacher_model is None:
        raise ValueError('GKD needs a teacher: set RLHFConfig.teacher_model, or use a LoRA student with '
                         '_teacher_use_disable_adapter=True to distil from its own frozen base.')

    from swift.dev.builders import build_model
    from swift.dev.config import DistributedConfig, ModelConfig
    from swift.dev.recipe.assembly import configure_frozen_adapter

    teacher_cfg = ModelConfig(model=rlhf_config.teacher_model)
    teacher_cfg.model_type = rlhf_config.teacher_model_type
    teacher_cfg.model_revision = rlhf_config.teacher_model_revision
    teacher_dist = DistributedConfig(mode='local', deepspeed=rlhf_config.teacher_deepspeed)
    teacher = build_model(teacher_cfg, teacher_dist)
    return configure_frozen_adapter(
        teacher,
        model.template if hasattr(model, 'template') else None,
        rlhf_config.teacher_adapters,
        role='teacher')


class _RemoteGKDTeacher:
    """Sparse top-k teacher backed by one or more vLLM ``/infer/`` servers."""

    def __init__(self, server_spec: str, topk: Optional[int], *, client_factory=None):
        if topk is None or topk < 1:
            raise ValueError('GKD teacher_model_server requires gkd_logits_topk >= 1.')
        from swift.rlhf_trainers.gkd_helpers import parse_teacher_model_server

        self.configs = parse_teacher_model_server(server_spec)
        self.topk = topk
        if client_factory is None:
            from swift.rlhf_trainers.vllm_client import VLLMInferClient

            def client_factory(url):
                return VLLMInferClient(base_urls=[url])
        self.clients = [client_factory(config.url) for config in self.configs]

    def _routing(self, extras: List[dict], tag_key: str) -> Dict[int, List[int]]:
        if len(self.configs) == 1:
            return {0: list(range(len(extras)))}
        tag_to_teacher = {tag: index for index, config in enumerate(self.configs) for tag in config.tags}
        routing = {index: [] for index in range(len(self.configs))}
        for sample_index, extra in enumerate(extras):
            value = extra.get(tag_key)
            tag = str(value) if value is not None else None
            teacher_index = tag_to_teacher.get(tag)
            if teacher_index is None:
                raise ValueError(f'GKD sample[{sample_index}] tag {tag!r} from {tag_key!r} matches no teacher.')
            routing[teacher_index].append(sample_index)
        return routing

    def score(self, features: List[dict], requests: List[Any], extras: List[dict], tag_key: str) -> Dict[str, Any]:
        import torch

        from swift.infer_engine import RequestConfig
        from swift.rlhf_trainers.utils import assemble_teacher_topk_logprobs, parse_prompt_logprobs

        parsed: List[Any] = [None] * len(requests)
        request_config = RequestConfig(prompt_logprobs=self.topk, max_tokens=1, temperature=0.0)
        for teacher_index, sample_indices in self._routing(extras, tag_key).items():
            if not sample_indices:
                continue
            subset = [requests[index] for index in sample_indices]
            responses = self.clients[teacher_index].infer(subset, request_config=request_config, use_tqdm=False)
            if len(responses) != len(subset):
                raise RuntimeError(f'Teacher server returned {len(responses)} responses for {len(subset)} requests.')
            for sample_index, response in zip(sample_indices, responses):
                parsed[sample_index] = parse_prompt_logprobs(response, topk=self.topk)
        if any(item is None for item in parsed):
            raise RuntimeError('Teacher server routing left one or more GKD samples unscored.')
        seq_len = max(len(feature['input_ids']) for feature in features)
        logprobs, indices = assemble_teacher_topk_logprobs(
            parsed,
            batch_size=len(features),
            seq_len=seq_len,
            cu_seqlens=None,
            topk=self.topk,
            device=torch.device('cpu'))
        return {'teacher_topk_logprobs': logprobs, 'teacher_topk_indices': indices}


def _gkd_sampling_params(rlhf_config: RLHFConfig, generation_config: Optional[GenerationConfig]) -> Dict[str, Any]:
    """SamplingParams dict for on-policy generation (max_completion_length + temperature)."""
    params: Dict[str, Any] = {
        'max_tokens': rlhf_config.max_completion_length,
        'temperature': rlhf_config.temperature,
    }
    if generation_config is not None:
        if generation_config.top_p is not None:
            params['top_p'] = generation_config.top_p
        if generation_config.top_k is not None:
            params['top_k'] = generation_config.top_k
    return params


class GKDLoop:
    """On-policy GKD loop: student generates, teacher scores, student distils toward the teacher.

    Peer of :class:`SFTLoop`. Uses the same one-micro-batch-per-forward_backward GA shape (so twinkle's
    grad-sync gate lines up), replacing the SFT forward with: generate -> teacher forward -> student
    forward_backward(teacher_logits=...). ``lmbda`` is the per-step probability of using on-policy
    generations vs. the dataset's own reference completions.
    """

    def __init__(
        self,
        model: TrainableModel,
        teacher: Any,
        template: Any,
        prompts: List[List[dict]],
        *,
        dataset_features: Optional[List[Optional[dict]]] = None,
        prompt_extras: Optional[List[dict]] = None,
        dataset_messages: Optional[List[List[dict]]] = None,
        lmbda: float = 0.5,
        sft_alpha: float = 0.0,
        gkd_logits_topk: Optional[int] = None,
        max_steps: int = 1,
        batch_size: int = 1,
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float = 1.0,
        seed: int = 42,
        offload_teacher_model: bool = False,
        teacher_tag_key: str = 'dataset',
        logging_steps: int = 1,
        output_dir: str = 'output',
        sampling_params: Optional[dict] = None,
        logging_config: Optional['LoggingConfig'] = None,
    ):
        self.model = model
        self.teacher = teacher
        self.template = template
        self.prompts = prompts
        self.dataset_features = dataset_features or [None] * len(prompts)
        self.prompt_extras = prompt_extras or [{} for _ in prompts]
        self.dataset_messages = dataset_messages or list(prompts)
        if not (len(self.dataset_features) == len(self.prompt_extras) == len(self.dataset_messages) == len(prompts)):
            raise ValueError('GKD features, extras, dataset messages, and prompts must have equal lengths.')
        self.lmbda = lmbda
        self.sft_alpha = sft_alpha
        self.gkd_logits_topk = gkd_logits_topk
        self.max_steps = max_steps
        self.batch_size = max(1, batch_size)
        self.gradient_accumulation_steps = max(1, gradient_accumulation_steps)
        self.max_grad_norm = max_grad_norm
        self.seed = seed
        self.offload_teacher_model = offload_teacher_model
        self.teacher_tag_key = teacher_tag_key
        if offload_teacher_model:
            if teacher in (None, 'disable_lora') or isinstance(teacher, _RemoteGKDTeacher):
                raise ValueError('offload_teacher_model requires a distinct local teacher model.')
            teacher.offload_to_cpu()
        self.logging_steps = logging_config.logging_steps if logging_config is not None else logging_steps
        self.logging_config = logging_config
        from swift.dev.recipe.tracking import RunTracker
        self.tracker = RunTracker(logging_config, output_dir)
        self.output_dir = output_dir
        self.sampling_params = sampling_params
        self.global_step = 0
        self.micro_step = 0
        self.history: list = []

    def _is_grad_sync_boundary(self) -> bool:
        ga = self.gradient_accumulation_steps
        return ga == 1 or ((self.micro_step - 1) % ga == 0 and self.micro_step > 1)

    def _batch_indices(self, step: int) -> List[int]:
        n = len(self.prompts)
        if n == 0:
            raise ValueError('GKD requires at least one prompt.')
        start = (step * self.batch_size) % n
        return [(start + i) % n for i in range(self.batch_size)]

    def _prompt_batch(self, step: int) -> List[List[dict]]:
        """The prompts for one step: a rolling window over the prompt list (wraps around)."""
        return [self.prompts[index] for index in self._batch_indices(step)]

    def _dataset_batch(self, step: int) -> List[dict]:
        features = [self.dataset_features[index] for index in self._batch_indices(step)]
        if any(feature is None for feature in features):
            raise ValueError('The selected GKD dataset batch contains a row without an assistant completion.')
        return [deepcopy(feature) for feature in features]

    def _extras_batch(self, step: int) -> List[dict]:
        return [deepcopy(self.prompt_extras[index]) for index in self._batch_indices(step)]

    def _messages_batch(self, step: int, *, dataset: bool) -> List[List[dict]]:
        source = self.dataset_messages if dataset else self.prompts
        return [deepcopy(source[index]) for index in self._batch_indices(step)]

    def _uses_student_generation(self, step: int) -> bool:
        return random.Random(self.seed + step).random() <= self.lmbda

    def _generate_features(self, prompts: List[List[dict]]) -> List[dict]:
        """On-policy generate from the student's live weights and rebuild training features.

        Rebuilds each feature from the prompt+response token ids with response-only, next-token
        shifted labels (identical convention to run_grpo's SamplerRollout), so the teacher and student
        forwards both see the same tokens and the JSD is computed over the response positions only.
        """
        from twinkle.data_format import SamplingParams, Trajectory

        from swift.dev.rollout import SHIFTED_KEY

        params = SamplingParams(**dict(self.sampling_params or {}))
        trajectories = [Trajectory(messages=list(messages)) for messages in prompts]
        responses = self.model.generate(trajectories, sampling_params=params)

        features: List[dict] = []
        for response in responses:
            prompt_tokens = list(response.prompt_token_ids or [])
            if not prompt_tokens:
                raise RuntimeError('model.generate returned no prompt_token_ids; a template must be set so the '
                                   'prompt is encoded before generation.')
            for seq in response.sequences:
                response_tokens = list(seq.tokens or [])
                if not response_tokens:
                    continue
                aligned = [-100] * len(prompt_tokens) + response_tokens
                labels = list(aligned[1:]) + [-100]
                features.append({'input_ids': prompt_tokens + response_tokens, 'labels': labels, SHIFTED_KEY: True})
        if not features:
            raise RuntimeError('GKD step produced no non-empty completions to distil on.')
        return features

    def _teacher_logits(self, features: List[dict]) -> Any:
        """Full-vocab teacher logits for a local teacher."""
        if self.teacher == 'disable_lora':
            outputs = self.model.forward_only(inputs=features, disable_lora=True, return_logits=True)
            return outputs['logits']
        if self.offload_teacher_model:
            self.teacher.reload_to_gpu()
        try:
            outputs = self.teacher.forward_only(inputs=features, return_logits=True)
            return outputs['logits']
        finally:
            if self.offload_teacher_model:
                self.teacher.offload_to_cpu()

    def _teacher_requests(
        self, features: List[dict], messages_batch: List[List[dict]], extras: List[dict]
    ) -> List[Any]:
        from swift.infer_engine.protocol import RolloutInferRequest
        from swift.rlhf_trainers.utils import get_response_prefix_ids, replace_assistant_response_with_ids

        requests = []
        prefix_ids = get_response_prefix_ids(self.template) if self.template is not None else None
        request_fields = ('images', 'audios', 'videos', 'tools', 'objects', 'chat_template_kwargs')
        for feature, messages, extra in zip(features, messages_batch, extras):
            labels = feature.get('labels')
            if labels is None:
                raise ValueError('GKD teacher-server features require labels to recover exact response token ids.')
            response_ids = [int(label) for label in labels if int(label) != -100]
            if not response_ids:
                raise ValueError('GKD teacher-server features contain no response tokens to score.')
            if not messages or messages[-1].get('role') != 'assistant':
                messages.append({'role': 'assistant', 'content': None})
            messages = replace_assistant_response_with_ids(
                messages, response_ids, non_thinking_prefix_ids=prefix_ids)
            kwargs = {name: deepcopy(extra[name]) for name in request_fields if extra.get(name) is not None}
            requests.append(RolloutInferRequest(messages=messages, **kwargs))
        return requests

    def _teacher_kwargs(
        self, features: List[dict], messages_batch: List[List[dict]], extras: List[dict]
    ) -> Dict[str, Any]:
        if isinstance(self.teacher, _RemoteGKDTeacher):
            requests = self._teacher_requests(features, messages_batch, extras)
            return self.teacher.score(features, requests, extras, self.teacher_tag_key)
        return {'teacher_logits': self._teacher_logits(features)}

    def fit(self) -> list:
        """Run max_steps GKD steps. Each step: generate -> teacher forward -> student forward_backward."""
        ga = self.gradient_accumulation_steps
        step = 0
        try:
            while self.global_step < self.max_steps:
                self.micro_step += 1
                use_student = self._uses_student_generation(step)
                features = (self._generate_features(self._prompt_batch(step)) if use_student else
                            self._dataset_batch(step))
                messages_batch = self._messages_batch(step, dataset=not use_student)
                extras = self._extras_batch(step)
                step += 1
                teacher_kwargs = self._teacher_kwargs(features, messages_batch, extras)
                self.model.forward_backward(
                    inputs=features,
                    gradient_accumulation_steps=ga,
                    **teacher_kwargs,
                    topk=self.gkd_logits_topk,
                    apply_sft_loss=bool(self.sft_alpha > 0 and not use_student))
                is_boundary = self._is_grad_sync_boundary()
                self.model.clip_grad_and_step(max_grad_norm=self.max_grad_norm, gradient_accumulation_steps=ga)
                if is_boundary:
                    self._record_step()
            return self.history
        finally:
            self.tracker.close()

    def _record_step(self) -> None:
        self.global_step += 1
        metrics = self.model.calculate_metric(is_training=True)
        loss = float(metrics['loss']) if metrics.get('loss') is not None else float('nan')
        record = {'step': self.global_step, 'loss': loss}
        if metrics.get('grad_norm') is not None:
            record['grad_norm'] = float(metrics['grad_norm'])
        record = self.tracker.log(record, self.global_step)
        self.history.append(record)
        should_log = (self.tracker.should_log(self.global_step) if self.logging_config is not None else
                      bool(self.logging_steps and self.global_step % self.logging_steps == 0))
        if should_log:
            logger.info(f'step {self.global_step}  loss={record["loss"]:.4f}')

    def save(self, name: str = 'checkpoint-final') -> str:
        """Persist the student policy + training state via twinkle's native save."""
        return self.model.save(name, output_dir=self.output_dir, save_optimizer=True)


def _gkd_rows_from_dataset(
    dataset_config: DatasetConfig,
    template: Any,
) -> Tuple[List[List[dict]], List[Optional[dict]], List[dict], List[List[dict]]]:
    """Load prompts, encoded completions, routing extras, and exact dataset messages."""
    from swift.dev.recipe.run_infer import _load_prompt_rows

    rows = _load_prompt_rows(dataset_config, None, split_dataset_ratio=0.0)
    prompts: List[List[dict]] = []
    features: List[Optional[dict]] = []
    extras: List[dict] = []
    dataset_messages: List[List[dict]] = []
    for row in rows:
        messages = deepcopy(row.get('messages') or [])
        if not messages:
            continue
        has_completion = messages[-1].get('role') == 'assistant'
        prompts.append(deepcopy(messages[:-1] if has_completion else messages))
        dataset_messages.append(messages)
        features.append(template.encode(row) if has_completion else None)
        extras.append({key: deepcopy(value) for key, value in row.items() if key != 'messages'})
    if not prompts:
        raise ValueError('run_gkd found no prompt messages in the dataset rows.')
    return prompts, features, extras, dataset_messages
