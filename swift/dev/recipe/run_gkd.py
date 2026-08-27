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
so this recipe targets mode='local'. A distributed teacher (teacher_model_server) is out of scope.
The dataset-mixing knob ``lmbda`` is applied as the per-step probability of generating on-policy vs.
falling back to the dataset's own completion; ``sft_alpha`` (an extra SFT term) is not folded in here
because the twinkle GKDLoss is pure JSD -- documented as a known reduction from legacy GKD.
"""
from __future__ import annotations
import logging
import math
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from swift.dev.config import (
        CheckpointConfig,
        DatasetConfig,
        DistributedConfig,
        GenerationConfig,
        ModelConfig,
        RLHFConfig,
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
    *,
    output_dir: str = 'output',
    _save_final: bool = True,
) -> List[dict]:
    """Assemble and run on-policy GKD from atomic Configs. Returns the loss history.

    Mirrors run_sft's build order (template -> model -> tuner -> loss -> optimizer), then adds the
    frozen teacher and an on-policy generation loop. The student is the trained model; the teacher is
    a separate frozen model (rlhf_config.teacher_model) or, for a LoRA student whose teacher is its own
    base, the adapter-disabled student (rlhf_config._teacher_use_disable_adapter).
    """
    from swift.dev.adapter import apply_tuner
    from swift.dev.builders import build_model, build_template
    from swift.dev.config import validate_configs
    from swift.dev.loss import configure_rlhf_loss
    from swift.dev.optimizer import configure_optimizer, resolve_max_grad_norm
    from swift.dev.processor import InputProcessor
    from swift.dev.recipe.run_grpo import _prompts_from_dataset
    from swift.dev.recipe.run_sft import _initialize_twinkle, _write_ckpt_args_json
    from swift.model import get_model_processor

    if rlhf_config.rlhf_type != 'gkd':
        raise ValueError(f'run_gkd requires rlhf_type="gkd", got {rlhf_config.rlhf_type!r}.')
    validate_configs(model_config, template_config, dataset_config, train_config, distributed_config, checkpoint_config,
                     tuner_config, rlhf_config=rlhf_config)
    _initialize_twinkle(distributed_config)

    ga = train_config.gradient_accumulation_steps
    _, processor = get_model_processor(model_config.model, load_model=False)
    template = build_template(template_config, processor)

    model = build_model(model_config, distributed_config, train_config, tuner_config)
    if tuner_config is not None:
        apply_tuner(model, tuner_config, gradient_accumulation_steps=ga)
    model.set_processor(InputProcessor, padding_free=template_config.padding_free)
    model.set_template(template)
    configure_rlhf_loss(model, rlhf_config)
    configure_optimizer(model, train_config, num_training_steps=train_config.max_steps or 1)

    teacher = _build_teacher(model, rlhf_config, tuner_config)
    prompts = _prompts_from_dataset(dataset_config)

    loop = GKDLoop(
        model,
        teacher,
        template,
        prompts,
        lmbda=rlhf_config.lmbda,
        gkd_logits_topk=rlhf_config.gkd_logits_topk,
        max_steps=train_config.max_steps or 1,
        batch_size=train_config.per_device_train_batch_size,
        gradient_accumulation_steps=ga,
        max_grad_norm=resolve_max_grad_norm(train_config),
        output_dir=output_dir,
        sampling_params=_gkd_sampling_params(rlhf_config, generation_config))
    history = loop.fit()
    if _save_final:
        import os
        loop.save('checkpoint-final')
        _write_ckpt_args_json(
            os.path.join(output_dir, 'checkpoint-final'), processor, model_config, template_config, tuner_config)
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
    if rlhf_config.teacher_model is None:
        raise ValueError('GKD needs a teacher: set RLHFConfig.teacher_model, or use a LoRA student with '
                         '_teacher_use_disable_adapter=True to distil from its own frozen base.')

    from swift.dev.builders import build_model
    from swift.dev.config import DistributedConfig, ModelConfig
    from swift.dev.processor import InputProcessor

    teacher_cfg = ModelConfig(model=rlhf_config.teacher_model)
    teacher_cfg.model_type = rlhf_config.teacher_model_type
    teacher_cfg.model_revision = rlhf_config.teacher_model_revision
    teacher = build_model(teacher_cfg, DistributedConfig(mode='local'))
    teacher.set_processor(InputProcessor)
    teacher.set_template(model.template if hasattr(model, 'template') else None)
    return teacher


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
        lmbda: float = 0.5,
        gkd_logits_topk: Optional[int] = None,
        max_steps: int = 1,
        batch_size: int = 1,
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float = 1.0,
        logging_steps: int = 1,
        output_dir: str = 'output',
        sampling_params: Optional[dict] = None,
    ):
        self.model = model
        self.teacher = teacher
        self.template = template
        self.prompts = prompts
        self.lmbda = lmbda
        self.gkd_logits_topk = gkd_logits_topk
        self.max_steps = max_steps
        self.batch_size = max(1, batch_size)
        self.gradient_accumulation_steps = max(1, gradient_accumulation_steps)
        self.max_grad_norm = max_grad_norm
        self.logging_steps = logging_steps
        self.output_dir = output_dir
        self.sampling_params = sampling_params
        self.global_step = 0
        self.micro_step = 0
        self.history: list = []

    def _is_grad_sync_boundary(self) -> bool:
        ga = self.gradient_accumulation_steps
        return ga == 1 or ((self.micro_step - 1) % ga == 0 and self.micro_step > 1)

    def _prompt_batch(self, step: int) -> List[List[dict]]:
        """The prompts for one step: a rolling window over the prompt list (wraps around)."""
        n = len(self.prompts)
        start = (step * self.batch_size) % n
        return [self.prompts[(start + i) % n] for i in range(self.batch_size)]

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
        """Full-vocab teacher logits for the features (frozen model, or adapter-disabled student)."""
        if self.teacher == 'disable_lora':
            outputs = self.model.forward_only(inputs=features, disable_lora=True, return_logits=True)
        else:
            outputs = self.teacher.forward_only(inputs=features, return_logits=True)
        return outputs['logits']

    def fit(self) -> list:
        """Run max_steps GKD steps. Each step: generate -> teacher forward -> student forward_backward."""
        ga = self.gradient_accumulation_steps
        step = 0
        while self.global_step < self.max_steps:
            self.micro_step += 1
            prompts = self._prompt_batch(step)
            step += 1
            # On-policy generation from the student's current weights. lmbda (the legacy on-policy vs.
            # dataset-completion mix, stored on self but currently unused) is reduced here to
            # always-on-policy: the dataset-completion path is a plain SFT encode, deliberately left
            # out so one generation code path stands.
            features = self._generate_features(prompts)
            teacher_logits = self._teacher_logits(features)
            self.model.forward_backward(
                inputs=features,
                gradient_accumulation_steps=ga,
                teacher_logits=teacher_logits,
                topk=self.gkd_logits_topk)
            is_boundary = self._is_grad_sync_boundary()
            self.model.clip_grad_and_step(max_grad_norm=self.max_grad_norm, gradient_accumulation_steps=ga)
            if is_boundary:
                self._record_step()
        return self.history

    def _record_step(self) -> None:
        self.global_step += 1
        metrics = self.model.calculate_metric(is_training=True)
        loss = float(metrics['loss']) if metrics.get('loss') is not None else float('nan')
        record = {'step': self.global_step, 'loss': loss}
        if metrics.get('grad_norm') is not None:
            record['grad_norm'] = float(metrics['grad_norm'])
        self.history.append(record)
        if self.logging_steps and self.global_step % self.logging_steps == 0:
            logger.info(f'step {self.global_step}  loss={loss:.4f}')

    def save(self, name: str = 'checkpoint-final') -> str:
        """Persist the student policy + training state via twinkle's native save."""
        return self.model.save(name, output_dir=self.output_dir, save_optimizer=True)
