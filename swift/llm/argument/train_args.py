# Copyright (c) Alibaba, Inc. and its affiliates.
import inspect
import math
import os
import platform
from dataclasses import dataclass, field
from typing import List, Literal, Optional

import json
import torch
import torch.distributed as dist
from transformers import Seq2SeqTrainingArguments
from transformers.utils import is_torch_npu_available
from transformers.utils.versions import require_version

from swift.llm import MODEL_MAPPING, TEMPLATE_MAPPING
from swift.plugin import LOSS_MAPPING, extra_tuners
from swift.trainers import TrainerFactory
from swift.utils import (add_version_to_work_dir, get_dist_setting, get_logger, get_pai_tensorboard_dir, is_dist,
                         is_liger_available, is_local_master, is_mp, is_pai_training_job, use_torchacc)
from swift.utils.module_mapping import MODEL_KEYS_MAPPING
from .base_args import BaseArguments

logger = get_logger()


@dataclass
class Seq2SeqTrainingOverrideArguments(Seq2SeqTrainingArguments):
    """Override the default value in `Seq2SeqTrainingArguments`"""

    output_dir: str = 'output'
    gradient_checkpointing: Optional[bool] = None

    save_steps: Optional[int] = None
    save_total_limit: int = 2  # save last and best. -1: all checkpoints
    logging_steps: int = 5
    adam_beta2: float = 0.95
    learning_rate: Optional[float] = None
    weight_decay: float = 0.1
    gradient_accumulation_steps: Optional[int] = None
    lr_scheduler_type: str = 'cosine'
    lr_scheduler_kwargs: Optional[str] = None  # json
    warmup_ratio: float = 0.05
    dataloader_num_workers: Optional[int] = None
    report_to: List[str] = field(default_factory=lambda: ['tensorboard'])
    eval_strategy: Literal['steps', 'epoch', 'no'] = 'steps'

    def __post_init__(self):
        from swift.hub import hub
        if hub.try_login(self.hub_token):
            logger.info('hub login successful!')


@dataclass
class MegatronArguments:

    # megatron
    train_backend: Literal['transformers', 'megatron'] = 'transformers'
    tp: int = 1
    pp: int = 1
    min_lr: Optional[float] = None
    sequence_parallel: bool = False


@dataclass
class TorchAccArguments:
    model_layer_cls_name: Optional[str] = field(
        default=None,
        metadata={'help': "Decoder Class name of model, e.g. 'QWenBlock' for QWen, 'LlamaDecoderLayer' for LLama"})
    metric_warmup_step: Optional[float] = 0
    fsdp_num: int = 1

    def __post_init__(self):
        """Prepare torchacc"""
        if use_torchacc():
            self.dataloader_drop_last = True


@dataclass
class SftArguments(BaseArguments, Seq2SeqTrainingOverrideArguments, MegatronArguments, TorchAccArguments):
    freeze_parameters: List[str] = field(default_factory=list)
    freeze_vit: bool = False
    freeze_parameters_ratio: float = 0.  # 0 ~ 1
    additional_trainable_parameters: List[str] = field(default_factory=list)

    add_output_dir_suffix: Optional[bool] = None
    resume_only_model: bool = False

    packing: bool = False

    # multimodal
    loss_name: Optional[str] = field(default=None, metadata={'help': f'loss_func choices: {list(LOSS_MAPPING.keys())}'})

    loss_scale: str = 'default'

    attn_type: str = 'flash_attention'

    # streaming dataset
    streaming: bool = False
    streaming_val_size: int = 0
    streaming_buffer_size: int = 16384
    batch_size: int = 1
    eval_batch_size: Optional[int] = None
    acc_steps: int = 1

    # other
    test_oom_error: bool = field(
        default=False,
        metadata={
            'help':
            'If set to True, the train_dataset will be sorted in descending order based on max_length, '
            'enabling faster detection of OOM (Out of Memory) errors.'
        })
    lazy_tokenize: Optional[bool] = None
    preprocess_num_proc: int = 1
    ignore_args_error: bool = False  # True: notebook compatibility
    check_model_is_latest: bool = True

    acc_strategy: Literal['token', 'sentence'] = 'token'
    gpu_memory_fraction: Optional[float] = None

    sequence_parallel_size: int = 1

    def prepare_deepspeed(self):
        """Prepare deepspeed settings"""
        ds_config_folder = os.path.abspath(os.path.join(__file__, '..', '..', 'ds_config'))
        deepspeed_mapping = {
            'default-zero2': 'zero2.json',
            'default-zero3': 'zero3.json',
            'zero2-offload': 'zero2_offload.json',
            'zero3-offload': 'zero3_offload.json',
        }
        for ds_name, ds_config in deepspeed_mapping.items():
            if self.deepspeed == ds_name:
                self.deepspeed = os.path.join(ds_config_folder, ds_config)
                break

        if self.deepspeed is not None:
            if is_mp():
                raise ValueError('DeepSpeed is not compatible with MP. '
                                 f'n_gpu: {torch.cuda.device_count()}, '
                                 f'local_world_size: {self.local_world_size}.')
            require_version('deepspeed')
            if self.deepspeed.endswith('.json') or os.path.isfile(self.deepspeed):
                with open(self.deepspeed, 'r', encoding='utf-8') as f:
                    self.deepspeed = json.load(f)
            logger.info(f'Using deepspeed: {self.deepspeed}')

    def prepare_ddp_backend(self):
        """Prepare ddp of course"""
        if is_dist():
            if is_torch_npu_available():
                torch.npu.set_device(self.local_rank)
            else:
                torch.cuda.set_device(self.local_rank)
            self.seed += self.rank  # Avoid the same dropout
            if self.ddp_backend is None:
                self.ddp_backend = 'nccl'
            if self.ddp_backend == 'gloo' and self.quantization_bit != 0:
                raise ValueError('not supported, please use `nccl`')

    def prepare_train_type(self):
        """Some arguments will be decided by the sft_type"""
        if self.is_adapter():
            assert self.freeze_parameters_ratio == 0., (
                'lora does not support `freeze_parameters_ratio`, please set `--sft_type full`')
            assert len(self.additional_trainable_parameters) == 0, (
                'lora does not support `additional_trainable_parameters`, please set `--sft_type full`')
            if self.is_quant_model():
                assert self.quantization_bit == 0, (
                    f'{self.model_type} is already a quantized model and does not need to be quantized again.')
            if self.learning_rate is None:
                self.learning_rate = 1e-4
            if self.eval_steps is None:
                self.eval_steps = 50
        elif self.sft_type == 'full':
            if self.freeze_vit:
                if self.get_model_group():
                    vision_tower = MODEL_KEYS_MAPPING[self.get_model_group()].vision_tower
                    if vision_tower:
                        self.freeze_parameters += vision_tower
            assert 0 <= self.freeze_parameters_ratio <= 1
            assert self.quantization_bit == 0, 'Full parameter fine-tuning does not support quantization.'
            assert self.dtype != 'fp16', ("Fine-tuning with dtype=='fp16' can lead to NaN issues. "
                                          'Please use fp32+AMP or bf16 to perform full parameter fine-tuning.')
            if isinstance(self.additional_trainable_parameters, str):
                self.additional_trainable_parameters = [self.additional_trainable_parameters]
            if self.learning_rate is None:
                self.learning_rate = 1e-5
            if self.eval_steps is None:
                self.eval_steps = 200
        elif self.sft_type not in extra_tuners:
            raise ValueError(f'sft_type: {self.sft_type}')

    def prepare_liger(self):
        """Liger kernel"""
        if self.use_liger:
            assert is_liger_available(), 'use_liger requires liger_kernels, try `pip install liger-kernel`'
            if self.loss_scale != 'default':
                logger.warn('use_liger is not compatible with `loss_scale`, setting to default...')
                self.loss_scale = 'default'

    def prepare_dataloader(self):
        """Prepare dataloader arguments"""
        template_info = TEMPLATE_MAPPING[self.template_type]
        if self.dataloader_num_workers is None:
            if 'dataloader_num_workers' in template_info:
                self.dataloader_num_workers = template_info['dataloader_num_workers']
            elif platform.system() == 'Windows':
                self.dataloader_num_workers = 0
            else:
                self.dataloader_num_workers = 1
            logger.info(f'Setting args.dataloader_num_workers: {self.dataloader_num_workers}')
        if 'dataloader_pin_memory' in template_info:
            self.dataloader_pin_memory = template_info['dataloader_pin_memory']
            logger.info(f'Setting args.dataloader_pin_memory: {self.dataloader_pin_memory}')

    def prepare_gradient_checkpointing(self):
        """Prepare gradient checkpointing arguments"""
        model_info = MODEL_MAPPING.get(self.model_type, {})
        support_gradient_checkpointing = model_info.get('support_gradient_checkpointing', True)
        if self.gradient_checkpointing is None:
            self.gradient_checkpointing = support_gradient_checkpointing
        elif not support_gradient_checkpointing and self.gradient_checkpointing:
            logger.warning(f'{self.model_type} not support gradient_checkpointing.')

    def __post_init__(self) -> None:
        BaseArguments.__post_init__(self)
        Seq2SeqTrainingOverrideArguments.__post_init__(self)
        TorchAccArguments.__post_init__(self)
        if is_pai_training_job():
            self._handle_pai_compat()
        self.prepare_deepspeed()
        self.handle_path()
        if self.sft_type == 'full' or self.train_backend == 'megatron':
            self.model_id_or_path = self.resume_from_checkpoint
        if self.resume_from_checkpoint:
            self.load_from_ckpt_dir()

        if self.save_steps is None:
            self.save_steps = self.eval_steps
        self.train_sampler_random = not self.test_oom_error
        self.load_json_or_path('lr_scheduler_kwargs')
        self.rank, self.local_rank, self.global_world_size, self.local_world_size = get_dist_setting()

        if self.train_backend == 'megatron' and self.sft_type == 'lora':
            logger.warning('Currently, only full parameter is supported. Setting args.sft_type: "full"')
            self.sft_type = 'full'

        if len(self.dataset) == 0:
            raise ValueError(f'self.dataset: {self.dataset}, Please input the training dataset.')

        self.prepare_train_type()
        self.prepare_liger()

        if self.eval_batch_size is None:
            if self.predict_with_generate:
                self.eval_batch_size = 1
            else:
                self.eval_batch_size = self.batch_size
        if self.save_total_limit == -1:
            self.save_total_limit = None

        if self.gradient_accumulation_steps is None:
            self.gradient_accumulation_steps = math.ceil(16 / self.batch_size / self.global_world_size)

        self._handle_streaming_args()
        if self.lazy_tokenize is None and not self.streaming:
            self.lazy_tokenize = self.is_multimodal
            logger.info(f'Setting args.lazy_tokenize: {self.lazy_tokenize}')
        self.prepare_dataloader()
        if 'qwen-audio' in self.model_type:
            assert self.preprocess_num_proc == 1 or self.lazy_tokenize, 'not support'

        self.prepare_gradient_checkpointing()

        if self.train_backend == 'transformers':
            self.init_transformers()
        else:
            self.init_megatron()

        self.prepare_output()

    def prepare_output(self):
        """Prepare the output folder"""
        if self.add_output_dir_suffix is None:
            self.add_output_dir_suffix = True
        if self.add_output_dir_suffix:
            if self.train_backend == 'megatron':
                self.output_dir = os.path.join(self.output_dir, f'{self.model_type}-tp{self.tp}-pp{self.pp}')
            else:
                self.output_dir = os.path.join(self.output_dir, self.model_type)
            self.output_dir = add_version_to_work_dir(self.output_dir)
            logger.info(f'output_dir: {self.output_dir}')
            if self.train_backend == 'transformers':
                self.training_args.output_dir = self.output_dir
                self.training_args.run_name = self.output_dir
        if is_local_master():
            os.makedirs(self.output_dir, exist_ok=True)
        if self.logging_dir is None:
            self.logging_dir = f'{self.output_dir}/runs'
            if self.train_backend == 'transformers':
                self.training_args.logging_dir = self.logging_dir

    def init_megatron(self):
        """Init megatron if you are using megatron to pt"""
        assert is_dist(), 'Please start in distributed mode.'
        dist.init_process_group(backend=self.ddp_backend)
        if self.min_lr is None:
            self.min_lr = self.learning_rate * 0.1

    def init_transformers(self) -> None:
        """Init transformer if you are using transformers models"""
        self.train_type = self.rlhf_type if hasattr(self, 'rlhf_type') else 'sft'
        training_args_cls, kwargs = TrainerFactory.get_training_args_info(self)
        additional_saved_files = []
        if self.sft_type == 'full':
            additional_saved_files = self.get_additional_saved_files()

        kwargs['neftune_noise_alpha'] = self.neftune_noise_alpha

        parameters = inspect.signature(training_args_cls.__init__).parameters
        for k in ['lr_scheduler_kwargs', 'include_num_input_tokens_seen', 'auto_find_batch_size']:
            if k in parameters:
                kwargs[k] = getattr(self, k)
        if 'eval_strategy' in parameters:
            kwargs['eval_strategy'] = self.eval_strategy
        else:
            kwargs['evaluation_strategy'] = self.eval_strategy

        if 'accelerator_config' in parameters:
            kwargs['accelerator_config'] = {'dispatch_batches': False}

        training_args = training_args_cls(
            output_dir=self.output_dir,
            logging_dir=self.logging_dir,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.eval_batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            max_grad_norm=self.max_grad_norm,
            num_train_epochs=self.num_train_epochs,
            max_steps=self.max_steps,
            lr_scheduler_type=self.lr_scheduler_type,
            warmup_ratio=self.warmup_ratio,
            warmup_steps=self.warmup_steps,
            logging_steps=self.logging_steps,
            save_strategy=self.save_strategy,
            save_steps=self.save_steps,
            save_total_limit=self.save_total_limit,
            remove_unused_columns=False,
            bf16=self.bf16,
            fp16=self.fp16,
            eval_steps=self.eval_steps,
            dataloader_num_workers=self.dataloader_num_workers,
            dataloader_pin_memory=self.dataloader_pin_memory,
            metric_for_best_model='rouge-l' if self.predict_with_generate else 'loss',
            greater_is_better=self.predict_with_generate,
            full_determinism=self.full_determinism,
            optim=self.optim,
            adam_beta1=self.adam_beta1,
            adam_beta2=self.adam_beta2,
            adam_epsilon=self.adam_epsilon,
            hub_model_id=self.hub_model_id,
            hub_private_repo=self.hub_private_repo,
            hub_strategy=self.hub_strategy,
            hub_token=self.hub_token,
            push_to_hub=self.push_to_hub,
            resume_from_checkpoint=self.resume_from_checkpoint,
            ignore_data_skip=self.ignore_data_skip,
            ddp_backend=self.ddp_backend,
            gradient_checkpointing=self.gradient_checkpointing,
            local_rank=self.local_rank,
            save_only_model=self.save_only_model,
            train_sampler_random=self.train_sampler_random,
            report_to=self.report_to,
            deepspeed=self.deepspeed,
            additional_saved_files=additional_saved_files,
            disable_tqdm=self.disable_tqdm,
            save_on_each_node=self.save_on_each_node,
            acc_strategy=self.acc_strategy,
            save_safetensors=self.save_safetensors,
            logging_first_step=True,
            metric_warmup_step=self.metric_warmup_step,
            fsdp=self.fsdp,
            fsdp_config=self.fsdp_config,
            dataloader_drop_last=self.dataloader_drop_last,
            seed=self.seed,
            data_seed=self.dataset_seed,
            loss_name=self.loss_name,
            **kwargs)

        training_args.ddp_find_unused_parameters = self.ddp_find_unused_parameters
        training_args.ddp_broadcast_buffers = self.ddp_broadcast_buffers
        training_args.ddp_timeout = self.ddp_timeout
        if is_dist() and training_args.ddp_find_unused_parameters is None:
            if self.gradient_checkpointing:
                training_args.ddp_find_unused_parameters = False
            else:
                training_args.ddp_find_unused_parameters = True

        if is_dist() and training_args.ddp_broadcast_buffers is None:
            if self.gradient_checkpointing:
                training_args.ddp_broadcast_buffers = False
            else:
                training_args.ddp_broadcast_buffers = True

        self.training_args = training_args

    def _handle_pai_compat(self) -> None:
        assert is_pai_training_job()
        logger.info('Handle pai compat...')
        pai_tensorboard_dir = get_pai_tensorboard_dir()
        if self.logging_dir is None and pai_tensorboard_dir is not None:
            self.logging_dir = pai_tensorboard_dir
            logger.info(f'Setting args.logging_dir: {self.logging_dir}')
        if self.add_output_dir_suffix is None:
            self.add_output_dir_suffix = False
            logger.info(f'Setting args.add_output_dir_suffix: {self.add_output_dir_suffix}')

    def _handle_streaming_args(self) -> None:
        """Streaming mode does not support some specific arguments"""
        if not self.streaming:
            return
        if self.max_steps == -1:
            raise ValueError('Please specify `max_steps` in streaming mode.')

        if self.packing:
            self.packing = False
            logger.warning('Packing is not supported for streaming dataset, set to False')

        if self.test_oom_error:
            self.test_oom_error = False
            logger.warning('test_oom_error is not supported for streaming dataset, set to False')

        if self.lazy_tokenize:
            self.lazy_tokenize = False
            logger.info('lazy_tokenize set to False in streaming dataset')

        if self.dataset_test_ratio > 0:
            logger.info('Set dataset_test_ratio to 0 in streaming mode.'
                        'You can manually set val_dataset and val_dataset_sample.'
                        'or set streaming_val_size instead to split from train dataset')
            self.dataset_test_ratio = 0

        if self.dataloader_num_workers is None or self.dataloader_num_workers > 0:
            logger.info('Set dataloader_num_workers to 0 in streaming mode')
            self.dataloader_num_workers = 0


@dataclass
class PtArguments(SftArguments):
    sft_type: Literal['lora', 'full', 'longlora', 'adalora', 'ia3', 'llamapro', 'vera', 'boft'] = 'full'
    target_modules: List[str] = field(default_factory=lambda: ['ALL'])
    lazy_tokenize: Optional[bool] = True
    eval_steps: int = 500


@dataclass
class RLHFArguments(SftArguments):
    rlhf_type: Literal['dpo', 'orpo', 'simpo', 'kto', 'cpo'] = 'dpo'
    ref_model_type: Optional[str] = field(
        default=None, metadata={'help': f'model_type choices: {list(MODEL_MAPPING.keys())}'})
    ref_model_id_or_path: Optional[str] = None
    ref_model_revision: Optional[str] = None

    beta: Optional[float] = None
    label_smoothing: float = 0
    # dpo: 'sigmoid', 'hinge', 'ipo', 'exo_pair', 'nca_pair', 'robust', 'bco_pair',
    #      'sppo_hard', 'aot', 'aot_pair', 'apo_zero', 'apo_down'
    # cpo: 'sigmoid', 'hinge', 'ipo', 'simpo'
    loss_type: Optional[str] = None
    # DPO
    # The alpha parameter from the [RPO](https://huggingface.co/papers/2404.19733) paper V3.
    # The paper recommends `rpo_alpha=1.0`.
    rpo_alpha: float = 1.
    # CPO
    cpo_alpha: float = 1.
    # SimPO
    simpo_gamma: float = 1
    # KTO
    desirable_weight: float = 1.0
    undesirable_weight: float = 1.0

    def __post_init__(self):
        self._check_simpo()
        self._set_default()
        self.ref_model_free = self.rlhf_type in ['cpo', 'orpo']
        super().__post_init__()

    def _check_simpo(self):
        if self.rlhf_type != 'simpo':
            return

        self.rlhf_type = 'cpo'
        if self.loss_type is None:
            self.loss_type = 'simpo'
        if self.beta is None:
            self.beta = 2.

    def _set_default(self):
        if self.beta is None:
            self.beta = 0.1
        if self.loss_type is None:
            if self.rlhf_type in ['dpo', 'cpo']:
                self.loss_type = 'sigmoid'  # else None
