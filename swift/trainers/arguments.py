# Copyright (c) Alibaba, Inc. and its affiliates.
import math
import os
import platform
from dataclasses import dataclass, field
from typing import List, Literal, Optional, Union

from transformers.training_args import TrainingArguments as HfTrainingArguments
from transformers.training_args_seq2seq import Seq2SeqTrainingArguments as HfSeq2SeqTrainingArguments

from swift.loss import loss_map
from swift.optimizers.galore import GaLoreConfig
from swift.utils import get_dist_setting, get_logger, is_liger_available, is_mp, json_parse_to_dict

logger = get_logger()


@dataclass
class TrainArgumentsMixin:
    """A dataclass mixin for configuring model training parameters.

    Args:
        per_device_train_batch_size (int): The batch size per GPU/TPU core for training. Defaults to 1.
        per_device_eval_batch_size (int): The batch size per GPU/TPU core for evaluation. Defaults to 1.
        gradient_accumulation_steps (Optional[int]): The number of update steps to accumulate gradients for before
            performing an optimizer step.
        tuner_backend (Optional[str]): The backend to use for parameter-efficient fine-tuning (e.g., 'peft'). Defaults
            to None.
        gradient_checkpointing (bool): If True, use gradient checkpointing to save memory at the cost of a slower
            backward pass. Defaults to True.
        vit_gradient_checkpointing (Optional[bool]): A specific gradient checkpointing setting for the Vision
            Transformer part of the model. Defaults to None.
        gradient_checkpointing_kwargs (Optional[Union[dict, str]]): Keyword arguments for
            `torch.utils.checkpoint.checkpoint`. Defaults to None.
        logging_first_step (bool): Whether to log the first global step. Defaults to True.
        logging_steps (int): Log every `logging_steps` global steps. Defaults to 5.
        router_aux_loss_coef (float): The coefficient for the router auxiliary loss in Mixture-of-Experts models.
            Defaults to 0.0.
        enable_dft_loss (bool): Whether to enable Diversity-from-Diversity (DFD) loss.
            See https://arxiv.org/abs/2508.05629. Defaults to False.
        enable_channel_loss (bool): Whether to enable channel loss. Defaults to False.
        weight_decay (float): The weight decay to apply (if not zero) to all layers except bias and LayerNorm weights.
            Defaults to 0.1.
        adam_beta2 (float): The beta2 hyperparameter for the AdamW optimizer. Defaults to 0.95.
        lr_scheduler_type (str): The learning rate scheduler type to use. Defaults to 'cosine'.
        lr_scheduler_kwargs (Optional[Union[dict, str]]): Additional keyword arguments for the learning rate scheduler,
            passed as a JSON string or a dictionary. Defaults to None.
        report_to (List[str]): The list of integrations to report results to (e.g., 'tensorboard', 'wandb'). Defaults
            to ['tensorboard']. If you specify `--report_to wandb`, you can set the project name through `WANDB_PROJECT`
            and specify the API KEY corresponding to your account through `WANDB_API_KEY`.
        dataloader_num_workers (Optional[int]): The number of subprocesses to use for data loading. Defaults to None.
        dataloader_persistent_workers (bool): If True, the data loader workers will not be shut down after a dataset
            has been consumed once. Defaults to False.
        dataloader_prefetch_factor (Optional[int]): The number of batches loaded in advance by each worker. Defaults
            to None.
        use_liger_kernel (bool): Whether to use the Liger kernel for optimization. Defaults to False.
        check_model (bool): If True, checks local model files for corruption or modification and provides a warning.
            Should be set to False in an offline environment. Defaults to True.
        acc_strategy (Literal['token', 'seq']): The strategy for calculating accuracy during training and validation.
            Can be 'token' for token-level accuracy or 'seq' for sequence-level accuracy. Defaults to 'token'.
        train_dataloader_shuffle (bool): Whether to shuffle the training data. Defaults to True.
        max_epochs (Optional[int]): The total number of training epochs to perform. Overrides `num_train_epochs`.
            Defaults to None.
        aligner_lr (Optional[float]): A specific learning rate for the aligner part of the model. Defaults to None.
        vit_lr (Optional[float]): A specific learning rate for the Vision Transformer part of the model. Defaults to
            None.
        use_logits_to_keep (Optional[bool]): If enabled, reduces VRAM usage and speeds up training by calculating and
            storing only the necessary logits based on the labels during the forward pass. If None, the behavior is
            automatically determined. Defaults to None.
        ds3_gather_for_generation (bool): In DeepSpeed ZeRO-3, whether to gather model parameters for generation.
            Defaults to True.
        resume_only_model (bool): When resuming from a checkpoint, whether to load only the model weights and not the
            optimizer/scheduler states. Defaults to False.
        optimizer (Optional[str]): The name of a custom optimizer from a plugin. If None, a default optimizer is used.
            See documentation for available choices. Defaults to None.
        loss_type (Optional[str]): The name of a custom loss function from a plugin. If None, the model's default loss
            function is used. Defaults to None.
        metric (Optional[str]): The name of a custom metric from a plugin. If None, it defaults to 'nlg' when
            `predict_with_generate=True`. Defaults to None.
        eval_use_evalscope (bool): Whether to use EvalScope for evaluation during training. Must be set to `True` to
            enable it. Refer to examples for usage details. Defaults to False.
        eval_dataset (List[str]): A list of evaluation dataset names. Multiple datasets can be specified, separated
            by spaces.
        eval_dataset_args (Optional[Union[str, dict]]): Arguments for the evaluation dataset(s), provided as a JSON
            string or a dictionary.
        eval_limit (Optional[int]): The maximum number of samples to use from the evaluation dataset. Defaults to None.
        eval_generation_config (Optional[Union[str, dict]]): Model inference configuration for evaluation, provided as
            a JSON string or a dictionary, e.g., `{'max_tokens': 512}`. Defaults to None.
        extra_eval_args (Optional[Union[str, dict]]): Extra arguments for evaluation, provided as a JSON string or a
            dictionary.
        use_flash_ckpt (bool): Whether to enable DLRover Flash Checkpoint. When enabled, weights are first saved to
            shared memory and then asynchronously persisted to disk. Currently does not support the safetensors format.
            It is recommended to use this with `PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"` to prevent CUDA OOM
            errors during training. Defaults to False.
    """
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: Optional[int] = None
    tuner_backend: Optional[str] = None

    gradient_checkpointing: bool = True
    vit_gradient_checkpointing: Optional[bool] = None
    gradient_checkpointing_kwargs: Optional[Union[dict, str]] = None
    logging_first_step: bool = True
    logging_steps: int = 5
    router_aux_loss_coef: float = 0.
    enable_dft_loss: bool = False  # https://arxiv.org/abs/2508.05629
    enable_channel_loss: bool = False

    weight_decay: float = 0.1
    adam_beta2: float = 0.95
    lr_scheduler_type: str = 'cosine'
    lr_scheduler_kwargs: Optional[Union[dict, str]] = None
    report_to: List[str] = field(default_factory=lambda: ['tensorboard'])
    dataloader_num_workers: Optional[int] = None
    dataloader_persistent_workers: bool = False
    dataloader_prefetch_factor: Optional[int] = None
    use_liger_kernel: bool = False

    # extra
    check_model: bool = True
    acc_strategy: Literal['token', 'seq'] = 'token'
    train_dataloader_shuffle: bool = True
    max_epochs: Optional[int] = None
    aligner_lr: Optional[float] = None
    vit_lr: Optional[float] = None
    use_logits_to_keep: Optional[bool] = None
    ds3_gather_for_generation: bool = True
    resume_only_model: bool = False

    optimizer: Optional[str] = None
    loss_type: Optional[str] = field(default=None, metadata={'help': f'loss_func choices: {list(loss_map.keys())}'})
    metric: Optional[str] = None

    # train-eval loop args
    eval_use_evalscope: bool = False
    eval_dataset: List[str] = field(default_factory=list)
    eval_dataset_args: Optional[Union[str, dict]] = None
    eval_limit: Optional[int] = None
    eval_generation_config: Optional[Union[str, dict]] = None
    extra_eval_args: Optional[Union[str, dict]] = None

    # dlrover flash_checkpoint
    use_flash_ckpt: bool = False

    @staticmethod
    def _patch_liger_kernel():
        # fix logits_to_keep
        from liger_kernel.transformers.model import loss_utils
        origin_LigerForCausalLMLoss = loss_utils.LigerForCausalLMLoss

        def LigerForCausalLMLoss(hidden_states, *args, **kwargs):
            hidden_states = hidden_states.contiguous()
            for key in ['cu_seq_lens_q', 'cu_seq_lens_k', 'max_length_q', 'max_length_k']:
                kwargs.pop(key, None)
            return origin_LigerForCausalLMLoss(hidden_states, *args, **kwargs)

        loss_utils.LigerForCausalLMLoss = LigerForCausalLMLoss
        logger.info('Patch liger_kernel successfully.')

    def _init_liger(self):
        if self.use_liger_kernel:
            assert is_liger_available(), 'use_liger_kernel requires liger_kernels, try `pip install liger-kernel`'
            try:
                self._patch_liger_kernel()
            except Exception:
                pass

    def __post_init__(self):
        if is_mp() and self.use_liger_kernel:
            raise ValueError('liger_kernel does not support device_map. '
                             'Please use DDP/DeepSpeed for multi-GPU training.')

        if self.optimizer is None and (self.vit_lr is not None or self.aligner_lr is not None):
            self.optimizer = 'multimodal'
        if self.gradient_accumulation_steps is None:
            world_size = get_dist_setting()[2]
            self.gradient_accumulation_steps = max(1, math.ceil(16 / self.per_device_train_batch_size / world_size))
            logger.info(f'Setting args.gradient_accumulation_steps: {self.gradient_accumulation_steps}')
        if self.lr_scheduler_kwargs:
            self.lr_scheduler_kwargs = json_parse_to_dict(self.lr_scheduler_kwargs)
        if 'wandb' in self.report_to:
            os.environ.setdefault('WANDB_PROJECT', 'ms-swift')
        if self.vit_gradient_checkpointing is None:
            self.vit_gradient_checkpointing = self.gradient_checkpointing
        if self.gradient_checkpointing_kwargs:
            self.gradient_checkpointing_kwargs = json_parse_to_dict(self.gradient_checkpointing_kwargs)
        self._init_liger()
        if self.dataloader_num_workers is None:
            if platform.system() == 'Windows':
                self.dataloader_num_workers = 0
            else:
                self.dataloader_num_workers = 1
            logger.info(f'Setting args.dataloader_num_workers: {self.dataloader_num_workers}')
        if self.dataloader_prefetch_factor is None and self.dataloader_num_workers > 0:
            self.dataloader_prefetch_factor = 2
        if self.eval_use_evalscope:
            try:
                import evalscope
            except ImportError:
                raise ImportError('evalscope is not installed, please install it by `pip install evalscope`')
            self.eval_dataset_args = json_parse_to_dict(self.eval_dataset_args)
            self.eval_generation_config = json_parse_to_dict(self.eval_generation_config)
            self.extra_eval_args = json_parse_to_dict(self.extra_eval_args)

        super().__post_init__()


@dataclass
class RLHFArgumentsMixin:
    """A dataclass mixin for GKD and CHORD training.

    Args:
        sft_alpha (float): The weight for the SFT loss component in GKD. The final loss is calculated as
            `gkd_loss + sft_alpha * sft_loss`. Defaults to 0.
        chord_sft_dataset (List[str]): The SFT dataset(s) used to provide expert data for the CHORD algorithm. Defaults
            to `[]`.
        chord_sft_per_device_train_batch_size (Optional[int]): The SFT mini-batch size per device for the CHORD
            algorithm. Defaults to None.
        chord_enable_phi_function (bool): Whether to enable the token-wise weighting function phi (φ) in the CHORD
            algorithm. Defaults to False.
        chord_mu_warmup_steps (Optional[int]): The number of training steps for the mu (μ) value to warm up to its peak
            value. Defaults to None.
        chord_mu_decay_steps (Optional[int]): The number of training steps for the mu (μ) value to decay from its peak
            to its valley value. Defaults to None.
        chord_mu_peak (Optional[float]): The peak value for mu (μ) during its schedule. Defaults to None.
        chord_mu_valley (Optional[float]): The final (valley) value for mu (μ) after decay. Defaults to None.
    """
    # gkd
    sft_alpha: float = 0
    # chord
    chord_sft_dataset: List[str] = field(default_factory=list)
    chord_sft_per_device_train_batch_size: Optional[int] = None

    chord_enable_phi_function: bool = False
    chord_mu_warmup_steps: Optional[int] = None
    chord_mu_decay_steps: Optional[int] = None
    chord_mu_peak: Optional[float] = None
    chord_mu_valley: Optional[float] = None


@dataclass
class SwiftArgumentsMixin(RLHFArgumentsMixin, TrainArgumentsMixin):
    """A dataclass for configuring additional training parameters.

    Args:
        train_type (Optional[str]): The training type. Can be 'lora', 'full', 'longlora', 'adalora', 'llamapro',
            'adapter', 'vera', 'boft', 'fourierft', or 'reft'. Defaults to 'lora'.
        local_repo_path (Optional[str]): Path to a local repository. Some models (e.g., deepseek-vl2) depend on a
            GitHub repo for loading. Using a local repo avoids network issues during 'git clone'. Defaults to None.
        galore_config (Optional[GaLoreConfig]): GaLore configuration. Defaults to None.
        task_type (Optional[str]): The type of task. Can be 'causal_lm', 'seq_cls', 'embedding', 'reranker', or
            'generative_reranker'. Defaults to 'causal_lm'. If set to 'seq_cls', you usually need to also set
            '--num_labels' and '--problem_type'.
        problem_type (Optional[str]): Required for classification models (i.e., when task_type is 'seq_cls'). Can be
            'regression', 'single_label_classification', or 'multi_label_classification'. Defaults to None, which is
            resolved to 'regression' if the model is a reward_model or num_labels is 1, and
            'single_label_classification' otherwise.
        """
    # Value copied from SftArguments
    train_type: Optional[str] = None
    local_repo_path: Optional[str] = None
    task_type: Optional[str] = None
    problem_type: Optional[str] = None

    def __post_init__(self):
        if hasattr(self, 'output_dir'):
            self.output_dir = os.path.abspath(os.path.expanduser(self.output_dir))
        super().__post_init__()


@dataclass
class VllmArguments:
    """VllmArguments is a dataclass that holds the configuration for vllm.

    Args:
        vllm_gpu_memory_utilization (float): GPU memory utilization. Default is 0.9.
        vllm_tensor_parallel_size (int): Tensor parallelism size. Default is 1.
        vllm_pipeline_parallel_size (int): Pipeline parallelism size. Default is 1.
        vllm_enable_expert_parallel (bool): Flag to enable expert parallelism for MoE models. Default is False.
        vllm_max_num_seqs (int): Maximum number of sequences. Default is 256.
        vllm_max_model_len (Optional[int]): Maximum model length. Default is None.
        vllm_disable_custom_all_reduce (bool): Flag to disable custom all-reduce. Default is True.
        vllm_enforce_eager (bool): Flag to enforce eager execution. Default is False.
        vllm_limit_mm_per_prompt (Optional[str]): Limit multimedia per prompt. Default is None.
        vllm_max_lora_rank (int): Maximum LoRA rank. Default is 16.
        vllm_enable_prefix_caching (Optional[bool]): Flag to enable automatic prefix caching. Default is None.
        vllm_use_async_engine (Optional[bool]): Whether to use async engine for vLLM. Default is None,
            which will be set to True for encode tasks (embedding, seq_cls, reranker, generative_reranker),
            deployment scenarios (swift deploy) and False otherwise.
        vllm_quantization (Optional[str]): The quantization method for vLLM. Default is None.
        vllm_reasoning_parser (Optional[str]): The reasoning parser for vLLM. Default is None.
        vllm_disable_cascade_attn (bool): Flag to disable cascade attention. Default is False.
        vllm_mm_processor_cache_gb (Optional[float]): MM processor cache size in GB. Default is None.
        vllm_speculative_config (Optional[Union[dict, str]]): Speculative decoding configuration, passed in as a JSON
            string. Defaults to None.
        vllm_engine_kwargs (Optional[Union[dict, str]]): Additional parameters for vllm, formatted as a JSON string.
            Defaults to None.
        vllm_data_parallel_size (int): Data parallelism size for vLLM rollout. Default is 1.
    """
    # vllm
    vllm_gpu_memory_utilization: float = 0.9
    vllm_tensor_parallel_size: int = 1
    vllm_pipeline_parallel_size: int = 1
    vllm_enable_expert_parallel: bool = False
    vllm_max_num_seqs: int = 256
    vllm_max_model_len: Optional[int] = None
    vllm_disable_custom_all_reduce: bool = True
    vllm_enforce_eager: bool = False
    vllm_limit_mm_per_prompt: Optional[Union[dict, str]] = None  # '{"image": 5, "video": 2}'
    vllm_max_lora_rank: int = 16
    vllm_enable_prefix_caching: Optional[bool] = None
    vllm_use_async_engine: Optional[bool] = None
    vllm_quantization: Optional[str] = None
    vllm_reasoning_parser: Optional[str] = None
    vllm_disable_cascade_attn: bool = False
    vllm_mm_processor_cache_gb: Optional[float] = None
    vllm_speculative_config: Optional[Union[dict, str]] = None
    vllm_engine_kwargs: Optional[Union[dict, str]] = None
    # rollout
    vllm_data_parallel_size: int = 1

    def __post_init__(self):
        if self.vllm_limit_mm_per_prompt is not None:
            self.vllm_limit_mm_per_prompt = json_parse_to_dict(self.vllm_limit_mm_per_prompt)
        if self.vllm_speculative_config is not None:
            self.vllm_speculative_config = json_parse_to_dict(self.vllm_speculative_config)
        self.vllm_engine_kwargs = json_parse_to_dict(self.vllm_engine_kwargs)

    def get_vllm_engine_kwargs(self):
        adapters = self.adapters
        if hasattr(self, 'adapter_mapping'):
            adapters = adapters + list(self.adapter_mapping.values())
        kwargs = {
            'gpu_memory_utilization': self.vllm_gpu_memory_utilization,
            'tensor_parallel_size': self.vllm_tensor_parallel_size,
            'pipeline_parallel_size': self.vllm_pipeline_parallel_size,
            'enable_expert_parallel': self.vllm_enable_expert_parallel,
            'max_num_seqs': self.vllm_max_num_seqs,
            'max_model_len': self.vllm_max_model_len,
            'disable_custom_all_reduce': self.vllm_disable_custom_all_reduce,
            'enforce_eager': self.vllm_enforce_eager,
            'limit_mm_per_prompt': self.vllm_limit_mm_per_prompt,
            'max_lora_rank': self.vllm_max_lora_rank,
            'enable_lora': len(adapters) > 0,
            'max_loras': max(len(adapters), 1),
            'enable_prefix_caching': self.vllm_enable_prefix_caching,
            'use_async_engine': self.vllm_use_async_engine,
            'quantization': self.vllm_quantization,
            'reasoning_parser': self.vllm_reasoning_parser,
            'disable_cascade_attn': self.vllm_disable_cascade_attn,
            'mm_processor_cache_gb': self.vllm_mm_processor_cache_gb,
            'speculative_config': self.vllm_speculative_config,
            'num_labels': self.num_labels,
            'engine_kwargs': self.vllm_engine_kwargs,
        }
        if self.task_type in ('embedding', 'seq_cls') or 'reranker' in self.task_type:
            kwargs['task_type'] = self.task_type

        return kwargs


@dataclass
class TrainingArguments(SwiftArgumentsMixin, HfTrainingArguments):
    pass


@dataclass
class Seq2SeqTrainingArguments(SwiftArgumentsMixin, HfSeq2SeqTrainingArguments):
    pass
