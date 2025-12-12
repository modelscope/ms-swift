# Copyright (c) Alibaba, Inc. and its affiliates.
import math
import os
import platform
from dataclasses import dataclass, field
from typing import List, Literal, Optional, Union

from transformers.training_args import TrainingArguments as HfTrainingArguments
from transformers.training_args_seq2seq import Seq2SeqTrainingArguments as HfSeq2SeqTrainingArguments

from swift.plugin import loss_mapping
from swift.utils import get_dist_setting, get_logger, is_liger_available, is_mp, json_parse_to_dict
from .optimizers.galore import GaLoreConfig

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
            to ['tensorboard'].
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
    loss_type: Optional[str] = field(default=None, metadata={'help': f'loss_func choices: {list(loss_mapping.keys())}'})
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
            self.dataloader_prefetch_factor = 10
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
        padding_side (Optional[str]): The padding side for training when batch_size >= 2. Can be 'left' or 'right'.
            Defaults to 'right'.
            Note: For inference with batch_size >= 2, only left padding is performed. PPO and GKD default to 'left'.
        padding_free (Optional[bool]): Whether to flatten data within a batch to avoid padding, reducing VRAM usage
            and speeding up training. Sequences within the same batch remain isolated. Defaults to False. Currently
            supports CPT, SFT, DPO, GRPO, KTO, and GKD.
            Note: It is recommended to use this with '--attn_impl flash_attn' and 'transformers>=4.44'. Compared to
            packing, padding_free has no preprocessing overhead, but packing is faster and has more stable VRAM usage.
        task_type (Optional[str]): The type of task. Can be 'causal_lm', 'seq_cls', 'embedding', 'reranker', or
            'generative_reranker'. Defaults to 'causal_lm'. If set to 'seq_cls', you usually need to also set
            '--num_labels' and '--problem_type'.
        problem_type (Optional[str]): Required for classification models (i.e., when task_type is 'seq_cls'). Can be
            'regression', 'single_label_classification', or 'multi_label_classification'. Defaults to None, which is
            resolved to 'regression' if the model is a reward_model or num_labels is 1, and
            'single_label_classification' otherwise.
        """
    # Value copied from TrainArguments
    train_type: Optional[str] = None
    local_repo_path: Optional[str] = None
    galore_config: Optional[GaLoreConfig] = None
    padding_side: Optional[str] = None
    padding_free: Optional[bool] = None
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
        vllm_use_async_engine (bool): Whether to use async engine for vLLM. Default is False.
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
    vllm_use_async_engine: bool = False
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
class RolloutTrainerArgumentsMixin(VllmArguments):
    """A dataclass for configuring parameters required for GRPO training rollout.

    This mixin provides arguments for controlling the generation process during rollout, especially when using vLLM as
    the inference backend for generation.

    Args:
        top_k (int): The number of highest probability vocabulary tokens to keep for top-k-filtering. Defaults to 50.
        top_p (float): If set to a float < 1, only the smallest set of most probable tokens with probabilities that
            add up to top_p or higher are kept for generation. Defaults to 0.9.
        repetition_penalty (float): The parameter for repetition penalty. 1.0 means no penalty. Defaults to 1.0.
        stop_words (List[str]): A list of strings that will stop the generation when they are generated. Defaults to an
            empty list.
        use_vllm (bool): Whether to use vLLM as the inference backend for GRPO generation. Defaults to False.
        vllm_mode (Literal['server', 'colocate']): The vLLM integration mode. 'server' mode uses a vLLM server launched
            by swift rollout for sampling. 'colocate' mode deploys vLLM within the same process. For full-parameter
            training in 'server' mode, the environment variable `SWIFT_UPDATE_WEIGHTS_BUCKET_SIZE` can be set to
            control the bucket size for weight synchronization (in MB, defaults to 512). Defaults to 'colocate'.
        vllm_enable_prefix_caching (bool): A pass-through parameter for vLLM. Enables prefix caching. Used in
            'colocate' mode. Defaults to True.
        vllm_enable_lora (bool): Enables the vLLM Engine to load LoRA adapters. This is used to accelerate weight
            synchronization during LoRA training. See documentation for details. Used in 'colocate' mode. Defaults
            to False.
        lora_rank (int): The rank for the LoRA adapter loaded by the vLLM engine. When using `vllm_enable_lora`, this
            should be greater than or equal to (ideally equal to) the rank of the LoRA model being trained. Used in
            'colocate' mode. Defaults to 8.
        vllm_server_base_url (Optional[List[str]]): The base URL of the vLLM server (e.g., "http://localhost:8000").
            If set, `vllm_server_host` and `vllm_server_port` are ignored. Used in 'server' mode. Defaults to None.
        vllm_server_host (Optional[List[str]]): The host address(es) of the vLLM
            server. Used in 'server' mode. Defaults to None.
        vllm_server_port (List[int]): The port(s) of the vLLM server. Used in 'server' mode. Defaults to `[8000]`.
        vllm_server_timeout (float): Timeout in seconds for connecting to the vLLM server. Used in 'server' mode.
            Defaults to 240.0.
        vllm_client: Internal client instance for vLLM server communication. Not intended to be set by the user.
        async_generate (bool): Whether to perform asynchronous rollout to improve training speed. Note: When enabled,
            sampling uses the model from the previous update step and is not compatible with multi-turn scenarios.
            Defaults to False.
        sleep_level (int): Specifies the level of GPU memory release for vLLM during training steps. Options: 0
            (no release), 1, 2. A higher level releases more memory but may incur overhead. Defaults to 0.
        move_model_batches (Optional[int]): The number of batches after which the model is moved back to the GPU if it
            was offloaded. Used for memory management during training. Defaults to None.
        offload_optimizer (bool): Whether to offload optimizer states to CPU/RAM when performing inference with vLLM to
            save GPU memory. Defaults to False.
        offload_model (bool): Whether to offload the model weights to CPU/RAM when performing inference with vLLM.
            Defaults to False.
        wandb_log_unique_prompts (Optional[bool]): Whether to log unique prompts to Weights & Biases for analysis
            during training. Defaults to None.
    """
    # generation args
    top_k: int = 50
    top_p: float = 0.9
    repetition_penalty: float = 1.
    stop_words: List[str] = field(default_factory=list)

    # vllm
    use_vllm: bool = False
    vllm_mode: Optional[Literal['server', 'colocate']] = None
    # internal vllm (colocate)
    vllm_max_num_seqs: Optional[int] = None
    vllm_enable_prefix_caching: bool = True  # overwrite
    vllm_enable_lora: bool = False
    lora_rank: int = 8  # for vllm lora adapter
    # external vllm (server)
    vllm_server_base_url: Optional[List[str]] = None
    vllm_server_host: Optional[List[str]] = None
    vllm_server_port: List[int] = field(default_factory=lambda: [8000])
    vllm_server_timeout: float = 240.0
    vllm_client = None  # Not required to set, used for client instantiation
    vllm_server_group_port: Optional[List[int]] = None
    enable_flattened_weight_sync: bool = True
    async_generate: bool = False

    sleep_level: int = 0
    move_model_batches: Optional[int] = None
    offload_optimizer: bool = False
    offload_model: bool = False

    wandb_log_unique_prompts: Optional[bool] = None


@dataclass
class GRPOArgumentsMixin(RolloutTrainerArgumentsMixin):
    """A dataclass for configuring parameters for algorithms like DAPO, Dr.GRPO, GSPO, RLOO, and REINFORCE++.

    Args:
        epsilon (float): The clipping coefficient. Defaults to 0.2.
        epsilon_high (Optional[float]): The upper clipping coefficient. If set, it forms a clipping range of
            `[epsilon, epsilon_high]` with epsilon. Defaults to None.
        delta (Optional[float]): The upper clipping value for two-sided GRPO from the INTELLECT-2 tech
            report. If set, it is recommended to be greater than `1 + epsilon`. Defaults to None.
        cosine_min_len_value_wrong (float): The reward for wrong answers with zero completion length
            (r^w_0 in the paper). Defaults to -0.5.
        cosine_max_len_value_wrong (float): The reward for wrong answers with maximum completion length
            (r^w_L in the paper). Defaults to 0.0.
        cosine_min_len_value_correct (float): The reward for correct answers with zero completion length
            (r^c_0 in the paper). Defaults to 1.0.
        cosine_max_len_value_correct (float): The reward for correct answers with maximum completion length
            (r^c_L in the paper). Defaults to 0.5.
        cosine_max_len (Optional[int]): The maximum length for generated text (Lmax in the paper). Defaults
            to `max_completion_length`.
        repetition_n_grams (int): The n-gram size for repetition detection. Defaults to 3.
        repetition_max_penalty (float): The maximum penalty value, used to control the strength of the
            penalty. Defaults to -1.0.
        reward_model (Optional[List[str]]): The reward model(s) to use. Defaults to None.
        reward_model_plugin (Optional[List[str]]): The plugin logic for the reward model. Defaults to 'orm'
            logic. See custom reward models for details. Defaults to None.
        sync_ref_model (bool): Whether to periodically synchronize the `ref_model`. Defaults to False.
        ref_model_sync_steps (int): The synchronization frequency. Defaults to 512.
        ref_model_mixup_alpha (float): Controls the mixup between the current model and the previous
            `ref_model` during updates. Defaults to 0.6.
        multi_turn_scheduler (Optional[str]): Parameter for multi-turn GRPO. Pass the corresponding plugin
            name. The implementation should be added in `plugin/multi_turn.py`. Defaults to None.
        max_turns (Optional[int]): The maximum number of turns for multi-turn GRPO. If None, there is no
            limit. Defaults to None.
        completion_length_limit_scope (Literal['total', 'per_round']): The scope of the
            `max_completion_length` limit in multi-turn dialogue. 'total' limits the total output length across all
            turns, while 'per_round' limits the output length for each turn. Defaults to 'per_round'.
        vllm_server_pass_dataset (bool): Pass extra dataset information to the vLLM server, used for
            multi-turn training. Defaults to False.
        dynamic_sample (bool): If True, filters out data with a reward standard deviation of 0 within a group
            and samples new data. Defaults to False.
        max_resample_times (int): When `dynamic_sample` is enabled, this limits the number of resampling
            attempts. Defaults to 3.
        overlong_filter (bool): If True, skips samples that are truncated due to being too long, so they are
            not included in the loss calculation. Defaults to False.
        soft_max_length (Optional[int]): The maximum generation length of the model (L_max in the paper).
            Defaults to `max_completion_length`.
        soft_cache_length (Optional[int]): Controls the length penalty interval (L_cache in the paper).
            The interval is `[soft_max_length - soft_cache_length, soft_max_length]`. Defaults to None.
        scale_rewards (Optional[Literal['group', 'batch', 'none']]): Specifies the reward scaling strategy.
            Options are 'group' (scale by intra-group standard deviation), 'batch' (scale by the entire batch's
            standard deviation), or 'none' (no scaling). The default value is tied to `advantage_estimator`: 'group'
            for 'grpo', 'none' for 'rloo', and 'batch' for 'reinforce_plus_plus'. In ms-swift < 3.10, this was a
            boolean where `True` corresponded to 'group' and `False` to 'none'.
        log_entropy (bool): Log the dynamics of entropy values during training. See documentation for
            details. Defaults to False.
        top_entropy_quantile (float): Only tokens with entropy in the top specified quantile participate in
            the loss calculation. A value of 1.0 means no tokens are filtered. See documentation for details.
            Defaults to 1.0.
        importance_sampling_level (Literal['token', 'sequence', 'sequence_token']): Controls the importance
            sampling ratio calculation. 'token' mode retains the original log probability ratio for each token.
            'sequence' mode averages the log probability ratios of all valid tokens in the sequence. The GSPO paper
            uses 'sequence' level for stable training. Defaults to 'token'.
        tau_pos (float): The temperature parameter for positive dominance in the SAPO algorithm, controlling the
            sharpness of the soft gating function. Larger values ​​result in sharper gating (approaching hard
            clipping), while smaller values ​​result in smoother gating. The default value is 1.0.
        tau_neg (float): The temperature parameter for negative dominance in the SAPO algorithm, controlling the
            sharpness of the soft gating function. Typically, `tau_neg` is set > `tau_pos` to impose stronger
            constraints on negative dominance. The default value is 1.05.
        advantage_estimator (Literal['grpo', 'rloo', 'reinforce_plus_plus']): The advantage estimation
            function to use. 'grpo' calculates the relative advantage within a group. Options are 'grpo', 'rloo',
            'reinforce_plus_plus'. Defaults to 'grpo'.
        kl_in_reward (Optional[bool]): Controls how the KL divergence regularization term is handled. If
            `False`, it's an independent term in the loss function. If `True`, KL is directly incorporated into the
            reward (subtracted from it). The default is tied to `advantage_estimator`: `False` for 'grpo', `True` for
            'rloo' and 'reinforce_plus_plus'.
        generation_batch_size (Optional[int]): The batch size for sampling completions. It should be a
            multiple of `num_processes * per_device_train_batch_size`. Defaults to `per_device_batch_size *
            gradient_accumulation_steps * num_processes`.
        steps_per_generation (Optional[int]): The number of optimization steps per generation round. Only
            one of `steps_per_generation` and `generation_batch_size` can be set. Defaults to
            `gradient_accumulation_steps`.
        dataset_shuffle (Optional[bool]): Whether to shuffle the dataset. Defaults to True.
        rollout_importance_sampling_mode (Optional[Literal['token_truncate', 'token_mask', 'sequence_truncate',
            'sequence_mask']]): The training-pull inconsistency correction mode. Options are `token_truncate`,
            `token_mask`, `sequence_truncate`, and `sequence_mask`. Defaults to None, disabling correction.
            See the documentation for details.
        rollout_importance_sampling_threshold (float): The threshold for importance sampling weights, used to truncate
            or mask extreme weights. Defaults to 2.0.
        log_rollout_offpolicy_metrics (bool): Whether to log rollout off-policy diagnostic metrics (KL, PPL, chi2, etc.)
            when `rollout_importance_sampling_mode` is not set. When `rollout_importance_sampling_mode` is set,
            metrics are always logged regardless of this setting. Defaults to False.
    """
    epsilon: float = 0.2
    epsilon_high: Optional[float] = None
    delta: Optional[float] = None

    # reward function args, see details in swift/plugin/orm.py
    # cosine reward, https://arxiv.org/abs/2502.03373
    cosine_min_len_value_wrong: float = -0.5  # r^w_0 in paper, Reward for wrong answers with zero completion length.
    cosine_max_len_value_wrong: float = 0.0  # r^w_L in paper, Reward for wrong answers with max completion length.
    cosine_min_len_value_correct: float = 1.0  # r^c_0 in paper, Reward for correct answers with zero completion length.
    cosine_max_len_value_correct: float = 0.5  # r^c_L in paper, Reward for correct answers with max completion length.
    cosine_max_len: Optional[int] = None  # Lmax in paper, default equal to max_completion_length
    # repetition penalty, https://arxiv.org/abs/2502.03373
    repetition_n_grams: int = 3
    repetition_max_penalty: float = -1.0

    reward_model: Optional[List[str]] = None
    reward_model_plugin: Optional[List[str]] = None

    # sync ref model
    sync_ref_model: bool = False
    ref_model_sync_steps: int = 512
    ref_model_mixup_alpha: float = 0.6

    # multi turn
    multi_turn_scheduler: Optional[str] = None
    max_turns: Optional[int] = None
    completion_length_limit_scope: Literal['total', 'per_round'] = 'per_round'
    vllm_server_pass_dataset: bool = False

    # DAPO, https://arxiv.org/abs/2503.14476
    dynamic_sample: bool = False
    max_resample_times: int = 3
    overlong_filter: bool = False
    soft_max_length: Optional[int] = None
    soft_cache_length: Optional[int] = None

    # Dr. GRPO, https://arxiv.org/abs/2503.20783
    scale_rewards: Optional[Literal['group', 'batch', 'none']] = None

    # entropy
    log_entropy: bool = False
    # Beyond the 80/20 Rule, https://arxiv.org/abs/2506.01939
    top_entropy_quantile: float = 1.0

    # GSPO https://arxiv.org/abs/2507.18071
    importance_sampling_level: Literal['token', 'sequence', 'sequence_token'] = 'token'

    # SAPO https://arxiv.org/abs/2511.20347
    # Temperature parameters for soft adaptive gate
    tau_pos: float = 1.0
    tau_neg: float = 1.05

    # RLOO, REINFORCE++
    advantage_estimator: Literal['grpo', 'rloo', 'reinforce_plus_plus'] = 'grpo'
    # If false, add KL into loss, otherwise add into reward
    kl_in_reward: Optional[bool] = None  # rloo/reinforce_plus_plus: true, grpo: false (default)

    generation_batch_size: Optional[int] = None
    steps_per_generation: Optional[int] = None

    # dataset
    dataset_shuffle: Optional[bool] = True

    # Rollout Importance Sampling Correction (off-policy correction)
    # Set to None to disable, or choose from: 'token_truncate', 'token_mask', 'sequence_truncate', 'sequence_mask'
    rollout_importance_sampling_mode: Optional[Literal['token_truncate', 'token_mask', 'sequence_truncate',
                                                       'sequence_mask']] = None
    rollout_importance_sampling_threshold: float = 2.0  # Threshold for truncation/masking (C in paper)
    log_rollout_offpolicy_metrics: bool = False  # Log off-policy metrics even when IS correction is disabled


@dataclass
class TrainingArguments(SwiftArgumentsMixin, HfTrainingArguments):
    pass


@dataclass
class Seq2SeqTrainingArguments(SwiftArgumentsMixin, HfSeq2SeqTrainingArguments):
    pass
