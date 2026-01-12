from dataclasses import dataclass, field
from typing import List, Literal, Optional

from swift.trainers import VllmArguments


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
        structured_outputs_regex (Optional[str]): A regular expression pattern for structured outputs (guided
            decoding). When set, the model's generation is constrained to match the specified regex pattern. This is
            useful for tasks requiring structured outputs like reasoning chains. Defaults to None (disabled).
            Only effective when using vLLM backend (`use_vllm=True`).
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
    # # structured outputs (guided decoding), only effective for vllm backend
    structured_outputs_regex: Optional[str] = None

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
        num_generations_eval (Optional[int]): Number of generations to sample during evaluation. This allows
            using fewer generations during evaluation to save computation. If `None`, uses the value of
            `num_generations`. Defaults to None.
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

    # reward function args, see details in swift/plugins/orm.py
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
    num_generations_eval: Optional[int] = None

    # dataset
    dataset_shuffle: Optional[bool] = True

    # Rollout Importance Sampling Correction (off-policy correction)
    # Set to None to disable, or choose from: 'token_truncate', 'token_mask', 'sequence_truncate', 'sequence_mask'
    rollout_importance_sampling_mode: Optional[Literal['token_truncate', 'token_mask', 'sequence_truncate',
                                                       'sequence_mask']] = None
    rollout_importance_sampling_threshold: float = 2.0  # Threshold for truncation/masking (C in paper)
    log_rollout_offpolicy_metrics: bool = False  # Log off-policy metrics even when IS correction is disabled
    # Off-Policy Sequence Masking: mask out sequences that deviate too much from rollout policy
    # If set, compute mean(rollout_per_token_logps - per_token_logps) per sequence,
    # and mask sequences where this delta > threshold AND advantage < 0
    # Falls back to old_per_token_logps if rollout_per_token_logps is not available
    off_policy_sequence_mask_delta: Optional[float] = None
