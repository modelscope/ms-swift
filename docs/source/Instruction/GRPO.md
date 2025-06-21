# GRPO

论文地址

[DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/abs/2402.03300)

环境安装
```bash
pip install math_verify==0.5.2 # reward function
pip install -U trl
```

GRPOTrainer在swift3.5.dev进行了代码重构，如果你使用的swift版本<3.5, 请参考[stable文档](https://github.com/modelscope/ms-swift/blob/v3.4.1/docs/source/Instruction/GRPO.md)

**更新日志**
- **2025-05-29** — 支持了padding_free(--padding_free true)和序列并行(--sequence_parallel_size N)。
- **2025-05-23** — 支持自定义采样批量大小，参考 generation_batch_size / steps_per_generation 参数。
- **2025-05-22** — swift rollout 支持 data_parallel_size 参数。
- **2025-05-16** - 增加 ref_model 同步逻辑，参考参数 sync_ref_model。
- **2025-05-13** — 为了代码的可读性和维护性， GRPOTrainer代码重构，Internal mode 支持vLLM>=0.8。
- **2025-05-11** — 支持生成式奖励模型，通过 reward_model_plugin 自定义奖励模型逻辑。有关更多详细信息，请参阅[自定义奖励模型](#自定义奖励模型)部分。
- **2025-04-30** — external vllm server 的启动命令改为 `swift rollout`。



6. **奖励模型**

除了基于规则的奖励函数外，本框架还支持使用奖励模型作为奖励函数。在使用奖励模型时，需要指定 reward_model 参数，该参数与 model 参数类似，用于指定奖励模型的路径或名称。需要注意的是，reward_model 和 reward_funcs 至少需要指定一个。


## 参数与运行脚本
参数
- per_device_train_batch_size: 每个设备训练批量大小，在GRPO中，指 completion 的批次大小。
- per_device_eval_batch_size: 每个设备评估批量大小，在GRPO中，指 completion 的批次大小。
- generation_batch_size: 采样completion批量大小，需要是 num_processes * per_device_train_batch_size 的倍数，默认等于 per_device_batch_size * gradient_accumulation_steps * num_processes
- steps_per_generation: 每轮生成的优化步数，默认等于gradient_accumulation_steps。与generation_batch_size 只能同时设置一个
- num_generations: 每个prompt采样的数量，论文中的G值，需要被 generation_batch_size 或 per_device_batch_size * steps_per_generation * num_processes 整除，默认为8
- max_completion_length: 采样生成的最大长度，默认为512
- ds3_gather_for_generation: 该参数适用于DeepSpeed ZeRO-3。如果启用，策略模型权重将被收集用于生成，从而提高生成速度。然而，禁用此选项允许训练超出单个GPU VRAM的模型，尽管生成速度会变慢。禁用此选项与vLLM生成不兼容。默认为True
- reward_funcs: 奖励函数，根据模型生成结果进行打分，内置accuracy、format、cosine和repetition四个rule-based函数，详细见 swift/plugin/orm.py 文件
- reward_weights: 每个奖励函数的权重。必须与奖励函数和奖励模型的总数量匹配。如果为 None，则所有奖励的权重都相等，为`1.0`
  - 提示：如果GRPO训练中包含`--reward_model`，则其加在奖励函数的最后位置
- reward_model: 同model, 使用奖励模型作为奖励函数，与reward_funcs至少需要指定一个。
- reward_model_plugin: 奖励模型逻辑，默认为orm逻辑, 详细见[自定义奖励模型](#自定义奖励模型)。
- dataset_shuffle: 是否对dataset进行随机操作，默认为True
- loss_type: loss 归一化的类型，可选项为['grpo', 'bnpo', 'dr_grpo'], 默认为'grpo', 具体查看该[pr](https://github.com/huggingface/trl/pull/3256#discussion_r2033213348)
- log_completions: 是否记录训练中的模型生成内容，搭配 `--report_to wandb` 使用。默认为False
  - 提示：若没有设置`--report_to wandb`，则会在checkpoint中创建`completions.jsonl`来存储生成内容
- use_vllm: 是否使用 vLLM 作为 GRPO 生成的 infer_backend，默认为False。
- vllm_mode: vLLM 集成模式，可选项为 `server` 和 `colocate`。server 模式使用 `swift rollout` 拉起的 vLLM 服务器进行采样，colocate 模式在程序内部署 vLLM。
- vllm_mode server 参数
  - vllm_server_base_url: vLLM server的Base URL(比如 http://local_host:8000), 默认为None。设置后，忽略host和port设置。
  - vllm_server_host：vLLM server host地址，默认为None，使用外部vLLM server时使用.
  - vllm_server_port vLLM server 服务端口，默认为8000.
  - vllm_server_timeout 连接vLLM server的超时时间，默认为 240s.
  - async_generate: 异步rollout以提高训练速度，注意开启时采样会使用上一轮更新的模型进行采样，不支持多轮场景。默认`false`.
- vllm_mode colocate 参数
  - vllm_gpu_memory_utilization: vllm透传参数，默认为0.9.
  - vllm_max_model_len: vllm透传参数，默认为None.
  - vllm_enforce_eager: vllm透传参数，默认为False.
  - vllm_limit_mm_per_prompt: vllm透传参数，默认为None.
  - vllm_enable_prefix_caching: vllm透传参数，默认为True.
  - sleep_level: 训练时释放 vLLM 显存，可选项为[0, 1], 默认为0，不释放.
  - offload_optimizer: 是否在vLLM推理时offload optimizer参数，默认为False。
  - offload_model: 是否在vLLM推理时 offload 模型，默认为False。
  - gc_collect_after_offload: 是否在offload结束时进行gc（python gc和GPU gc），默认为False。
  - completion_length_limit_scope: 在多轮对话中，`max_completion_length` 的限制范围。
  `total`限制所有对话轮次的总输出长度不超过`max_completion_length`, `per_round`限制每一轮的输出长度。
  默认为`per_round`, 当前仅对 colocate mode 生效。
- num_iterations: 每个批次代更新次数，默认为1。
- epsilon: clip 系数，默认为0.2。
- epsilon_high: upper clip 系数，默认为None，设置后与epsilon共同构成[epsilon, epsilon_high]裁剪范围。
- delta: [INTELLECT-2 tech report](https://huggingface.co/papers/2505.07291)中双侧 GRPO 上界裁剪值。若设置，建议大于 1 + epsilon。默认为None。
- sync_ref_model: 是否定期同步ref_model，默认为False。
- ref_model_mixup_alpha: 控制在更新过程中model和先前ref_model之间的混合。更新公式为 $π_{ref} = α * π_θ + (1 - α) * π_{ref_{prev}}$。默认为0.6。
- ref_model_sync_steps：同步频率，默认为512。
- move_model_batches: 在模型向vLLM等快速推理框架移动参数时，将layers分为多少个batch. 默认为None, 代表整个模型不进行拆分，否则拆分为move_model_batches+1(非layer参数)+1(多模态部分参数)个。注意：该参数仅对LoRA(PEFT)训练有意义。
- multi_turn_func: 多轮GRPO参数, 传入对应的plugin名称, 同时在plugin/multi_turn.py中添加好对应的实现。
- dynamic_sample：筛除group内奖励标准差为0的数据，额外采样新数据，默认为False。
- max_resample_times：dynamic_sample设置下限制重采样次数，默认3次。
- overlong_filter：跳过超长截断的样本，不参与loss计算，默认为False。
- padding_free: 去掉所有padding token，并将有效token拼接到一个batch中，仅支持flash_attn.
- sequence_parallel_size: 序列并行段数

奖励函数参数，见[内置奖励函数](#内置奖励函数)

训练脚本参考[这里](https://github.com/modelscope/ms-swift/tree/main/examples/train/grpo)

## DAPO
[Decoupled Clip and Dynamic sAmpling Policy Optimization (DAPO)](https://arxiv.org/abs/2503.14476)在GRPO的基础上设置了几种trick，分别是
- Clip Higher
- Dynamic Sampling
- Overlong Filtering
- Token level Loss
- Soft Overlong Punishment

以上trick，我们可以基于GRPOTrainer，设置以下参数实现。

其中Token level Loss是通过使用参数 loss type `bnpo` 实现

| 参数                 | 类型      | 值      |
|----------------------|-----------|-------------|
|`--loss_type`        | `str`      | `bnpo`     |
| `--epsilon_high`     | `float`   | `0.28`      |
| `--dynamic_sample`   | `bool`    | `true`      |
| `--overlong_filter`  | `bool`    | `true`      |
| `--reward_funcs`     | `str`     | `soft_overlong`|
| `--max_resample_times` | `int`    | `3`        |
