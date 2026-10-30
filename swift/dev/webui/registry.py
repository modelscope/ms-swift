# Copyright (c) ModelScope Contributors. All rights reserved.
"""The canvas component registry: descriptors for the minimal generic set.

Organised by mathematical role, not by algorithm -- one descriptor serves a whole
family via enum params (estimator / loss / reward families point at the twinkle
class families). Roles of reference models are declared by wiring: a frozen model
may be KL anchor, preference reference, teacher or scorer, and the canvas renders
that guidance from the port ``role`` fields. Agentic training is first-class: the
twinkle_agentic building blocks (sandbox envs, toolsets, verifiers, judges,
challengers, multi-turn and external rollouts) all carry descriptors here.
"""
from __future__ import annotations

from .descriptor import ComponentDescriptor, ParamSpec, PortSpec

ADVANTAGE_OPTIONS = (
    'grpo', 'grpo_passk', 'gae', 'rloo', 'remax', 'gpg', 'opo',
    'reinforce_plus_plus', 'reinforce_pp_baseline',
)
LOSS_OPTIONS = (
    'cross_entropy', 'channel', 'chunked_cross_entropy', 'mse',
    'grpo', 'ppo', 'ppo_value', 'gspo', 'sapo', 'cispo', 'bnpo', 'dr_grpo', 'real',
    'gpg', 'gkd', 'opsd', 'mopd', 'kl_cov', 'clip_cov', 'geo_mean',
    'dpo', 'kto', 'cpo', 'orpo', 'simpo', 'dro', 'dppo_kl', 'dppo_tv',
    'infonce', 'online_contrastive', 'cosine_similarity',
)
REWARD_OPTIONS = (
    'math', 'math_verify', 'boxed_math', 'format', 'gsm8k', 'gsm8k_brevity',
    'olympiad_bench', 'multimodal_accuracy', 'f1', 'cot', 'tool_explore',
    'checks', 'rubric_judge',
)

P = PortSpec

POLICY_MODEL = ComponentDescriptor(
    'policy_model', '策略模型', '资源',
    'The trainable policy π_θ: the single actor being optimised. The base port is the '
    'same object viewed with adapters disabled (the LoRA KL anchor), not a second model.',
    ports_out=(P('policy', 'MODEL', 'policy_view'), P('base', 'MODEL', 'base_view')),
    params=(
        ParamSpec('model', '模型', 'Qwen/Qwen3-8B', 'checkpoint id'),
        ParamSpec('train_type', '训练类型', 'lora', 'lora or full: a parameter tier, not a topology fork',
                  options=('lora', 'full')),
        ParamSpec('adapters', '续训 adapter', '', 'checkpoint path; resume training from this adapter', group='训练面'),
        ParamSpec('loss', '损失', 'cross_entropy', 'default objective; a wired loss node or code component overrides it',
                  options=LOSS_OPTIONS, group='训练面'),
        ParamSpec('optim', '优化器', 'adamw', 'optimizer family; lr and schedule live in the train config', group='训练面'),
        ParamSpec('parallel_spec', '并行规格', 'tp2dp4', 'placement of the trainable mesh', group='放置'),
        # 模板面预处理：与 TemplateConfig 字段同名（编码/截断/loss 掩码都在模板里做）
        ParamSpec('max_length', '最大长度', '', 'token budget; empty = template default', group='模板'),
        ParamSpec('truncation_strategy', '截断策略', '', 'delete / left / right / split; empty = off', group='模板'),
        ParamSpec('loss_scale', '损失掩码', 'default', 'which turns carry loss (default / last_round / all)', group='模板'),
        ParamSpec('system', '系统提示词', '', 'prepended system message; empty = none', group='模板'),
    ),
    factory='swift.dev.builders.model:build_model',
)

FROZEN_MODEL = ComponentDescriptor(
    'frozen_model', '冻结模型', '资源',
    'A frozen model copy whose role is declared by wiring: KL anchor, preference '
    'reference, distillation teacher or reward scorer -- the model itself carries no role.',
    ports_out=(P('model', 'MODEL', 'frozen_reference'),),
    params=(
        ParamSpec('model', '模型', '同策略初始权重', 'checkpoint id'),
        ParamSpec('ema', 'EMA 影子', 'false',
                  'true = sync this anchor as an EMA shadow (rl.sync_ref_model); needs the ref wiring',
                  options=('true', 'false')),
    ),
    factory='swift.dev.builders.model:build_ref_model',
)

API_MODEL = ComponentDescriptor(
    'api_model', '外部 API 模型', '资源',
    'An external OpenAI-compatible endpoint used as judge, teacher or utility model: '
    'a reference that never occupies training mesh.',
    ports_out=(P('model', 'MODEL', 'external_api'),),
    params=(
        ParamSpec('base_url', '端点', '', 'OpenAI-compatible base url'),
        ParamSpec('model', '模型名', '', 'the served model name at that endpoint'),
    ),
    factory='twinkle_agentic.protocol.openai:OpenAI',
)

SAMPLER = ComponentDescriptor(
    'sampler', '采样引擎', '资源',
    'A resident inference engine serving the policy view: generation happens here, '
    'weight sync targets are declared by wiring this handle into the sync effect.',
    ports_out=(P('sampler', 'SAMPLER', 'engine'),),
    params=(
        ParamSpec('engine', '引擎', 'vllm', 'inference engine kind', options=('vllm', 'sglang', 'torch')),
        ParamSpec('temperature', '温度', '1.0', 'default sampling temperature'),
        ParamSpec('max_model_len', '最大长度', '8192', 'engine context length; passed through engine args'),
        ParamSpec('gpu_memory_util', '显存占比', '0.8', 'fraction of GPU memory the engine may claim'),
        ParamSpec('sleep_level', '休眠级别', '0',
                  'colocate device time-sharing: 0 = engine stays resident, 1 = offload weights + drop KV, 2 = discard both',
                  options=('0', '1', '2')),
    ),
    factory='swift.dev.builders.sampler',
    client_factory='twinkle_client.sampler:vLLMSampler',
)

DATASET = ComponentDescriptor(
    'dataset', '数据集', '资源',
    'A declared dataset of rows (messages plus free columns): an offline statement, '
    'never a member of the training loop frame.',
    ports_out=(P('data', 'DATA', 'rows'),),
    params=(
        ParamSpec('dataset', '数据集', '', 'dataset id or path'),
        # 预处理旋钮与 DatasetConfig 字段同名（loader 内部按它们构造 Preprocessor 家族）
        ParamSpec('packing', '样本打包', 'false', 'pack short rows into fixed-length sequences', group='预处理'),
        ParamSpec('packing_strategy', '打包策略', 'wrapped', 'packed-sample boundary policy', group='预处理'),
        ParamSpec('streaming', '流式读取', 'false', 'iterate the dataset lazily instead of materializing', group='预处理'),
        ParamSpec('lazy_tokenize', '惰性编码', 'false', 'tokenize inside the dataloader instead of eager map', group='预处理'),
        ParamSpec('split_dataset_ratio', '验证集比例', '0', 'hold out this fraction as val split', group='预处理'),
    ),
    factory='swift.dev.builders.dataset',
)

TOOLSET = ComponentDescriptor(
    'toolset', '工具集', '资源',
    'The JSON-schema tool declarations the policy may call: rolled into the chat '
    'template and mirrored onto the environment tool surface.',
    ports_out=(P('tools', 'TOOL', 'tool_decls'),),
    params=(ParamSpec('tools', '工具', '', 'tool JSON schemas'),),
    factory='twinkle_agentic.tools.tool_manager:ToolManager',
)

SANDBOX_ENV = ComponentDescriptor(
    'sandbox_env', '沙箱环境', '资源',
    'The sandbox where tool calls execute: local subprocess, OpenEnv pool or '
    'microVM. Multi-turn rollouts lease it per episode.',
    ports_out=(P('env', 'ENV', 'sandbox'),),
    params=(ParamSpec('kind', '环境形态', 'local', 'sandbox backend', options=('local', 'openenv', 'agentenv')),),
    factory='twinkle_agentic.envs',
)

POLICY_ENDPOINT = ComponentDescriptor(
    'policy_endpoint', '策略端点', '资源',
    'A local OpenAI-shaped endpoint served by the policy that is being trained: '
    'external agents hit it, so answers always come from the fresh weights.',
    ports_in=(P('policy', 'MODEL', 'policy_view'),),
    ports_out=(P('endpoint', 'MODEL', 'serving_view'),),
    params=(ParamSpec('port', '端口', '8000', 'local serving port'),),
    factory='twinkle_agentic.rollout.endpoint:PolicyEndpoint',
)

STREAMING_DRIVER = ComponentDescriptor(
    'streaming_driver', '流式驱动器', '迭代器',
    'The async control plane: admits prompts, overlaps generation with training under '
    'a staleness budget, and hands the learner assembled batches.',
    ports_in=(P('prompts', 'DATA', 'prompt_stream'),
              P('sampler', 'SAMPLER', 'engine', many=True),
              P('score', 'REWARD', 'scorer', optional=True),
              P('after', 'CTRL', 'next_round', optional=True)),
    ports_out=(P('samples', 'TRAJ', 'batch_stream'),),
    params=(
        ParamSpec('async_mode', '异步模式', 'none', 'none = sync colocate-style, one_step_off = one-batch lookahead, fully_async = stale-by-budget',
                  options=('none', 'one_step_off', 'fully_async')),
        ParamSpec('max_staleness', '最大滞后', '1', 'how many publishes behind the freshest policy a batch may train on'),
        ParamSpec('num_generations', '组宽', '8', 'samples per prompt group'),
    ),
    factory='twinkle_agentic.async_rl.streaming_driver:StreamingDriver',
)

DATALOADER = ComponentDescriptor(
    'dataloader', '数据加载器', '迭代器',
    'for batch in dataloader: the offline and colocate iterator. The loop frame '
    'starts here; the dataset itself stays outside as a declaration.',
    ports_in=(P('data', 'DATA', 'rows'), P('after', 'CTRL', 'next_round', optional=True)),
    ports_out=(P('batch', 'DATA', 'batches'),),
    params=(
        ParamSpec('batch_size', '批大小', '8'),
        ParamSpec('shuffle', '打乱', 'true'),
        ParamSpec('num_workers', '加载进程', '0'),
    ),
    factory='twinkle.dataloader:DataLoader',
)

BATCH_ROLLOUT = ComponentDescriptor(
    'batch_rollout', '批量生成', '迭代器',
    'One blocking batch generation over a prompt batch (the first stage of the '
    'colocate loop body, not an iterator itself).',
    ports_in=(P('prompts', 'DATA', 'prompt_batch'), P('sampler', 'SAMPLER', 'engine')),
    ports_out=(P('samples', 'TRAJ', 'sample_batch'),),
    params=(
        ParamSpec('num_generations', '组宽', '8'),
        ParamSpec('temperature', '温度', '1.0'),
        ParamSpec('max_new_tokens', '生成长度', '512'),
    ),
)

MULTI_TURN_ROLLOUT = ComponentDescriptor(
    'multi_turn_rollout', '多轮生成', '迭代器',
    'The tool-calling episode engine: the policy converses with tools inside the '
    'sandbox until the episode closes; the ledger yields the trainable trajectory.',
    ports_in=(P('prompts', 'DATA', 'prompt_stream'), P('sampler', 'SAMPLER', 'engine'),
              P('tools', 'TOOL', 'tool_decls', optional=True), P('env', 'ENV', 'sandbox', optional=True)),
    ports_out=(P('samples', 'TRAJ', 'episodes'),),
    params=(
        ParamSpec('max_turns', '最大轮数', '8', 'turns per episode; 1 degrades to single-turn'),
    ),
    factory='twinkle_agentic.rollout.multi_turn:MultiTurnRollout',
)

EXTERNAL_ROLLOUT = ComponentDescriptor(
    'external_rollout', '外部 agent 驱动', '迭代器',
    'Control inversion: an external agent program (CLI) drives the episode against '
    'the policy endpoint and reports each round back as a trainable trajectory.',
    ports_in=(P('endpoint', 'MODEL', 'serving_view'),),
    ports_out=(P('samples', 'TRAJ', 'episodes'),),
    params=(ParamSpec('run_script', '驱动脚本', '', 'the agent program to launch'),),
    factory='twinkle_agentic.rollout.external:ExternalRollout',
)

CHALLENGER = ComponentDescriptor(
    'challenger', '自动出题器', '迭代器',
    'A dynamic prompt source: generates tasks scored by rubric (novelty / usefulness '
    '/ complexity), turning the dataset itself into a co-evolving distribution.',
    ports_in=(P('judge', 'MODEL', 'rubric_judge'),),
    ports_out=(P('data', 'DATA', 'generated_prompts'),),
    params=(ParamSpec('theme', '主题', '', 'task family description'),),
    factory='twinkle_agentic.challenger:AgenticChallenger',
)

ADVANTAGE_ESTIMATOR = ComponentDescriptor(
    'advantage_estimator', '优势估计器', '阶段',
    'A(s,a) estimator: turns rewards plus per-channel logps into a baseline-relative '
    'signal, aligned row-for-row with the TRAJ batch.',
    ports_in=(P('samples', 'TRAJ', 'batch'), P('reward', 'REWARD', 'scores', optional=True),
              P('critic', 'MODEL', 'value_head', optional=True),
              P('ref', 'MODEL', 'kl_ref', optional=True),
              P('teacher', 'MODEL', 'teacher', many=True)),
    ports_out=(P('batch', 'TRAJ', 'batch_with_advantage'), P('adv', 'ADV', 'signal')),
    params=(ParamSpec('estimator', '估计器', 'grpo', 'baseline family; plain math names',
                      options=ADVANTAGE_OPTIONS),),
    factory='twinkle.advantage',
)

TEACHER_SIGNAL = ComponentDescriptor(
    'teacher_signal', '教师信号', '阶段',
    'Distillation signal: the teacher forward on the same samples produces logits or '
    'response logps; the output batch is the assembled training kwargs.',
    ports_in=(P('samples', 'TRAJ', 'batch'), P('teacher', 'MODEL', 'teacher', many=True)),
    ports_out=(P('batch', 'TRAJ', 'training_kwargs'),),
    params=(
        # signal form is NOT a user knob: the loss family (gkd/opsd/mopd) decides it
        ParamSpec('teacher_weights', '教师权重', '等权', 'per-teacher mixture weight; empty = equal'),
        ParamSpec('gkd_logits_topk', 'logits top-k', '', 'top-k vocab slice for GKD full-vocab logits'),
    ),
)

REWARD_SCORER = ComponentDescriptor(
    'reward_scorer', '打分器', '阶段',
    'r(τ) → scalar row: rule verifiers, model scorers, programmatic checks or an '
    'LLM judge with weighted rubric -- one column per wired scorer.',
    ports_in=(P('traj', 'TRAJ', 'batch', optional=True), P('model', 'MODEL', 'scorer', optional=True)),
    ports_out=(P('reward', 'REWARD', 'score_columns'),),
    params=(
        ParamSpec('scorer', '打分器', 'math', 'verifier / judge family; plain math names',
                  options=REWARD_OPTIONS),
        ParamSpec('weights', '权重', '1.0', 'per-scorer mixture weight'),
    ),
    factory='twinkle.reward',
)

TRAIN_STEP = ComponentDescriptor(
    'train_step', '训练步', '训练',
    'The invariant carrier: forward_backward → GA boundary → clip → step, with '
    'publish/commit pairing. The objective comes from whichever is wired: a loss '
    'node, a code component LOSS port, or the policy model loss param as fallback. '
    'The skeleton stays closed on purpose -- silently-wrong training has no entry point here.',
    ports_in=(P('samples', 'TRAJ', 'batch'), P('adv', 'ADV', 'signal', optional=True),
              P('policy', 'MODEL', 'policy_view'), P('loss', 'LOSS', 'objective', optional=True)),
    ports_out=(P('ckpt', 'CKPT', 'checkpoint'), P('then', 'CTRL', 'after_step')),
    params=(
        ParamSpec('publish_cadence', '发布节奏', '每 1 个优化步', 'optimizer steps per publish'),
        ParamSpec('max_steps', '步数上限', '1000'),
        ParamSpec('mini_micro_batch', '批划分', '8 / 2'),
        ParamSpec('max_grad_norm', '梯度裁剪', '1.0', 'clip_grad_and_step bound'),
        ParamSpec('save_steps', '保存间隔', '500', 'checkpoint cadence in optimizer steps'),
    ),
)

OFFLINE_TRAINER = ComponentDescriptor(
    'offline_trainer', '离线训练', '训练',
    'The supervised / preference loop body: batches from the dataloader, reference '
    'logps from the wired ref view, no rollout machinery at all.',
    ports_in=(P('model', 'MODEL', 'policy_view'), P('ref', 'MODEL', 'preference_ref', optional=True),
              P('data', 'DATA', 'batches'), P('traj', 'TRAJ', 'surviving_samples', optional=True)),
    ports_out=(P('ckpt', 'CKPT', 'checkpoint'),),
    params=(
        ParamSpec('max_steps', '步数上限', '2000'),
        ParamSpec('mini_micro_batch', '批划分', '8 / 2'),
        ParamSpec('max_grad_norm', '梯度裁剪', '1.0', 'clip_grad_and_step bound'),
        ParamSpec('save_steps', '保存间隔', '500', 'checkpoint cadence in optimizer steps'),
    ),
)

WEIGHT_SYNC = ComponentDescriptor(
    'weight_sync', '权重同步', '效果',
    'A pure effect: the trained weights are pushed into the sampler declared on the '
    'synced port. Strategy and backend decide how the bytes travel.',
    ports_in=(P('policy', 'MODEL', 'policy_view'),
              P('synced', 'SAMPLER', 'receive_weights', many=True),
              P('after', 'CTRL', 'after_step', optional=True)),
    ports_out=(P('then', 'CTRL', 'after_publish'),),
    params=(
        ParamSpec('strategy', '策略', 'in_place', 'in_place / adapter_snapshot', options=('in_place', 'adapter_snapshot')),
        ParamSpec('backend', '后端', 'full', 'full snapshot / sharded delta', options=('full', 'delta_sharded')),
    ),
    factory='swift.dev.weight_sync.manager',
)

MERGE = ComponentDescriptor(
    'merge', '产物合并', '效果',
    'Folds trained adapters into full weights for export; for full-finetune runs the '
    'same node degrades to a pure export/convert step (a declared form, not an error).',
    ports_in=(P('ckpt', 'CKPT', 'checkpoint'),),
    ports_out=(P('model', 'MODEL', 'merged_weights'),),
    params=(ParamSpec('merge_lora', '合并 adapter', 'true', 'false = pure export for full-finetune checkpoints'),),
)

INFER = ComponentDescriptor(
    'infer', '批量推理', '部署',
    'Offline batch inference over the merged model: rows in, predictions out. The '
    'rows come from the wired data port or, equivalently, the dataset param -- the '
    'same declaration the run_infer dataset_config carries.',
    ports_in=(P('model', 'MODEL', 'merged_weights'), P('data', 'DATA', 'rows', optional=True)),
    ports_out=(P('predictions', 'DATA', 'rows'),),
    params=(
        ParamSpec('dataset', '数据集', '', 'dataset id or path; feeds the run when no data port is wired'),
        ParamSpec('num_return_sequences', '采样路数', '1', 'best-of-n width; >=2 with output_format dpo/grpo'),
        ParamSpec('output_format', '输出格式', 'all', 'all / dpo pairs / grpo groups with rollout tokens',
                  options=('all', 'dpo', 'grpo')),
        ParamSpec('max_new_tokens', '生成长度', '512'),
    ),
)

EVAL = ComponentDescriptor(
    'eval', '评测', '部署',
    'A model scored on a data distribution: one node for both evaluation forms -- '
    'an empty eval_url builds a sampler in-process (generative or loss scoring), '
    'a set eval_url scores the served endpoint instead. Metrics leave on the '
    'METRIC channel, the first consumer of that channel.',
    ports_in=(P('model', 'MODEL', 'scored_view'), P('data', 'DATA', 'rows', optional=True)),
    ports_out=(P('metrics', 'METRIC', 'score_report'),),
    params=(
        ParamSpec('eval_dataset', '评测集', '', 'benchmark dataset id(s); feeds the run when no data port is wired'),
        ParamSpec('eval_limit', '评测条数', '0', 'cap on rows per dataset; 0 = all'),
        ParamSpec('eval_url', '评测端点', '', 'OpenAI-compatible url; empty = build a sampler locally'),
        ParamSpec('eval_generation_config', '生成配置', '', 'generation overrides (json or path)'),
        ParamSpec('result_jsonl', '结果落盘', '', 'append the per-sample report here when set'),
    ),
    factory='swift.dev.recipe.infer.run_eval:run_eval',
)

VALUE_MODEL = ComponentDescriptor(
    'value_model', '价值模型', '资源',
    'The trained value head V(s) of PPO: a second trainable model with its own optimizer and '
    'value loss. Placement shares the policy mesh (checked at compile time); it never enters '
    'the publish face -- the sampler only serves the policy.',
    ports_out=(P('value', 'MODEL', 'value_head'),),
    params=(
        ParamSpec('model', '模型', 'Qwen/Qwen3-8B（value 头）'),
        ParamSpec('train_type', '训练类型', 'full', 'lora or full: a parameter tier, not a topology fork',
                  options=('lora', 'full')),
        ParamSpec('parallel_spec', '并行规格', '与 policy 同 mesh（编译期校验）', group='放置'),
    ),
    factory='swift.dev.builders.model:build_model',
)

DEPLOY = ComponentDescriptor(
    'deploy', '部署', '部署',
    'Serve the trained model as a live OpenAI-shaped endpoint (twinkle-server): '
    'one command, one served model, one base url out.',
    ports_in=(P('model', 'MODEL', 'merged_weights'),),
    ports_out=(P('endpoint', 'ENDPOINT', 'served'),),
    params=(
        ParamSpec('infer_backend', '推理后端', 'vllm', options=('vllm', 'sglang', 'torch')),
        ParamSpec('max_model_len', '最大长度', '8192'),
        ParamSpec('route_prefix', '路由前缀', '/v1', 'legacy-compatible chat-completions prefix'),
    ),
    factory='swift.dev.recipe.infer.run_deploy:run_deploy',
)

QUANTIZE = ComponentDescriptor(
    'quantize', '量化', '效果',
    'Post-training weight quantisation (gptq / awq / bnb): model in, quantised model out.',
    ports_in=(P('model', 'MODEL', 'merged_weights'),),
    ports_out=(P('model', 'MODEL', 'quantized_weights'),),
    params=(
        ParamSpec('quant_method', '方法', 'gptq', options=('gptq', 'awq', 'bnb')),
        ParamSpec('quant_bits', '位数', '4'),
    ),
    factory='swift.dev.recipe.export.quantize:run_quantize',
)

HUB_PUSH = ComponentDescriptor(
    'hub_push', '推送 Hub', '效果',
    'Publish the finished weights to the model hub -- the last purely declarative step.',
    ports_in=(P('model', 'MODEL', 'final_weights'),),
    params=(
        ParamSpec('hub_model_id', '目标仓库', '', 'hub repo id to push to (required)'),
        ParamSpec('commit_message', '提交说明', 'webui workflow push'),
        ParamSpec('hub', '目标', 'ModelScope', 'ModelScope / HuggingFace'),
        ParamSpec('private', '私有', 'true', 'create the repo private when missing'),
    ),
    factory='swift.dev.recipe.export.hub:run_push_to_hub',
)

TRAJECTORY_FILTER = ComponentDescriptor(
    'trajectory_filter', '轨迹过滤', '阶段',
    'Rejection sampling on the data plane: keep the winners per prompt group '
    '(best_of_n / threshold / top_k) and drop the rest -- RFT trains on what survives.',
    ports_in=(P('samples', 'TRAJ', 'batch'), P('reward', 'REWARD', 'score_columns', optional=True)),
    ports_out=(P('batch', 'TRAJ', 'kept_batch'),),
    params=(
        ParamSpec('select', '保留策略', 'best_of_n', 'keep 1 winner / above threshold / top-k of the group',
                  options=('best_of_n', 'threshold', 'top_k')),
        ParamSpec('threshold', '阈值', '0.0'),
        ParamSpec('top_k', '保留数', '1'),
    ),
    factory='swift.dev.recipe.rft.run_rft',
)

LOSS_NODE = ComponentDescriptor(
    'loss_node', '损失', '损失',
    'The training objective as a wired callable: one node per math name (clip_cov, '
    'infonce, gkd, ...); weighted sums compose loss graphs.',
    ports_in=(P('traj', 'TRAJ', 'batch', optional=True), P('adv', 'ADV', 'signal', optional=True),
              P('model', 'MODEL', 'policy_view', optional=True)),
    ports_out=(P('loss', 'LOSS', 'objective'),),
    params=(ParamSpec('loss', '损失', 'cross_entropy', 'plain math names from the twinkle loss family',
                      options=LOSS_OPTIONS),),
    factory='twinkle.loss',
)

CODE_COMPONENT = ComponentDescriptor(
    'code_component', '代码组件', '代码',
    'A first-class user node: a small pure transform over the data plane (parse, '
    'reward shaping, metric, cross-actor bridge, or a hand-written objective on '
    'the LOSS port). Ports and output schema are declared by the author; the '
    'token ledger stays closed.',
    ports_in=(P('samples', 'TRAJ', 'batch', optional=True), P('refs', 'MODEL', 'forward_channels', many=True)),
    ports_out=(P('batch', 'TRAJ', 'batch'), P('loss', 'LOSS', 'custom_objective')),
    params=(ParamSpec('name', '名称', 'my_transform'),),
)

REGISTRY: dict[str, ComponentDescriptor] = {
    d.type: d for d in (
        POLICY_MODEL, FROZEN_MODEL, API_MODEL, VALUE_MODEL, SAMPLER, DATASET, TOOLSET,
        SANDBOX_ENV, POLICY_ENDPOINT, STREAMING_DRIVER, DATALOADER, BATCH_ROLLOUT,
        MULTI_TURN_ROLLOUT, EXTERNAL_ROLLOUT, CHALLENGER, ADVANTAGE_ESTIMATOR, TEACHER_SIGNAL,
        REWARD_SCORER, TRAJECTORY_FILTER, TRAIN_STEP, OFFLINE_TRAINER, WEIGHT_SYNC, MERGE,
        INFER, EVAL, DEPLOY, QUANTIZE, HUB_PUSH, LOSS_NODE, CODE_COMPONENT,
    )
}


def validate_registry() -> None:
    """Registration-time self-check: every port kind must have a channel schema.

    Kept as a plain function (not hidden in module import) so the export smoke run
    exercises it explicitly.
    """
    from .schema import CHANNEL_SCHEMAS

    for descriptor in REGISTRY.values():
        for port in descriptor.ports_in + descriptor.ports_out:
            if port.kind not in CHANNEL_SCHEMAS:
                raise ValueError(f'{descriptor.type}.{port.key}: unknown channel kind {port.kind!r}')
