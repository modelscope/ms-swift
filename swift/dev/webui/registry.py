# Copyright (c) ModelScope Contributors. All rights reserved.
"""通用组件 v3 注册表：数学职责、真实端口与类型化配置的唯一声明。"""
from __future__ import annotations

from .descriptor import ComponentDescriptor, ParamSpec, PortSpec

REGISTRY_VERSION = 3
ADVANTAGE_OPTIONS = ('grpo', 'grpo_passk', 'gae', 'rloo', 'remax', 'gpg', 'opo',
                     'reinforce_plus_plus', 'reinforce_pp_baseline')
LOSS_OPTIONS = ('cross_entropy', 'channel', 'chunked_cross_entropy', 'mse', 'grpo', 'ppo',
                'ppo_value', 'gspo', 'sapo', 'cispo', 'bnpo', 'dr_grpo', 'real', 'gpg',
                'gkd', 'opsd', 'mopd', 'kl_cov', 'clip_cov', 'geo_mean', 'dpo', 'kto',
                'cpo', 'orpo', 'simpo', 'dro', 'dppo_kl', 'dppo_tv', 'infonce',
                'online_contrastive', 'cosine_similarity')
REWARD_OPTIONS = ('math', 'math_verify', 'boxed_math', 'format', 'gsm8k', 'gsm8k_brevity',
                  'olympiad_bench', 'multimodal_accuracy', 'f1', 'cot', 'tool_explore',
                  'checks', 'rubric_judge')
P = PortSpec
Q = ParamSpec
D = ComponentDescriptor

MODEL = D('model', '模型', 'Resource', '唯一参数化函数 f_theta；独立冻结模型必须独立构造。',
    ports_in=(P('artifact', 'ARTIFACT', '模型权重来源', ('model', 'training'), optional=True),),
    ports_out=(P('model', 'MODEL', '唯一模型对象'),),
    params=(Q('model', '模型来源', '', '模型 ID/路径；接入 artifact 时由真实产物提供'),
            Q('train_type', '训练类型', 'lora', options=('lora', 'full')),
            Q('freeze', '冻结参数', False, type='bool'),
            Q('backend', '训练后端', 'hf', options=('hf', 'megatron', 'unsloth')),
            Q('mode', '执行模式', 'ray', options=('ray', 'local', 'remote')),
            Q('parallel_spec', '并行规格', '', '由 DeviceMesh 解释，不强制单卡'),
            Q('model_config', '模型配置', {}, type='dict'),
            Q('train_config', '训练参数选择', {}, type='dict'),
            Q('tuner_config', '调参配置', {}, type='dict'),
            Q('distributed_config', '完整放置配置', {}, type='dict')),
    owner_kind='MODEL')

DATASET = D('dataset', '数据集', 'Resource', '原始数据分布与预处理配置，不进入循环。',
    ports_out=(P('data', 'DATASET', '原生数据来源'),),
    params=(Q('dataset', '数据来源', [], type='list', item_type='str', required=True),
            Q('dataset_config', '预处理配置', {}, type='dict')))

SAMPLER = D('sampler', '采样器', 'Resource', '常驻生成引擎；权重初始来源与编码配置是参数，同步由 weight_sync 表达。',
    ports_out=(P('sampler', 'SAMPLER', '生成引擎'),),
    params=(Q('model', '模型来源（id/路径，同时是 tokenizer 来源）', '', required=True),
            Q('engine', '引擎', 'vllm', options=('vllm', 'sglang', 'torch')),
            Q('parallel_spec', '并行规格', ''), Q('mode', '执行模式', 'ray', options=('ray', 'local', 'remote')),
            Q('engine_args', '引擎参数', {}, type='dict'),
            Q('model_config', '模型加载配置', {}, type='dict'),
            Q('template_config', '编码配置', {}, type='dict'),
            Q('sleep_level', '休眠级别', 0, type='int', options=(0, 1, 2))), owner_kind='SAMPLER')

ROLLOUT = D('rollout', '多轮生成器', 'Resource', '在环境 lease 内执行多轮生成，不包含评分与优势估计。',
    ports_in=(P('sampler', 'SAMPLER', '生成 backend'), P('tools', 'TOOL', '工具插件与声明', optional=True),
              P('env', 'ENV', '环境 lease', optional=True)),
    ports_out=(P('rollout', 'ROLLOUT', 'episode 生成机制'),),
    params=(Q('tokenizer', 'Tokenizer 来源（id/路径）', '', required=True),
            Q('template_config', '编码配置', {}, type='dict'),
            Q('max_turns', '最大轮数', 8, type='int', minimum=1),
            Q('generation_config', '生成参数', {}, type='dict'),
            Q('rollout_config', '多轮配置', {}, type='dict')))

GROUP_RULE = D('group_rule', '分组采样规则', 'Resource', '完整组准入规则，不执行基线前向。',
    ports_out=(P('rule', 'GROUP_RULE', '准入规则'),),
    params=(Q('resample', '重采规则', 'none', options=('none', 'zero_variance')),
            Q('kwargs', '准入参数', {}, type='dict')))

TOOLSET = D('toolset', '工具集', 'Resource', '实际插件来源；运行实现按环境 lease 绑定，声明即插件模块内的实例。',
    ports_out=(P('tools', 'TOOL', '工具插件'),),
    params=(Q('plugins', '插件来源', [], type='list', item_type='str'),))
SANDBOX_ENV = D('sandbox_env', '沙箱环境', 'Resource', '原生环境池与 lease 生命周期。',
    ports_out=(P('env', 'ENV', '环境池'),), params=(Q('sandbox_config', '环境配置', {}, type='dict'),), owner_kind='ENV')
POLICY_ENDPOINT = D('policy_endpoint', '本地推理服务', 'Resource', '复用常驻 sampler；只管理本会话拥有的应用。',
    ports_in=(P('sampler', 'SAMPLER', '服务引擎'),),
    ports_out=(P('endpoint', 'ENDPOINT', '就绪端点'),),
    params=(Q('port', '端口', 8000, type='int', minimum=1, maximum=65535),
            Q('server_config', '服务配置', {}, type='dict')), effects=('serve',), owner_kind='ENDPOINT')
API_ENDPOINT = D('api_endpoint', '外部 API', 'Resource', '明确外部连接；凭据仅从运行环境读取。',
    ports_out=(P('endpoint', 'ENDPOINT', '外部 API 连接'),),
    params=(Q('base_url', '服务地址', '', required=True), Q('model', '服务模型', '', required=True),
            Q('api_key_env', '凭据环境变量名', ''),
            Q('timeout', '超时秒数', 60.0, type='float', minimum=0)), owner_kind='ENDPOINT')
CODE_COMPONENT = D('code_component', '自定义组件', 'Stage', '显式声明端口、生命周期与副作用的用户组件。',
    params=(Q('entry', '入口名', 'fn', required=True),
            Q('lifecycle', '生命周期', 'Stage', options=('Resource', 'Iterator', 'Stage', 'Task')),
            Q('ports_in', '输入声明', [], type='list', item_type='dict'),
            Q('ports_out', '输出声明', [], type='list', item_type='dict'),
            Q('effects', '副作用', [], type='list', item_type='str', unique_items=True),
            Q('config', '用户配置', {}, type='dict')), effects=('code',))

DATALOADER = D('dataloader', '数据加载器', 'Iterator', '常驻游标、epoch 与 RNG；完整消费后推进当前批次。',
    ports_in=(P('data', 'DATASET', '数据来源'), P('after', 'CTRL', '反馈后取下一批', optional=True)),
    ports_out=(P('batch', 'DATA', '当前原始批次', ('batch',)), P('source', 'SOURCE', '控制句柄', control=True)),
    params=(Q('batch_size', '批大小', 8, type='int', minimum=1),
            Q('shuffle', '打乱', True, type='bool'), Q('seed', '随机种子', 42, type='int', minimum=0),
            Q('num_workers', '加载进程数', 0, type='int', minimum=0),
            Q('drop_last', '丢弃尾批', False, type='bool')))
STREAMING_DRIVER = D('streaming_driver', '流式采样', 'Iterator', '采样准入、滞后、背压和批次确认控制面。',
    ports_in=(P('prompts', 'DATASET', '数据分布'),
              P('engine', 'SAMPLER', '具名采样资源', many=True, optional=True),
              P('episode', 'ROLLOUT', '多轮来源', optional=True),
              P('groups', 'GROUP_RULE', '完整组规则', optional=True),
              P('after', 'CTRL', '下一批反馈', optional=True)),
    ports_out=(P('samples', 'TRAJ', '当前轨迹', ('rollout',)), P('source', 'SOURCE', '发布与确认控制', control=True)),
    params=(Q('async_mode', '异步模式', 'none', options=('none', 'one_step_off', 'fully_async')),
            Q('max_staleness', '最大滞后', 1, type='int', minimum=0),
            Q('num_generations', '组宽', 8, type='int', minimum=1),
            Q('batch_size', '消费批大小', 8, type='int', minimum=1),
            Q('buffer_size', '缓冲上限', 64, type='int', minimum=1),
            Q('generation_config', '生成配置', {}, type='dict'),
            Q('driver_config', '控制面配置', {}, type='dict')))
CHALLENGER = D('challenger', '自动出题器', 'Iterator', '显式出题/解题来源； 一次提取同批两路训练轨迹。',
    ports_in=(P('backend', 'SAMPLER', '出题 backend'),
              P('solver_rollout', 'ROLLOUT', '解题多轮机制', optional=True),
              P('env', 'ENV', 'episode 环境'),
              P('api', 'ENDPOINT', '额外代理 API', optional=True),
              P('after', 'CTRL', '下一批反馈', optional=True)),
    ports_out=(P('questions', 'TRAJ', '同批出题轨迹', ('rollout',)),
               P('answers', 'TRAJ', '同批解题轨迹', ('rollout',)), P('source', 'SOURCE', '批次控制', control=True)),
    params=(Q('seed', '种子任务', '', required=True), Q('max_tasks', '每批题数', 8, type='int', minimum=1),
            Q('tokenizer', 'Tokenizer 来源（id/路径）', '', required=True),
            Q('template_config', '编码配置', {}, type='dict'),
            Q('challenger_config', '出题机制配置', {}, type='dict')))

GENERATE = D('generate', '批量生成', 'Stage', '仅生成 episode，评分与筛选独立执行。',
    ports_in=(P('prompts', 'DATA', '原始提示批次', ('batch',)),
              P('engine', 'SAMPLER', '生成引擎', optional=True),
              P('episode', 'ROLLOUT', '多轮引擎', optional=True)),
    ports_out=(P('samples', 'TRAJ', '生成轨迹', ('rollout',)),),
    params=(Q('num_generations', '组宽', 8, type='int', minimum=1, phase='call'),
            Q('generation_config', '生成参数', {}, type='dict', phase='call')),
    exclusive_inputs=(('engine', 'episode'),))
SCORE = D('score', '评分', 'Stage', '按参数声明的评分函数对齐打分列；缺失不截断也不补零。',
    ports_in=(P('samples', 'TRAJ', '待评分轨迹', ('rollout',)),),
    ports_out=(P('reward', 'REWARD', '结果评分', ('outcome',)),
               P('process', 'REWARD', '过程评分', ('process',), optional=True)),
    params=(Q('scorer', '注册评分函数名', 'math', required=True),
            Q('scorer_kwargs', '评分构造参数', {}, type='dict', phase='call')))
MODEL_FORWARD = D('model_forward', '模型前向', 'Stage', '按显式字段集合返回原生模型输出，调用方式不改变模型身份。',
    ports_in=(P('model', 'MODEL', '前向模型'),
              P('traj', 'TRAJ', '前向数据；内部按 tokenizer 参数编码', accepted_kinds=('DATA', 'BATCH'))),
    ports_out=tuple(P(key, 'SIGNAL', key, (key,), optional=True, enabled_by='outputs')
                    for key in ('logps', 'values', 'logits', 'topk', 'embeddings')),
    params=(Q('tokenizer', 'Tokenizer 来源（id/路径）', '', required=True),
            Q('template_config', '编码配置', {}, type='dict'),
            Q('outputs', '输出字段', ['logps'], type='list', item_type='str',
              item_options=('logps', 'values', 'logits', 'topk', 'embeddings'), required=True, unique_items=True, phase='call'),
            Q('disable_lora', '禁用 LoRA', False, type='bool', phase='call'),
            Q('temperature', '温度', 1.0, type='float', minimum=1e-12, phase='call'),
            Q('top_k', 'Top-k 宽度', 32, type='int', minimum=1, phase='call'),
            Q('micro_batch_size', '前向微批大小', 1, type='int', minimum=1, phase='call')),
    execution='remote')
REDUCE = D('reduce', '归约', 'Stage', '按显式轴、mask 和权重做数值归约，不访问模型。',
    ports_in=(P('signal', 'SIGNAL', '数值输入', accepted_kinds=('ADV',)),
              P('mask', 'SIGNAL', '明确 mask', optional=True), P('weights', 'SIGNAL', '数值权重', optional=True)),
    ports_out=(P('out', 'SIGNAL', '数学归约结果', profile_param='output_profile'),),
    params=(Q('op', '算子', 'mean', options=('sum', 'mean', 'variance', 'logsumexp'), phase='call'),
            Q('axis', '归约轴', 'token', options=('token', 'row', 'group'), phase='call'),
            Q('output_profile', '结果 Profile', 'row', options=('row', 'scalar'), phase='call')),
    execution='remote')
REWARD_SHAPE = D('reward_shape', '奖励整形', 'Stage', 'r - baseline - beta*KL；KL 必须两路前向或显式已计算数值。',
    ports_in=(P('reward', 'REWARD', '原评分', ('outcome', 'process')),
              P('logps', 'SIGNAL', '被比较 logps', ('logps',), optional=True),
              P('reference', 'SIGNAL', '比较基准 logps', ('logps',), optional=True),
              P('baseline', 'SIGNAL', '明确基线', ('row', 'scalar'), optional=True)),
    ports_out=(P('reward', 'REWARD', '整形奖励', profile_param='profile'),),
    params=(Q('beta', 'KL 系数', 0.0, type='float', minimum=0, phase='call'),
            Q('kl', 'KL 估计', 'none', options=('none', 'k1', 'k2', 'k3'), phase='call'),
            Q('profile', '结果 Profile', 'outcome', options=('outcome', 'process'), phase='call')),
    required_when=tuple(('kl', method, ('logps', 'reference')) for method in ('k1', 'k2', 'k3')),
    execution='remote')
ADVANTAGE = D('advantage', '优势估计', 'Stage', '仅消费奖励、分组身份与显式 values/baseline；不执行前向。',
    ports_in=(P('reward', 'REWARD', '实际奖励', ('outcome', 'process')),
              P('values', 'SIGNAL', '值函数估计', ('values',), optional=True),
              P('baseline', 'SIGNAL', '显式基线', ('row', 'scalar'), optional=True),
              P('traj', 'TRAJ', '批次分组与 token 身份', ('rollout',))),
    ports_out=(P('adv', 'ADV', '优势', profile_param='profile'),
               P('returns', 'SIGNAL', '独立 returns', ('values',), optional=True)),
    params=(Q('estimator', '估计器', 'grpo', options=ADVANTAGE_OPTIONS, phase='call'),
            Q('profile', '结果 Profile', 'row', options=('row', 'token'), phase='call'),
            Q('gamma', '折扣', 1.0, type='float', minimum=0, maximum=1, phase='call'),
            Q('lam', 'GAE lambda', 1.0, type='float', minimum=0, maximum=1, phase='call'),
            Q('kwargs', '估计参数', {}, type='dict', phase='call')),
    required_when=(('estimator', 'gae', ('values',)), ('estimator', 'remax', ('baseline',))),
    execution='remote')
SIGNAL_COMBINE = D('signal_combine', '数值组合', 'Stage', '对齐通道的加权和或概率混合；logps 概率混合使用 logsumexp。',
    ports_in=(P('signals', 'SIGNAL', '具名对齐数值', many=True, accepted_kinds=('ADV',)),),
    ports_out=(P('out', 'SIGNAL', '与输入数学类型一致', mirror_input='signals'),),
    params=(Q('mode', '运算', 'weighted_sum', options=('weighted_sum', 'probability_mixture'), phase='call'),
            Q('weights', '逐 slot 权重', {}, type='dict', phase='call')),
    execution='remote')
SELECT = D('select', '样本筛选', 'Stage', '一次选择索引应用全部对齐载荷；成对与完整组不拆散。',
    ports_in=(P('batch', 'TRAJ', '原始对齐数据', optional=False, accepted_kinds=('DATA', 'BATCH')),
              P('reward', 'REWARD', '筛选评分', ('outcome',)),
              P('signals', 'SIGNAL', '具名同步筛选输入', many=True, optional=True, accepted_kinds=('ADV',))),
    ports_out=(P('batch', 'TRAJ', '同一次选择后的数据', mirror_input='batch'),
               P('reward', 'REWARD', '同一次选择后的评分', mirror_input='reward'),
               P('signals', 'SIGNAL', '逐 slot 同步输出', optional=True, mirror_input='signals')),
    params=(Q('select', '选择规则', 'threshold', options=('threshold', 'top_k', 'best_of_n'), phase='call'),
            Q('threshold', '阈值', 0.0, type='float', phase='call'),
            Q('top_k', '保留数', 1, type='int', minimum=1, phase='call')))
TRAIN_STEP = D('train_step', '训练步骤', 'Stage', '真实 forward_backward 与 optimizer step；loss、优化器与编码是参数，不负责保存和发布 cadence。',
    ports_in=(P('model', 'MODEL', '实际更新模型', access='write'),
              P('batch', 'TRAJ', '待训练数据；轨迹在内部按 tokenizer 参数编码', accepted_kinds=('DATA', 'BATCH')),
              P('arguments', 'SIGNAL', 'slot 是 loss 实际参数名', many=True, optional=True, accepted_kinds=('ADV',)),
              P('after', 'CTRL', '正向完成依赖', optional=True)),
    ports_out=(P('step', 'STEP', '真实训练回执', ('step',)), P('model', 'MODEL', '训练后的模型引用'),
               P('metric', 'METRIC', '小型摘要', ('summary',)),
               P('then', 'CTRL', '更新完成')),
    params=(Q('tokenizer', 'Tokenizer 来源（id/路径）', '', required=True),
            Q('template_config', '编码配置', {}, type='dict'),
            Q('loss', '损失函数', 'cross_entropy', options=LOSS_OPTIONS),
            Q('loss_kwargs', '损失构造参数', {}, type='dict'),
            Q('optim', '优化器族', 'adamw'),
            Q('lr', '学习率', 1e-5, type='float', minimum=0),
            Q('lr_scheduler_type', '调度器', 'cosine'),
            Q('num_training_steps', 'Scheduler 总步数', 1000, type='int', minimum=1),
            Q('warmup_ratio', '预热比例', 0.05, type='float', minimum=0, maximum=1),
            Q('weight_decay', '权重衰减', 0.0, type='float', minimum=0),
            Q('optimizer_config', '其他原生更新参数', {}, type='dict'),
            Q('micro_batch_size', '微批大小', 1, type='int', minimum=1, phase='call'),
            Q('gradient_accumulation_steps', '梯度累积步数', 1, type='int', minimum=1, phase='call'),
            Q('max_grad_norm', '梯度裁剪', 1.0, type='float', minimum=0, phase='call')),
    effects=('train',))
WEIGHT_SYNC = D('weight_sync', '权重同步', 'Stage', '经原生分桶传输；流式发布登记、commit 后确认，不虚报 pending 完成。',
    ports_in=(P('model', 'MODEL', '权重来源；未接线时沿连线共享的作用域唯一解析', scope=True),
              P('targets', 'SAMPLER', '明确接收者', many=True, access='write'),
              P('source', 'SOURCE', '流式控制句柄，不触发 take', optional=True, control=True),
              P('step', 'STEP', '真实更新结果', ('step',), optional=True),
              P('after', 'CTRL', '完成依赖', optional=True)),
    ports_out=(P('sync', 'SYNC', '完成确认', ('outcome',)), P('then', 'CTRL', '同步已完成')),
    params=(Q('cadence', '同步间隔', 1, type='int', minimum=1, phase='call'),
            Q('sync_config', '原生同步配置', {}, type='dict')), effects=('sync',))
WEIGHT_AVERAGE = D('weight_average', '参数平均', 'Stage', 'source/target 的 copy 或 EMA；目标角色不写死，原生设备间传输。',
    ports_in=(P('source', 'MODEL', '来源模型；未接线时沿连线共享的作用域唯一解析', scope=True),
              P('target', 'MODEL', '目标模型', access='write'),
              P('step', 'STEP', '明确更新来源', ('step',))),
    ports_out=(P('sync', 'SYNC', '实际传输确认', ('outcome',)),),
    params=(Q('mode', '方式', 'ema', options=('copy', 'ema'), phase='call'),
            Q('alpha', 'EMA 系数', 0.9, type='float', minimum=0, maximum=1, phase='call')), effects=('sync',))
CHECKPOINT = D('checkpoint', '保存检查点', 'Stage', '实际模型导出与完整训练快照分 profile；完整快照仅在安全边界保存。',
    ports_in=(P('model', 'MODEL', '快照模型；未接线时沿连线共享的作用域唯一解析', scope=True),
              P('step', 'STEP', '训练状态来源', ('step',), optional=True),
              P('source', 'SOURCE', '循环游标来源', optional=True, control=True),
              P('after', 'CTRL', '完成依赖', optional=True)),
    ports_out=(P('artifact', 'ARTIFACT', '实际写出产物', profile_param='profile'),),
    params=(Q('name', '保存名', 'checkpoint', phase='call'),
            Q('profile', '保存内容', 'model', options=('model', 'training'), phase='call'),
            Q('cadence', '保存间隔', 1, type='int', minimum=1, phase='call')),
    required_when=(('profile', 'training', ('step', 'source')),), effects=('save',))

MERGE = D('merge', '权重合并', 'Task', '复用实体 MergeComponent，输出真实模型产物。',
    ports_in=(P('artifact', 'ARTIFACT', '原权重', ('model',)),), ports_out=(P('model', 'ARTIFACT', '合并权重', ('model',)),),
    params=(Q('output_dir', '输出目录', 'output/merged'), Q('merge_config', '合并配置', {}, type='dict')), effects=('save',))
QUANTIZE = D('quantize', '权重量化', 'Task', '复用实体 QuantizeComponent，校准来源明确。',
    ports_in=(P('artifact', 'ARTIFACT', '原权重', ('model',)), P('calib', 'DATASET', '校准来源', optional=True)),
    ports_out=(P('model', 'ARTIFACT', '量化权重', ('model',)),),
    params=(Q('output_dir', '输出目录', 'output/quantized'), Q('quantize_config', '量化配置', {}, type='dict')), effects=('save',))
INFER = D('infer', '批量推理', 'Task', '从权重来源重建推理引擎，推理数据必须明确。',
    ports_in=(P('model', 'MODEL', '会话模型（回读其权重来源）', optional=True),
              P('artifact', 'ARTIFACT', '权重来源', ('model',), optional=True),
              P('data', 'DATASET', ' 实际推理数据')),
    ports_out=(P('predictions', 'DATA', '预测行', ('raw',)), P('artifact', 'ARTIFACT', '预测产物', ('dataset',), optional=True)),
    params=(Q('infer_config', '推理配置（sampler 为引擎，含 template_config 等）', {}, type='dict'),),
    exclusive_inputs=(('model', 'artifact'),), effects=('save',))
EVAL = D('eval', '评测', 'Task', '复用 Evaluator 与公共 runner，不关闭借用的会话采样资源。',
    ports_in=(P('sampler', 'SAMPLER', '采样来源', optional=True), P('endpoint', 'ENDPOINT', 'API 来源', optional=True),
              P('data', 'DATASET', '明确评测数据')),
    ports_out=(P('metrics', 'METRIC', '真实评测摘要', ('summary',)), P('artifact', 'ARTIFACT', '真实报告', ('report',), optional=True)),
    params=(Q('tokenizer', 'Tokenizer 来源（id/路径）', '', required=True),
            Q('task_config', 'Evaluator 配置', {}, type='dict')), exclusive_inputs=(('sampler', 'endpoint'),))
DEPLOY = D('deploy', '部署服务', 'Task', '经明确授权启动服务并等待就绪，命令线程不被阻塞。',
    ports_in=(P('model', 'ARTIFACT', '权重来源', ('model',)),), ports_out=(P('endpoint', 'ENDPOINT', '就绪服务'),),
    params=(Q('deploy_config', '服务配置', {}, type='dict'),), effects=('external', 'serve'), owner_kind='ENDPOINT')
HUB_PUSH = D('hub_push', '发布模型', 'Task', '明确授权后调用公共 Hub API，返回实际发布结果。',
    ports_in=(P('model', 'ARTIFACT', '权重来源', ('model',)),),
    ports_out=(P('artifact', 'ARTIFACT', '实际发布结果', ('publication',)),),
    params=(Q('hub_model_id', '目标仓库', '', required=True),
            Q('hub', 'Hub', 'ModelScope', options=('ModelScope', 'HuggingFace')),
            Q('private', '私有仓库', True, type='bool'), Q('revision', '版本', 'main'),
            Q('commit_message', '提交说明', 'webui workflow push'), Q('token_env', '凭据环境变量名', '')),
    effects=('external',))

REGISTRY = {descriptor.type: descriptor for descriptor in (
    MODEL, DATASET, SAMPLER, ROLLOUT, GROUP_RULE,
    TOOLSET, SANDBOX_ENV, POLICY_ENDPOINT, API_ENDPOINT, CODE_COMPONENT,
    DATALOADER, STREAMING_DRIVER, CHALLENGER, GENERATE, SCORE,
    MODEL_FORWARD, REDUCE, REWARD_SHAPE, ADVANTAGE, SIGNAL_COMBINE, SELECT,
    TRAIN_STEP, WEIGHT_SYNC, WEIGHT_AVERAGE, CHECKPOINT, MERGE, QUANTIZE,
    INFER, EVAL, DEPLOY, HUB_PUSH)}


def validate_registry() -> None:
    from .schema import CHANNEL_SCHEMAS
    for descriptor in REGISTRY.values():
        params = {param.name: param for param in descriptor.params}
        for param in descriptor.params:
            if param.phase not in ('construct', 'call'):
                raise ValueError(f'{descriptor.type}.{param.name}: 非法参数阶段')
            if not param.required:
                param.validate(param.default)
        inputs = {port.key for port in descriptor.ports_in}
        for port in descriptor.ports_in + descriptor.ports_out:
            for kind in (port.kind, *port.accepted_kinds):
                if kind not in CHANNEL_SCHEMAS:
                    raise ValueError(f'{descriptor.type}.{port.key}: 未声明通道 {kind}')
            for profile in port.profiles:
                CHANNEL_SCHEMAS[port.kind].profile(profile)
            if port.access not in ('read', 'write'):
                raise ValueError(f'{descriptor.type}.{port.key}: 非法访问模式')
            for name in (port.enabled_by, port.profile_param):
                if name and name not in params:
                    raise ValueError(f'{descriptor.type}.{port.key}: 未声明动态参数 {name}')
            if port.mirror_input and port.mirror_input not in inputs:
                raise ValueError(f'{descriptor.type}.{port.key}: 未声明投射输入')
        for group in descriptor.exclusive_inputs:
            if not set(group) <= inputs:
                raise ValueError(f'{descriptor.type}: 互斥输入不存在')
        for name, _value, ports in descriptor.required_when:
            if name not in params or not set(ports) <= inputs:
                raise ValueError(f'{descriptor.type}: 条件输入声明不完整')
