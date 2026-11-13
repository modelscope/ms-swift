# Copyright (c) ModelScope Contributors. All rights reserved.
"""通用组件 v5 注册表：数学职责、真实端口与类型化配置的唯一声明。"""
from __future__ import annotations

from .descriptor import KIND_ALLOWS, KIND_CATEGORIES, ComponentDescriptor, ParamSpec, PortSpec

REGISTRY_VERSION = 5
ADVANTAGE_OPTIONS = ('grpo', 'grpo_passk', 'gae', 'rloo', 'remax', 'gpg', 'opo',
                     'reinforce_plus_plus', 'reinforce_pp_baseline')
# Every name here must be resolvable by twinkle's torch_loss_mapping -- the loss component's
# ``kind`` constructs straight out of that registry (the PPO preset ships kind='ppo'), so this
# list is its webui-facing mirror, NOT the dev recipe's own loss mapping.
LOSS_OPTIONS = ('cross_entropy', 'channel', 'mse', 'chunked_cross_entropy', 'grpo', 'ppo',
                'ppo_value', 'gspo', 'sapo', 'cispo', 'bnpo', 'dr_grpo', 'real', 'gpg',
                'gkd', 'opsd', 'mopd', 'kl_cov', 'clip_cov', 'geo_mean', 'dpo', 'kto',
                'cpo', 'orpo', 'simpo', 'dro', 'dppo_kl', 'dppo_tv', 'infonce',
                'online_contrastive', 'cosine_similarity')
P = PortSpec
Q = ParamSpec
D = ComponentDescriptor

# Card allocation: a GPU component declares either a card count or an explicit
# card list (ids win; setting both is a construction error). An empty spec
# defaults to one card. The session plans one named DeviceGroup per declaring
# component from these specs -- selecting N cards really occupies N cards.
GPU_PARAMS = (
    Q('gpu_count', '占卡数', 1, type='int', minimum=1,
      doc='声明占用几张卡；与卡列表同时设置是构造错误'),
    Q('gpu_ids', '卡列表', [], type='list', item_type='int', unique_items=True,
      doc='显式卡编号列表；非空时优先于占卡数'))

MODEL = D('model', '模型', 'Resource', '唯一参数化函数 f_theta；独立冻结模型必须独立构造。',
    ports_in=(P('artifact', 'ARTIFACT', '模型权重来源', ('model', 'training'), optional=True),),
    ports_out=(P('model', 'MODEL', '唯一模型对象'),),
    params=(Q('model', '模型来源', '', '模型 ID/路径；接入 artifact 时由真实产物提供'),
            Q('train_type', '训练类型', 'lora', options=('lora', 'full')),
            Q('freeze', '冻结参数', False, type='bool'),
            Q('backend', '训练后端', 'hf', options=('hf', 'megatron', 'unsloth'),
              doc='hf=transformers 后端；megatron 经 bridge 构建，接口等价'),
            Q('parallel_spec', '并行规格', '',
              doc='tp/pp/dp 显式规格；缺省由占卡数单维推导（tp=卡数）'),
            *GPU_PARAMS,
            Q('model_config', '模型配置', {}, type='dict'),
            Q('train_config', '训练参数选择', {}, type='dict'),
            Q('tuner_config', '调参配置', {}, type='dict'),
            Q('distributed_config', '完整放置配置', {}, type='dict'),
            Q('megatron_config', 'Megatron 配置', {}, type='dict',
              doc='仅 backend=megatron 时生效的 MegatronConfig 字段'),
            Q('moe_config', 'MoE 配置', {}, type='dict',
              doc='仅 backend=megatron 时生效的 MoEConfig 字段')),
    owner_kind='MODEL', surface='resource')

DATASET = D('dataset', '数据集', 'Resource', '原始数据分布与预处理配置，不进入循环。',
    ports_out=(P('data', 'DATASET', '原生数据来源'),),
    params=(Q('dataset', '数据来源', [], type='list', item_type='str', required=True),
            Q('dataset_config', '预处理配置', {}, type='dict')), surface='resource')

SAMPLER = D('sampler', '采样器', 'Resource', '常驻生成引擎；权重初始来源与编码配置是参数，同步由 weight_sync 表达。',
    ports_out=(P('sampler', 'SAMPLER', '生成引擎'),),
    params=(Q('model', '模型来源（id/路径，同时是 tokenizer 来源）', '', required=True),
            Q('engine', '引擎', 'vllm', options=('vllm', 'sglang', 'torch')),
            Q('parallel_spec', '并行规格', '',
              doc='tp/pp/dp 显式规格；缺省由占卡数单维推导（tp=卡数）'),
            *GPU_PARAMS,
            Q('engine_args', '引擎参数', {}, type='dict'),
            Q('model_config', '模型加载配置', {}, type='dict'),
            Q('template_config', '编码配置', {}, type='dict')), owner_kind='SAMPLER', surface='resource')

ROLLOUT = D('rollout', '多轮生成器', 'Resource', '在环境 lease 内执行多轮生成，不包含评分与优势估计。',
    ports_in=(P('sampler', 'SAMPLER', '生成 backend'), P('tools', 'TOOL', '工具插件与声明', optional=True),
              P('env', 'ENV', '环境 lease', optional=True)),
    ports_out=(P('rollout', 'ROLLOUT', 'episode 生成机制'),),
    params=(Q('tokenizer', 'Tokenizer 来源（id/路径，与采样器一致）', '', required=True),
            Q('model_config', '模型加载配置', {}, type='dict',
              doc='tokenizer 加载所需的身份字段（如本地目录的 model_type）；与采样器节点自身一致'),
            Q('template_config', '编码配置', {}, type='dict'),
            Q('max_turns', '最大轮数', 8, type='int', minimum=1),
            Q('generation_config', '生成参数', {}, type='dict'),
            Q('rollout_config', '多轮配置', {}, type='dict')), surface='resource')

GROUP_RULE = D('group_rule', '分组采样规则', 'Resource', '完整组准入规则，不执行基线前向。',
    ports_out=(P('rule', 'GROUP_RULE', '准入规则'),),
    params=(Q('resample', '重采规则', 'none', options=('none', 'zero_variance')),
            Q('kwargs', '准入参数', {}, type='dict')), surface='algorithm')

LOSS_COMP = D('loss', '损失函数', 'Resource', 'L(theta)：策略更新目标；注册名或自定义公式的损失对象。',
    ports_out=(P('loss', 'LOSS', '损失对象'),),
    params=(Q('kind', '损失函数', 'cross_entropy', options=LOSS_OPTIONS),
            Q('loss_kwargs', '损失构造参数', {}, type='dict'),
            Q('code', '自定义公式', '',
              doc='用户公式 def loss(**lanes)->float；设置时忽略注册名')),
    surface='algorithm')
SCORER_COMP = D('scorer', '评分函数', 'Resource', 'r(tau)：规则评分函数；注册名或自定义代码的评分器。',
    ports_out=(P('scorer', 'SCORER', '评分器对象'),),
    params=(Q('kind', '注册评分函数名', 'math',
              doc='orms 插件注册表中的名字（math/format/accuracy 等；插件可运行时扩充，故不静态枚举）'),
            Q('scorer_kwargs', '评分构造参数', {}, type='dict'),
            Q('code', '自定义评分代码', '',
              doc='用户代码 def score(rows)->list[float]；设置时忽略注册名')),
    surface='algorithm')

TOOLSET = D('toolset', '工具集', 'Resource', '实际插件来源；运行实现按环境 lease 绑定，声明即插件模块内的实例。',
    ports_out=(P('tools', 'TOOL', '工具插件'),),
    params=(Q('plugins', '插件来源', [], type='list', item_type='str'),), surface='resource')
SANDBOX_ENV = D('sandbox_env', '沙箱环境', 'Resource', '原生环境池与 lease 生命周期；后端可换，消费面不变。',
    ports_out=(P('env', 'ENV', '环境池'),),
    params=(Q('kind', '沙箱后端', 'local', options=('local', 'agentenv', 'openenv', 'openenv_client'),
              doc='local=本机 subprocess；agentenv=AgentENV microVM；'
                  'openenv=进程内嵌 OpenEnv 环境；openenv_client=连运行中的 OpenEnv 服务'),
            Q('sandbox_config', '环境配置', {}, type='dict',
              doc='所选后端构造器的真实字段；未知名构造期拒绝'),
            Q('api_key_env', '凭据环境变量名', '',
              doc='仅 agentenv：构造期读取该环境变量作为 api_key；留空则由 SDK 读 E2B_API_KEY')),
    owner_kind='ENV', surface='resource')
API_ENDPOINT = D('api_endpoint', '外部 API', 'Resource', '明确外部连接；凭据仅从运行环境读取。',
    ports_out=(P('endpoint', 'ENDPOINT', '外部 API 连接'),),
    params=(Q('base_url', '服务地址', '', required=True), Q('model', '服务模型', '', required=True),
            Q('api_key_env', '凭据环境变量名', ''),
            Q('timeout', '超时秒数', 60.0, type='float', minimum=0)), owner_kind='ENDPOINT', surface='resource')
CODE_COMPONENT = D('code_component', '自定义组件', 'Stage', '显式声明端口、生命周期与副作用的用户组件。',
    params=(Q('entry', '入口名', 'fn', required=True),
            Q('lifecycle', '生命周期', 'Stage', options=('Resource', 'Iterator', 'Stage', 'Task')),
            Q('ports_in', '输入声明', [], type='list', item_type='dict'),
            Q('ports_out', '输出声明', [], type='list', item_type='dict'),
            Q('effects', '副作用', [], type='list', item_type='str', unique_items=True),
            Q('config', '用户配置', {}, type='dict')), effects=('code',), surface='algorithm')

DATALOADER = D('dataloader', '数据加载器', 'Iterator', '常驻游标、epoch 与 RNG；完整消费后推进当前批次。',
    ports_in=(P('data', 'DATASET', '数据来源'), P('after', 'CTRL', '反馈后取下一批', optional=True)),
    ports_out=(P('batch', 'DATA', '当前原始批次', ('batch',)), P('source', 'SOURCE', '控制句柄', control=True)),
    params=(Q('batch_size', '批大小', 8, type='int', minimum=1),
            Q('source', '数据面', 'prompts', options=('prompts', 'rows'),
              doc='prompts=提示面（剥掉末轮 assistant，生成侧用）；rows=完整训练行（监督训练用）'),
            Q('shuffle', '打乱', True, type='bool'), Q('seed', '随机种子', 42, type='int', minimum=0),
            Q('num_workers', '加载进程数', 0, type='int', minimum=0),
            Q('drop_last', '丢弃尾批', False, type='bool')), surface='runtime')
STREAMING_DRIVER = D('streaming_driver', '流式采样', 'Iterator', '采样准入、滞后、背压和批次确认控制面；接线评分器时逐 episode 打分，交付合并具名列。',
    ports_in=(P('prompts', 'SOURCE', '提示流来源（dataloader 的 source 控制口；原生 loader 在其上）', control=True),
              P('engine', 'SAMPLER', '具名采样资源', many=True, optional=True),
              P('episode', 'ROLLOUT', '多轮来源', optional=True),
              P('scorers', 'SCORER', '准入前评分绑定：组规则与交付列共用同一 scorer 身份', many=True, optional=True),
              P('groups', 'GROUP_RULE', '完整组规则', optional=True),
              P('after', 'CTRL', '下一批反馈', optional=True)),
    ports_out=(P('traj', 'TRAJ', '当前轨迹（评分器接线时携带具名评分列）', ('rollout', 'scored')),
               P('source', 'SOURCE', '批次确认控制', control=True)),
    params=(Q('tokenizer', 'Tokenizer 来源（id/路径，与采样器一致）', '', required=True),
            Q('model_config', '模型加载配置', {}, type='dict',
              doc='tokenizer 加载所需的身份字段（如本地目录的 model_type）；与采样器节点自身一致'),
            Q('template_config', '编码配置', {}, type='dict'),
            Q('async_mode', '异步模式', 'none', options=('none', 'one_step_off', 'fully_async')),
            Q('max_staleness', '最大滞后', 1, type='int', minimum=0),
            Q('num_generations', '组宽', 8, type='int', minimum=1),
            Q('generation_config', '生成配置', {}, type='dict'),
            Q('rl_config', 'RL 配置', {}, type='dict'),
            Q('rollout_config', 'Rollout 配置', {}, type='dict'),
            Q('train_batch_size', '训练批大小', 8, type='int', minimum=1),
            Q('gradient_accumulation_steps', '梯度累积', 1, type='int', minimum=1),
            Q('num_train_epochs', '训练轮数', 1.0, type='float'),
            Q('max_steps', '总步数', 0, type='int', minimum=0)), surface='runtime')
CHALLENGER = D('challenger', '自动出题器', 'Iterator', '显式出题/解题来源； 一次提取同批两路训练轨迹。',
    ports_in=(P('backend', 'SAMPLER', '出题 backend'),
              P('solver_rollout', 'ROLLOUT', '解题多轮机制', optional=True),
              P('env', 'ENV', 'episode 环境'),
              P('api', 'ENDPOINT', '额外代理 API', optional=True),
              P('after', 'CTRL', '下一批反馈', optional=True)),
    ports_out=(P('questions', 'TRAJ', '同批出题轨迹', ('rollout',)),
               P('answers', 'TRAJ', '同批解题轨迹', ('rollout',)), P('source', 'SOURCE', '批次控制', control=True)),
    params=(Q('seed', '种子任务', '', required=True), Q('max_tasks', '每批题数', 8, type='int', minimum=1),
            Q('tokenizer', 'Tokenizer 来源（id/路径，与采样器一致）', '', required=True),
            Q('model_config', '模型加载配置', {}, type='dict',
              doc='tokenizer 加载所需的身份字段（如本地目录的 model_type）；与采样器节点自身一致'),
            Q('template_config', '编码配置', {}, type='dict'),
            Q('challenger_config', '出题机制配置', {}, type='dict')), surface='algorithm')

GENERATE = D('generate', '批量生成', 'Stage', '仅生成 episode，评分与筛选独立执行。',
    ports_in=(P('prompts', 'DATA', '原始提示批次', ('batch',)),
              P('engine', 'SAMPLER', '生成引擎', optional=True),
              P('episode', 'ROLLOUT', '多轮引擎', optional=True)),
    ports_out=(P('traj', 'TRAJ', '生成轨迹', ('rollout',)),),
    params=(Q('num_generations', '组宽', 8, type='int', minimum=1, phase='call'),
            Q('generation_config', '生成参数', {}, type='dict', phase='call')),
    exclusive_inputs=(('engine', 'episode'),), surface='algorithm')
SCORE = D('score', '评分', 'Stage', '对接线的具名评分器打分，具名列写回轨迹；缺失不截断也不补零。',
    ports_in=(P('traj', 'TRAJ', '待评分轨迹', ('rollout', 'scored')),
              P('scorers', 'SCORER', '具名评分器', many=True)),
    ports_out=(P('traj', 'TRAJ', '携带具名评分列的轨迹', ('scored',)),
               P('metric', 'METRIC', '本批评分摘要', ('summary',))),
    params=(), surface='algorithm')
MODEL_FORWARD = D('model_forward', '模型前向', 'Stage', '按显式字段集合的一次独立前向；输出是完整 ModelOutput，logps/values 走响应 frame。',
    ports_in=(P('model', 'MODEL', '前向模型'),
              P('traj', 'TRAJ', '前向数据；内部按 tokenizer 参数编码', accepted_kinds=('DATA', 'BATCH'))),
    ports_out=(P('out', 'MODEL_OUTPUT', '完整前向结果；字段由 outputs 参数声明'),),
    params=(Q('tokenizer', 'Tokenizer 来源（id/路径，与模型一致）', '', required=True),
            Q('model_config', '模型加载配置', {}, type='dict',
              doc='tokenizer 加载所需的身份字段（如本地目录的 model_type）；与模型节点自身一致'),
            Q('template_config', '编码配置', {}, type='dict'),
            Q('outputs', '输出字段', ['logps'], type='list', item_type='str',
              item_options=('logps', 'values', 'logits', 'embeddings'), required=True, unique_items=True, phase='call',
              doc='logps/values 请求响应 frame；logits/embeddings 保持完整输入 frame'),
            Q('disable_lora', '禁用 LoRA', False, type='bool', phase='call'),
            Q('temperature', '温度', 1.0, type='float', minimum=1e-12, phase='call'),
            Q('micro_batch_size', '前向微批大小', 1, type='int', minimum=1, phase='call')),
    execution='remote', surface='algorithm')
PRIVILEGED_VIEW = D('privileged_view', '特权提示', 'Stage', '按行内 teacher_prompt 重编码教师视图；无该列的行原样传递，策略主轨迹不回写。',
    ports_in=(P('traj', 'TRAJ', '携带 teacher_prompt 列的轨迹', ('rollout', 'scored')),),
    ports_out=(P('traj', 'TRAJ', '教师前向视图', ('rollout', 'scored')),),
    params=(Q('tokenizer', 'Tokenizer 来源（id/路径，与教师一致）', '', required=True),
            Q('model_config', '模型加载配置', {}, type='dict',
              doc='tokenizer 加载所需的身份字段（如本地目录的 model_type）'),
            Q('template_config', '编码配置', {}, type='dict')),
    execution='remote', surface='algorithm')
ADVANTAGE = D('advantage', '优势估计', 'Stage', '轨迹与前向 lane 进、训练目标列回轨迹；不持模型、不执行前向。',
    ports_in=(P('traj', 'TRAJ', '携带具名评分列的轨迹', ('scored',)),
              P('ref_out', 'MODEL_OUTPUT', '参考前向（kl_in_reward/损失所需）', optional=True),
              P('teacher_out', 'MODEL_OUTPUT', '教师前向（重加权/SDAR）', optional=True),
              P('old_out', 'MODEL_OUTPUT', '更新前策略前向（重要性采样/诊断）', optional=True),
              P('value_out', 'MODEL_OUTPUT', '更新前 critic 前向（gae）', optional=True)),
    ports_out=(P('traj', 'TRAJ', '携带训练目标列的轨迹', ('train_ready',)),
               P('metric', 'METRIC', '估计摘要', ('summary',))),
    params=(Q('estimator', '估计器', 'grpo', options=ADVANTAGE_OPTIONS, phase='call'),
            Q('num_generations', '分组宽度', 8, type='int', minimum=1, phase='call',
              doc='必须与上游生成/驱动节点一致；组内估计按完整组进行'),
            Q('kwargs', '估计器构造参数', {}, type='dict', phase='call',
              doc='Advantage 构造旋钮（gamma/gae_lambda/normalize/kl_coef/beta/kl_in_reward 等）；未被该估计器读取的旋钮会被拒绝')),
    required_when=(('estimator', 'gae', ('value_out',)),),
    execution='remote', surface='algorithm')
# 'weighted_sum' is a weighted MEAN -- the implementation divides by the total weight, so the
# weights act as mixture coefficients (scale-invariant) the way the sibling probability_mixture
# mode's are. The descriptor text says so; the option key keeps its name (it is the graph
# parameter surface, and mixture semantics is what existing pinned behavior expects).
SIGNAL_COMBINE = D('signal_combine', '数值组合', 'Stage', '对齐通道的加权平均（权重为混合系数，同比缩放结果不变）或概率混合；logps 概率混合使用 logsumexp。',
    ports_in=(P('signals', 'MODEL_OUTPUT', '具名对齐数值', many=True),),
    ports_out=(P('out', 'MODEL_OUTPUT', '与输入数学类型一致', mirror_input='signals'),),
    params=(Q('mode', '运算', 'weighted_sum', options=('weighted_sum', 'probability_mixture'), phase='call'),
            Q('weights', '逐 slot 权重', {}, type='dict', phase='call')),
    execution='remote', surface='algorithm')
SELECT = D('select', '样本筛选', 'Stage', '一次选择索引应用全部对齐载荷；成对与完整组不拆散，奖励随轨迹保留。',
    ports_in=(P('batch', 'TRAJ', '原始对齐数据', optional=False, accepted_kinds=('DATA', 'BATCH')),
              P('signals', 'MODEL_OUTPUT', '具名同步筛选输入', many=True, optional=True)),
    ports_out=(P('batch', 'TRAJ', '同一次选择后的数据', mirror_input='batch'),
               P('signals', 'MODEL_OUTPUT', '逐 slot 同步输出', optional=True, mirror_input='signals'),
               P('metric', 'METRIC', '选择摘要（保留/丢弃计数）', ('summary',))),
    params=(Q('select', '选择规则', 'threshold', options=('threshold', 'top_k', 'best_of_n'), phase='call'),
            Q('threshold', '阈值', 0.0, type='float', phase='call'),
            Q('top_k', '保留数', 1, type='int', minimum=1, phase='call'),
            # Declared so best_of_n/top_k can actually group at the run's width: the component
            # reads it, but an undeclared key is rejected by validate_params -- the grouping was
            # pinned at the default 8 and silently mis-grouped any wider/narrower generate node.
            # Must match the upstream generate/driver num_generations.
            Q('num_generations', '分组宽度', 8, type='int', minimum=1, phase='call',
              doc='必须与上游生成/驱动节点一致')),
    surface='algorithm')
TRAIN_STEP = D('train_step', '训练步骤', 'Stage', '真实 forward_backward 与 optimizer step；训练目标从轨迹 lift，显式 lane 只作补充绑定。',
    ports_in=(P('model', 'MODEL', '实际更新模型；未接线时沿连线共享的作用域唯一解析', access='write', scope=True),
              P('loss', 'LOSS', '损失对象'),
              P('batch', 'TRAJ', '待训练数据；标准字段由模型侧 lift', accepted_kinds=('DATA', 'BATCH')),
              P('reference', 'MODEL_OUTPUT', '参考前向（DPO/KTO 参考信号）', optional=True),
              P('teacher', 'MODEL_OUTPUT', '教师前向（蒸馏教师信号）', optional=True),
              P('lanes', 'MODEL_OUTPUT', '显式字段绑定：slot 是字段名，自定义公式 loss 的补充 lane', many=True, optional=True),
              P('after', 'CTRL', '正向完成依赖', optional=True)),
    ports_out=(P('step', 'STEP', '真实训练回执', ('step',)), P('model', 'MODEL', '训练后的模型引用'),
               P('metric', 'METRIC', '小型摘要', ('summary',)),
               P('then', 'CTRL', '更新完成')),
    params=(Q('tokenizer', 'Tokenizer 来源（id/路径）', '', required=True),
            Q('model_config', '模型加载配置', {}, type='dict',
              doc='tokenizer 加载所需的身份字段（如本地目录的 model_type）；与模型节点自身一致'),
            Q('template_config', '编码配置', {}, type='dict'),
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
    effects=('train',), surface='algorithm')
WEIGHT_SYNC = D('weight_sync', '权重同步', 'Stage', '经原生分桶传输；同步完成后才确认，不虚报 pending 完成。',
    ports_in=(P('model', 'MODEL', '权重来源；未接线时沿连线共享的作用域唯一解析', scope=True),
              P('targets', 'SAMPLER', '明确接收者', many=True, access='write'),
              P('source', 'SOURCE', '流式控制句柄，不触发 take', optional=True, control=True),
              P('step', 'STEP', '真实更新结果', ('step',), optional=True),
              P('after', 'CTRL', '完成依赖', optional=True)),
    ports_out=(P('sync', 'SYNC', '完成确认', ('outcome',)), P('then', 'CTRL', '同步已完成')),
    params=(Q('cadence', '同步间隔', 1, type='int', minimum=1, phase='call'),
            Q('sync_config', '原生同步配置', {}, type='dict')), effects=('sync',), surface='runtime')
WEIGHT_AVERAGE = D('weight_average', '参数平均', 'Stage', 'source/target 的 copy 或 EMA；目标角色不写死，原生设备间传输。',
    ports_in=(P('source', 'MODEL', '来源模型；未接线时沿连线共享的作用域唯一解析', scope=True),
              P('target', 'MODEL', '目标模型', access='write'),
              P('step', 'STEP', '明确更新来源', ('step',))),
    ports_out=(P('sync', 'SYNC', '实际传输确认', ('outcome',)),),
    params=(Q('mode', '方式', 'ema', options=('copy', 'ema'), phase='call'),
            Q('alpha', 'EMA 系数', 0.9, type='float', minimum=0, maximum=1, phase='call')), effects=('sync',),
    surface='runtime')
CHECKPOINT = D('checkpoint', '保存检查点', 'Stage', '实际模型导出与完整训练快照分 profile；完整快照仅在安全边界保存。',
    ports_in=(P('model', 'MODEL', '快照模型；未接线时沿连线共享的作用域唯一解析', scope=True),
              P('step', 'STEP', '训练状态来源', ('step',), optional=True),
              P('source', 'SOURCE', '循环游标来源', optional=True, control=True),
              P('after', 'CTRL', '完成依赖', optional=True)),
    ports_out=(P('artifact', 'ARTIFACT', '实际写出产物', profile_param='profile'),),
    params=(Q('name', '保存名', 'checkpoint', phase='call'),
            Q('profile', '保存内容', 'model', options=('model', 'training'), phase='call'),
            Q('output_dir', '输出目录', 'output/checkpoints', phase='call'),
            Q('cadence', '保存间隔', 1, type='int', minimum=1, phase='call')),
    required_when=(('profile', 'training', ('step', 'source')),), effects=('save',), surface='runtime')

MERGE = D('merge', '权重合并', 'Task', '复用实体 MergeComponent，输出真实模型产物。',
    ports_in=(P('artifact', 'ARTIFACT', '原权重', ('model',)),), ports_out=(P('model', 'ARTIFACT', '合并权重', ('model',)),),
    params=(Q('output_dir', '输出目录', 'output/merged'), Q('merge_config', '合并配置', {}, type='dict')), effects=('save',),
    surface='artifact')
QUANTIZE = D('quantize', '权重量化', 'Task', '复用实体 QuantizeComponent，校准来源明确。',
    ports_in=(P('artifact', 'ARTIFACT', '原权重', ('model',)), P('calib', 'DATASET', '校准来源', optional=True)),
    ports_out=(P('model', 'ARTIFACT', '量化权重', ('model',)),),
    params=(Q('output_dir', '输出目录', 'output/quantized'), Q('quantize_config', '量化配置', {}, type='dict')), effects=('save',),
    surface='artifact')
INFER = D('infer', '批量推理', 'Task', '从权重来源重建推理引擎，推理数据必须明确。',
    ports_in=(P('model', 'MODEL', '会话模型（回读其权重来源）', optional=True),
              P('artifact', 'ARTIFACT', '权重来源', ('model',), optional=True),
              P('data', 'DATASET', ' 实际推理数据')),
    ports_out=(P('predictions', 'DATA', '预测行', ('raw',)), P('artifact', 'ARTIFACT', '预测产物', ('dataset',), optional=True)),
    params=(Q('infer_config', '推理配置（sampler 为引擎，含 template_config 等）', {}, type='dict'),),
    exclusive_inputs=(('model', 'artifact'),), effects=('save',), surface='artifact')
EVAL = D('eval', '评测', 'Task', '复用 Evaluator 与公共 runner，不关闭借用的会话采样资源。',
    ports_in=(P('sampler', 'SAMPLER', '采样来源', optional=True), P('endpoint', 'ENDPOINT', 'API 来源', optional=True),
              P('data', 'DATASET', '明确评测数据')),
    ports_out=(P('metrics', 'METRIC', '真实评测摘要', ('summary',)), P('artifact', 'ARTIFACT', '真实报告', ('report',), optional=True)),
    params=(Q('tokenizer', 'Tokenizer 来源（id/路径，与模型一致）', '', required=True),
            Q('model_config', '模型加载配置', {}, type='dict',
              doc='tokenizer 加载所需的身份字段（如本地目录的 model_type）；与模型节点自身一致'),
            Q('task_config', 'Evaluator 配置', {}, type='dict')), exclusive_inputs=(('sampler', 'endpoint'),),
    surface='artifact')
DEPLOY = D('deploy', '部署服务', 'Task', '经明确授权启动服务并等待就绪，命令线程不被阻塞。',
    ports_in=(P('model', 'ARTIFACT', '权重来源', ('model',)),), ports_out=(P('endpoint', 'ENDPOINT', '就绪服务'),),
    params=(Q('deploy_config', '服务配置', {}, type='dict'),), effects=('external', 'serve'), owner_kind='ENDPOINT',
    surface='artifact')
HUB_PUSH = D('hub_push', '发布模型', 'Task', '明确授权后调用公共 Hub API，返回实际发布结果。',
    ports_in=(P('model', 'ARTIFACT', '权重来源', ('model',)),),
    ports_out=(P('artifact', 'ARTIFACT', '实际发布结果', ('publication',)),),
    params=(Q('hub_model_id', '目标仓库', '', required=True),
            Q('hub', 'Hub', 'ModelScope', options=('ModelScope', 'HuggingFace')),
            Q('private', '私有仓库', True, type='bool'), Q('revision', '版本', 'main'),
            Q('commit_message', '提交说明', 'webui workflow push'),
            Q('token_env', '凭据环境变量名', '', required=True)),
    effects=('external',), surface='artifact')

REGISTRY = {descriptor.type: descriptor for descriptor in (
    MODEL, DATASET, SAMPLER, ROLLOUT, GROUP_RULE, LOSS_COMP, SCORER_COMP,
    TOOLSET, SANDBOX_ENV, API_ENDPOINT, CODE_COMPONENT,
    DATALOADER, STREAMING_DRIVER, CHALLENGER, GENERATE, SCORE,
    MODEL_FORWARD, PRIVILEGED_VIEW, ADVANTAGE, SIGNAL_COMBINE, SELECT,
    TRAIN_STEP, WEIGHT_SYNC, WEIGHT_AVERAGE, CHECKPOINT, MERGE, QUANTIZE,
    INFER, EVAL, DEPLOY, HUB_PUSH)}


def validate_registry() -> None:
    from .schema import CHANNEL_SCHEMAS
    # The wiring matrix must cover exactly the declared channels, and every
    # cross-kind adaptation a port declares must be backed by the matrix --
    # edge legality then has one source of truth for validator and frontend.
    if set(KIND_CATEGORIES) != set(CHANNEL_SCHEMAS):
        raise ValueError(f'kind matrix and channel schemas diverge: '
                         f'{sorted(set(KIND_CATEGORIES) ^ set(CHANNEL_SCHEMAS))}')
    for producer, allowed in KIND_ALLOWS.items():
        if producer not in KIND_CATEGORIES:
            raise ValueError(f'kind matrix adapts unknown kind {producer!r}')
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
            for accepted in port.accepted_kinds:
                # Same-kind acceptance is always legal (the matrix documents
                # "its own kind always is"); the table only carries the
                # CROSS-kind adaptations.
                if accepted != port.kind and port.kind not in KIND_ALLOWS.get(accepted, ()):
                    raise ValueError(f'{descriptor.type}.{port.key}: 通道适配 {accepted}->{port.kind} '
                                     f'不在 KIND_ALLOWS 矩阵内')
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
