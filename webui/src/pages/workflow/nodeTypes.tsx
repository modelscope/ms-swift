import type { ReactNode } from 'react';
import { Activity, ArrowLeftRight, Brain, ChartBar, ChartLine, CloudUpload, Code, Database, Filter, FlaskConical, Gauge, GitBranch, GitMerge, Globe, Lock, Minimize2, Network, RefreshCw, Repeat, Rocket, Shrink, Sigma, SlidersHorizontal, SquarePlus, Target, TrendingDown, TrendingUp, Trophy, Wrench, Zap } from 'lucide-react';

/**
 * 节点类型注册表：编排画布上「有哪些组件」的唯一定义源。
 *
 * 粒度上刻意不做「SFT 训练」这种一体节点——训练循环里真正要调的是
 * 损失、优势估计、奖励、指标这几块，包成一个节点就没得配了。
 * 所以训练被拆成：采样 → 奖励 → 优势 → 损失 → 训练循环，各自独立成节点。
 *
 * 加一种节点 = 在这里加一项，左侧组件面板、画布、YAML 生成都会自动跟随。
 */

/**
 * 端口数据类型。连线两端类型必须一致，画布会挡掉不匹配的连接；
 * 端口圆点和连线颜色都取自类型，扫一眼就知道这根线在传什么。
 */
export const PORT_TYPES = {
  MODEL: { label: '模型', color: '#3B82F6' },
  DATA: { label: '数据', color: '#A855F7' },
  SAMPLER: { label: '采样器', color: '#0EA5E9' },
  /**
   * 轨迹。一根 TRAJ 线传的是一批轨迹（batch × num_generations 条），
   * 每条轨迹自带 input_ids / labels / logprobs / turns。
   * logprobs 不单独拉线就是因为它本来就在轨迹里。
   */
  TRAJ: { label: '轨迹', color: '#22D3EE' },
  /** 环境池。一根 ENV 线代表一批 env slot，与轨迹按下标一一对应 */
  ENV: { label: '环境', color: '#84CC16' },
  TOOL: { label: '工具', color: '#D946EF' },
  /** 用户手写的 Python 函数，由代码节点产出 */
  FUNC: { label: '函数', color: '#94A3B8' },
  REWARD: { label: '奖励', color: '#EAB308' },
  ADV: { label: '优势', color: '#F97316' },
  LOSS: { label: '损失', color: '#EF4444' },
  METRIC: { label: '指标', color: '#14B8A6' },
  OPTIM: { label: '优化器', color: '#8B5CF6' },
  CKPT: { label: '权重', color: '#F59E0B' },
  REPORT: { label: '报告', color: '#22C55E' },
  ENDPOINT: { label: '服务', color: '#10B981' },
} as const;

export type PortTypeKey = keyof typeof PORT_TYPES;

export interface PortDef {
  key: string;
  label: string;
  type: PortTypeKey;
}

export interface NodeParam {
  label: string;
  value: string;
}

export interface NodeTypeDef {
  key: string;
  label: string;
  /** 组件面板里的分组 */
  category: string;
  /** 节点主色（按分组走），标题栏用它 */
  color: string;
  icon: ReactNode;
  inputs: PortDef[];
  outputs: PortDef[];
  /** 节点体里显示的参数摘要 */
  params: NodeParam[];
  /** 能否单独运行。纯声明节点（数据、模型、优化器）没有运行的概念 */
  runnable?: boolean;
  /** 面板 tooltip 里的一句话说明 */
  hint?: string;
  /**
   * 代码节点：节点体里显示可编辑的 Python 片段，而不是参数表。
   * 承载 action_mapper 这类没法参数化的自由逻辑。
   */
  code?: string;
  /**
   * 端口锁定。true 表示这个节点的连线不允许用户改动。
   *
   * 只用在轨迹过滤上：它的三路数据（轨迹、logprobs、优势）必须同进同出，
   * 且优势必须在过滤的上游算好。这两条一旦被改，训练结果会错但不报错——
   * 端口类型全都匹配，校验器查不出来。等图级校验器能表达跨节点约束再放开。
   */
  lockedPorts?: boolean;
}

/* 分组配色：同组同色，画布上按色块分区读图 */
const C = {
  input: '#3A6EA5',
  data: '#6D3FD1',
  rollout: '#0E7490',
  agent: '#3F6212',
  reward: '#A16207',
  adv: '#C2410C',
  loss: '#B4342B',
  metric: '#0F766E',
  loop: '#4F46E5',
  after: '#475569',
  code: '#334155',
};

export const NODE_TYPES: Record<string, NodeTypeDef> = {
  /* ===== 输入 ===== */
  dataset: {
    key: 'dataset',
    label: '数据集',
    category: '输入',
    color: C.data,
    icon: <Database />,
    inputs: [],
    outputs: [{ key: 'data', label: 'data', type: 'DATA' }],
    params: [
      { label: 'dataset', value: 'AI-MO/NuminaMath' },
      { label: 'split', value: 'train' },
    ],
  },
  model: {
    key: 'model',
    label: '策略模型',
    category: '输入',
    color: C.input,
    icon: <Brain />,
    inputs: [],
    outputs: [{ key: 'model', label: 'model', type: 'MODEL' }],
    params: [
      { label: 'model', value: 'Qwen/Qwen3-8B' },
      { label: 'train_type', value: 'lora' },
    ],
    hint: '被训练的那个模型',
  },
  ref_model: {
    key: 'ref_model',
    label: '参考模型',
    category: '输入',
    color: C.input,
    icon: <Lock />,
    inputs: [],
    outputs: [{ key: 'model', label: 'model', type: 'MODEL' }],
    params: [
      { label: 'ref_model', value: '同策略模型初始权重' },
      { label: 'frozen', value: 'true' },
    ],
    hint: '算 KL 用的冻结模型',
  },

  /* ===== 采样 ===== */
  rollout: {
    key: 'rollout',
    label: '采样 Rollout',
    category: '采样',
    color: C.rollout,
    icon: <Repeat />,
    inputs: [
      { key: 'model', label: 'model', type: 'MODEL' },
      { key: 'data', label: 'prompt', type: 'DATA' },
    ],
    outputs: [{ key: 'traj', label: 'trajectories', type: 'TRAJ' }],
    params: [
      { label: 'num_generations', value: '8' },
      { label: 'temperature', value: '1.0' },
      { label: 'max_tokens', value: '1024' },
      { label: 'backend', value: 'vllm (colocate)' },
    ],
    runnable: true,
    hint: '一个 prompt 采 N 条，GRPO 的一组就是它',
  },
  sampler: {
    key: 'sampler',
    label: '推理引擎',
    category: '采样',
    color: C.rollout,
    icon: <Zap />,
    inputs: [{ key: 'model', label: 'sync from', type: 'MODEL' }],
    outputs: [{ key: 'sampler', label: 'sampler', type: 'SAMPLER' }],
    params: [
      { label: 'engine', value: 'vLLMSampler' },
      { label: 'max_model_len', value: '8192' },
      { label: 'gpu_memory_util', value: '0.8' },
      { label: 'sync_weights', value: '每步（LoRA）' },
    ],
    hint: '独占一组 GPU 跑生成；接进来的 model 线表示每步同步权重',
  },
  multi_turn_rollout: {
    key: 'multi_turn_rollout',
    label: '多轮采样',
    category: '采样',
    color: C.rollout,
    icon: <ArrowLeftRight />,
    inputs: [
      { key: 'sampler', label: 'sampler', type: 'SAMPLER' },
      { key: 'env', label: 'env', type: 'ENV' },
      { key: 'tool', label: 'tools', type: 'TOOL' },
    ],
    outputs: [{ key: 'traj', label: 'trajectories', type: 'TRAJ' }],
    params: [
      { label: 'max_turns', value: '6' },
      { label: 'num_generations', value: '8' },
      { label: 'temperature', value: '1.0' },
      { label: 'max_tokens', value: '2048' },
      { label: 'truncation', value: 'delete' },
    ],
    runnable: true,
    hint: '模型发工具调用、环境回 observation，来回跑 max_turns 轮',
  },

  /* ===== Agent 环境 ===== */
  env_pool: {
    key: 'env_pool',
    label: '环境池',
    category: 'Agent 环境',
    color: C.agent,
    icon: <Globe />,
    inputs: [{ key: 'mapper', label: 'action_mapper', type: 'FUNC' }],
    outputs: [{ key: 'env', label: 'env slots', type: 'ENV' }],
    params: [
      { label: 'env_name', value: 'openspiel_env' },
      { label: 'game_name', value: 'blackjack' },
      { label: 'pool_size', value: 'batch × num_gen' },
      { label: 'placement', value: '驱动进程内' },
    ],
    runnable: true,
    hint: '一次持住 N 个环境实例，一条轨迹占一个 slot',
  },
  tool_schema: {
    key: 'tool_schema',
    label: '工具定义',
    category: 'Agent 环境',
    color: C.agent,
    icon: <Wrench />,
    inputs: [],
    outputs: [{ key: 'tool', label: 'tools', type: 'TOOL' }],
    params: [
      { label: 'tools', value: 'play(action)' },
      { label: 'enum', value: 'hit / stand' },
    ],
    hint: '模型看得到的工具声明，就是那段 JSON schema',
  },
  action_mapper: {
    key: 'action_mapper',
    label: '动作映射',
    category: 'Agent 环境',
    color: C.code,
    icon: <Code />,
    inputs: [],
    outputs: [{ key: 'fn', label: 'fn', type: 'FUNC' }],
    params: [],
    code: [
      "def mapper(tool_name, arguments):",
      "    a = arguments.get('action', 'stand')",
      "    return {'action_id': MAP[a],",
      "            'game_name': 'blackjack'}",
    ].join('\n'),
    hint: '把模型的 tool call 翻成环境认的 action，没法参数化，只能写代码',
  },
  code: {
    key: 'code',
    label: '代码节点',
    category: 'Agent 环境',
    color: C.code,
    icon: <Code />,
    inputs: [{ key: 'in', label: 'in', type: 'TRAJ' }],
    outputs: [{ key: 'out', label: 'out', type: 'TRAJ' }],
    params: [],
    code: 'def fn(x):\n    return x',
    hint: '自由 Python，端口类型自己声明；生成脚本时原样贴进去',
  },

  /* ===== 奖励与优势 ===== */
  reward_fn: {
    key: 'reward_fn',
    label: '奖励函数',
    category: '奖励与优势',
    color: C.reward,
    icon: <Sigma />,
    inputs: [{ key: 'traj', label: 'trajectories', type: 'TRAJ' }],
    outputs: [{ key: 'reward', label: 'reward', type: 'REWARD' }],
    params: [
      { label: 'reward_funcs', value: 'accuracy, format' },
      { label: 'reward_weights', value: '1.0, 0.2' },
    ],
    runnable: true,
    hint: '注册在 orms 里的规则打分，可挂多个并加权',
  },
  reward_model: {
    key: 'reward_model',
    label: '奖励模型',
    category: '奖励与优势',
    color: C.reward,
    icon: <Trophy />,
    inputs: [{ key: 'traj', label: 'trajectories', type: 'TRAJ' }],
    outputs: [{ key: 'reward', label: 'reward', type: 'REWARD' }],
    params: [
      { label: 'reward_model', value: 'Skywork-Reward-8B' },
      { label: 'weight', value: '1.0' },
    ],
    runnable: true,
  },
  advantage: {
    key: 'advantage',
    label: '组内优势',
    category: '奖励与优势',
    color: C.adv,
    icon: <TrendingUp />,
    inputs: [{ key: 'reward', label: 'reward', type: 'REWARD' }],
    outputs: [{ key: 'adv', label: 'advantage', type: 'ADV' }],
    params: [
      { label: 'advantage', value: 'GRPOAdvantage' },
      { label: 'scale', value: 'group' },
      { label: 'num_generations', value: '8' },
    ],
    runnable: true,
    hint: '组内减均值除标准差，不要 critic。scale 可选 group / batch / none / gdpo',
  },
  gae: {
    key: 'gae',
    label: 'GAE 优势',
    category: '奖励与优势',
    color: C.adv,
    icon: <TrendingDown />,
    inputs: [{ key: 'reward', label: 'reward', type: 'REWARD' }],
    outputs: [{ key: 'adv', label: 'advantage', type: 'ADV' }],
    params: [{ label: 'advantage', value: 'GAEAdvantage' }],
    runnable: true,
    hint: '带 critic 的广义优势估计，twinkle.advantage 里有',
  },
  episode_reward: {
    key: 'episode_reward',
    label: '回合奖励',
    category: '奖励与优势',
    color: C.reward,
    icon: <Trophy />,
    inputs: [{ key: 'traj', label: 'trajectories', type: 'TRAJ' }],
    outputs: [{ key: 'reward', label: 'reward', type: 'REWARD' }],
    params: [
      { label: 'source', value: 'EnvTool.episode_reward' },
      { label: 'reduce', value: '整局累加' },
    ],
    runnable: true,
    hint: '不用奖励函数，分数由环境直接给——输了还是赢了',
  },
  traj_filter: {
    key: 'traj_filter',
    label: '轨迹过滤',
    category: '奖励与优势',
    color: C.adv,
    icon: <Filter />,
    inputs: [
      { key: 'traj', label: 'trajectories', type: 'TRAJ' },
      { key: 'adv', label: 'advantage', type: 'ADV' },
    ],
    outputs: [
      { key: 'traj', label: 'trajectories', type: 'TRAJ' },
      { key: 'adv', label: 'advantage', type: 'ADV' },
    ],
    params: [
      { label: 'drop_if', value: 'len > max_length' },
      { label: 'drop_if', value: '可训 tokens = 0' },
      { label: 'min_batch', value: '≥ dp_size 否则整批跳过' },
    ],
    lockedPorts: true,
    hint: '丢掉超长和空轨迹。两路输出必须同进同出，所以连线锁定',
  },
  rloo: {
    key: 'rloo',
    label: 'RLOO 优势',
    category: '奖励与优势',
    color: C.adv,
    icon: <Activity />,
    inputs: [{ key: 'reward', label: 'reward', type: 'REWARD' }],
    outputs: [{ key: 'adv', label: 'advantage', type: 'ADV' }],
    params: [
      { label: 'advantage', value: 'RLOOAdvantage' },
      { label: 'scale', value: 'batch' },
    ],
    runnable: true,
    hint: '留一法基线，跟组内优势换一个节点就能切',
  },

  /* ===== 损失 ===== */
  policy_loss: {
    key: 'policy_loss',
    label: '策略损失',
    category: '损失',
    color: C.loss,
    icon: <Target />,
    inputs: [
      { key: 'traj', label: 'trajectories', type: 'TRAJ' },
      { key: 'adv', label: 'advantage', type: 'ADV' },
    ],
    outputs: [{ key: 'loss', label: 'loss', type: 'LOSS' }],
    params: [
      { label: 'loss_type', value: 'bnpo' },
      { label: 'epsilon', value: '0.2' },
      { label: 'epsilon_high', value: '0.28' },
      { label: 'is_level', value: 'token' },
    ],
    runnable: true,
    hint: '带 clip 的重要性采样目标',
  },
  grpo_loss: {
    key: 'grpo_loss',
    label: 'GRPO 损失',
    category: '损失',
    color: C.loss,
    icon: <Target />,
    inputs: [],
    outputs: [{ key: 'loss', label: 'loss', type: 'LOSS' }],
    params: [
      { label: 'loss', value: 'GRPOLoss' },
      { label: 'epsilon', value: '0.2' },
    ],
    hint: 'twinkle 里 set_loss 是模型配置，不吃数据流，所以没有输入口。'
      + '可换 GSPOLoss / BNPOLoss / CISPOLoss / DRGRPOLoss',
  },
  ce_loss: {
    key: 'ce_loss',
    label: '交叉熵损失',
    category: '损失',
    color: C.loss,
    icon: <TrendingDown />,
    inputs: [
      { key: 'model', label: 'model', type: 'MODEL' },
      { key: 'data', label: 'data', type: 'DATA' },
    ],
    outputs: [{ key: 'loss', label: 'loss', type: 'LOSS' }],
    params: [
      { label: 'loss_type', value: 'cross_entropy' },
      { label: 'enable_channel_loss', value: 'true' },
    ],
    runnable: true,
    hint: 'SFT 用这个',
  },
  dpo_loss: {
    key: 'dpo_loss',
    label: '偏好损失',
    category: '损失',
    color: C.loss,
    icon: <ArrowLeftRight />,
    inputs: [
      { key: 'model', label: 'model', type: 'MODEL' },
      { key: 'ref', label: 'ref', type: 'MODEL' },
      { key: 'data', label: 'pairs', type: 'DATA' },
    ],
    outputs: [{ key: 'loss', label: 'loss', type: 'LOSS' }],
    params: [
      { label: 'rlhf_type', value: 'dpo' },
      { label: 'beta', value: '0.1' },
    ],
    runnable: true,
  },
  kl_penalty: {
    key: 'kl_penalty',
    label: 'KL 惩罚',
    category: '损失',
    color: C.loss,
    icon: <Shrink />,
    inputs: [
      { key: 'traj', label: 'trajectories', type: 'TRAJ' },
      { key: 'ref', label: 'ref', type: 'MODEL' },
    ],
    outputs: [{ key: 'loss', label: 'kl', type: 'LOSS' }],
    params: [
      { label: 'beta', value: '0.04' },
      { label: 'kl_estimator', value: 'k3' },
    ],
    runnable: true,
    hint: '也可以改成在优势节点里开 kl_in_reward（归一化前从奖励里扣），二者选一',
  },
  loss_sum: {
    key: 'loss_sum',
    label: '损失合成',
    category: '损失',
    color: C.loss,
    icon: <SquarePlus />,
    inputs: [
      { key: 'a', label: 'loss a', type: 'LOSS' },
      { key: 'b', label: 'loss b', type: 'LOSS' },
      { key: 'c', label: 'loss c', type: 'LOSS' },
    ],
    outputs: [{ key: 'loss', label: 'total', type: 'LOSS' }],
    params: [{ label: 'weights', value: '1.0, 1.0, 1.0' }],
    hint: '多个损失加权求和后交给训练循环',
  },

  /* ===== 指标 ===== */
  metric: {
    key: 'metric',
    label: '训练指标',
    category: '指标',
    color: C.metric,
    icon: <ChartLine />,
    inputs: [
      { key: 'traj', label: 'trajectories', type: 'TRAJ' },
      { key: 'reward', label: 'reward', type: 'REWARD' },
    ],
    outputs: [{ key: 'metric', label: 'metric', type: 'METRIC' }],
    params: [
      { label: 'keys', value: 'reward/mean, reward/std' },
      { label: 'extra', value: 'completion/len, clip_ratio' },
      { label: 'log_every', value: '5' },
    ],
    hint: '想在曲线里看到什么，就在这里加 key',
  },
  channel_metric: {
    key: 'channel_metric',
    label: '分通道损失',
    category: '指标',
    color: C.metric,
    icon: <ChartBar />,
    inputs: [{ key: 'loss', label: 'loss', type: 'LOSS' }],
    outputs: [{ key: 'metric', label: 'metric', type: 'METRIC' }],
    params: [
      { label: 'metric', value: 'loss_<channel>' },
      { label: 'channel', value: '读数据集 channel 字段' },
    ],
    hint: '按数据来源分开看 loss，需要 enable_channel_loss',
  },

  /* ===== 训练循环 ===== */
  optimizer: {
    key: 'optimizer',
    label: '优化器',
    category: '训练循环',
    color: C.loop,
    icon: <SlidersHorizontal />,
    inputs: [],
    outputs: [{ key: 'optim', label: 'optim', type: 'OPTIM' }],
    params: [
      { label: 'optim', value: 'adamw_torch' },
      { label: 'lr', value: '1e-6' },
      { label: 'weight_decay', value: '0.01' },
    ],
  },
  scheduler: {
    key: 'scheduler',
    label: '学习率调度',
    category: '训练循环',
    color: C.loop,
    icon: <Gauge />,
    inputs: [{ key: 'optim', label: 'optim', type: 'OPTIM' }],
    outputs: [{ key: 'optim', label: 'optim', type: 'OPTIM' }],
    params: [
      { label: 'lr_scheduler', value: 'cosine' },
      { label: 'warmup_ratio', value: '0.05' },
    ],
  },
  trainer: {
    key: 'trainer',
    label: '训练循环',
    category: '训练循环',
    color: C.loop,
    icon: <RefreshCw />,
    inputs: [
      { key: 'model', label: 'model', type: 'MODEL' },
      { key: 'traj', label: 'trajectories', type: 'TRAJ' },
      { key: 'adv', label: 'advantage', type: 'ADV' },
      { key: 'loss', label: 'loss', type: 'LOSS' },
      { key: 'optim', label: 'optim', type: 'OPTIM' },
      { key: 'metric', label: 'metric', type: 'METRIC' },
    ],
    outputs: [{ key: 'ckpt', label: 'ckpt', type: 'CKPT' }],
    params: [
      { label: 'max_steps', value: '1000' },
      { label: 'mini_batch_size', value: '8' },
      { label: 'micro_batch_size', value: '2' },
      { label: 'max_grad_norm', value: '1.0' },
      { label: 'save_steps', value: '500' },
    ],
    runnable: true,
    hint: '内层按 mini_batch 切片做 forward_backward，每片一次 optim step',
  },

  /* ===== 训练之后 ===== */
  eval: {
    key: 'eval',
    label: '评测',
    category: '训练之后',
    color: C.after,
    icon: <FlaskConical />,
    inputs: [{ key: 'ckpt', label: 'ckpt', type: 'CKPT' }],
    outputs: [{ key: 'report', label: 'report', type: 'REPORT' }],
    params: [
      { label: 'eval_dataset', value: 'gsm8k, math500' },
      { label: 'limit', value: '200' },
    ],
    runnable: true,
  },
  gate: {
    key: 'gate',
    label: '分数门槛',
    category: '训练之后',
    color: C.after,
    icon: <GitBranch />,
    inputs: [{ key: 'report', label: 'report', type: 'REPORT' }],
    outputs: [
      { key: 'pass', label: 'pass', type: 'REPORT' },
      { key: 'fail', label: 'fail', type: 'REPORT' },
    ],
    params: [
      { label: 'metric', value: 'gsm8k' },
      { label: 'threshold', value: '>= 60' },
    ],
    runnable: true,
  },
  merge: {
    key: 'merge',
    label: 'LoRA 合并',
    category: '训练之后',
    color: C.after,
    icon: <GitMerge />,
    inputs: [{ key: 'ckpt', label: 'ckpt', type: 'CKPT' }],
    outputs: [{ key: 'model', label: 'model', type: 'MODEL' }],
    params: [{ label: 'merge_lora', value: 'true' }],
    runnable: true,
  },
  quantize: {
    key: 'quantize',
    label: '量化',
    category: '训练之后',
    color: C.after,
    icon: <Minimize2 />,
    inputs: [{ key: 'model', label: 'model', type: 'MODEL' }],
    outputs: [{ key: 'model', label: 'model', type: 'MODEL' }],
    params: [
      { label: 'quant_method', value: 'gptq' },
      { label: 'quant_bits', value: '4' },
    ],
    runnable: true,
  },
  deploy: {
    key: 'deploy',
    label: '部署',
    category: '训练之后',
    color: C.after,
    icon: <Rocket />,
    inputs: [{ key: 'model', label: 'model', type: 'MODEL' }],
    outputs: [{ key: 'endpoint', label: 'endpoint', type: 'ENDPOINT' }],
    params: [
      { label: 'infer_backend', value: 'vllm' },
      { label: 'max_model_len', value: '8192' },
    ],
    runnable: true,
  },
  push: {
    key: 'push',
    label: '推送 Hub',
    category: '训练之后',
    color: C.after,
    icon: <CloudUpload />,
    inputs: [{ key: 'model', label: 'model', type: 'MODEL' }],
    outputs: [],
    params: [
      { label: 'hub', value: 'ModelScope' },
      { label: 'private', value: 'true' },
    ],
    runnable: true,
  },
  subflow: {
    key: 'subflow',
    label: '子流程',
    category: '训练之后',
    color: C.after,
    icon: <Network />,
    inputs: [{ key: 'ckpt', label: 'in', type: 'CKPT' }],
    outputs: [{ key: 'report', label: 'out', type: 'REPORT' }],
    params: [{ label: 'ref', value: 'daily-regression' }],
    runnable: true,
  },
};

/** 组件面板的分组顺序，按训练循环的数据流从上到下排 */
export const CATEGORY_ORDER = [
  '输入',
  '采样',
  'Agent 环境',
  '奖励与优势',
  '损失',
  '指标',
  '训练循环',
  '训练之后',
];

export const NODE_TYPE_LIST = Object.values(NODE_TYPES);

/** 节点运行状态。单个节点可以独立跑，所以状态挂在节点上而非整图 */
export type NodeStatus = 'idle' | 'running' | 'done' | 'failed';

/**
 * 图上的一个节点。这是编排的「作者格式」：presets.ts 手排坐标、YAML/代码生成、
 * AI 上下文、输出模拟全都读它。画布内部会把它转成 React Flow 的节点结构，
 * 转换只发生在 NodeCanvas 一个地方。
 */
export interface GraphNode {
  id: string;
  type: string;
  x: number;
  y: number;
  status: NodeStatus;
  /** 允许逐节点覆盖默认参数 */
  params?: NodeParam[];
  /** 代码节点的当前内容，覆盖类型里的默认片段 */
  code?: string;
}

/**
 * 循环容器。画在节点下面一层的半透明框，标出“这一块每步重跑”。
 *
 * 目前只是视觉分区 + 参数标注，没做子图归属（拖节点不会跟随框、
 * 拖出框外也不会自动移出循环）。真容器要先想清楚跨迭代状态和
 * 控制依赖怎么表达，这一版先把语义画出来。
 */
export interface GraphFrame {
  id: string;
  label: string;
  x: number;
  y: number;
  w: number;
  h: number;
  /** 框右上角的一行注，写循环条件 */
  note?: string;
  color?: string;
}

export interface GraphEdge {
  id: string;
  from: string;
  fromPort: string;
  to: string;
  toPort: string;
}

/**
 * 节点宽度。
 *
 * 原来这里还有 HEADER_H / PORT_TOP / PORT_GAP / portY() / nodeHeight() 五个东西，
 * 用来手算每个端口圆点的绝对坐标、以及手算「适应视图」要缩到多少。
 * 换成 React Flow 之后这些全都不需要了：端口是 <Handle>，位置由它自己排；
 * 节点高度由内容撑开、由布局引擎量；fitView 是库自带的。
 *
 * 只剩宽度还留着，因为它是设计约束（一列放得下几个节点）而不是推导值，
 * 而且点组件面板往画布中央加节点时要拿它算居中偏移。
 */
export const NODE_W = 208;

/** 查端口定义，连线的颜色和类型校验都用它 */
export function findPort(
  typeKey: string,
  portKey: string,
  dir: 'in' | 'out',
): PortDef | undefined {
  const def = NODE_TYPES[typeKey];
  if (!def) return undefined;
  return (dir === 'in' ? def.inputs : def.outputs).find((p) => p.key === portKey);
}
