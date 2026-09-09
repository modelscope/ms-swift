import type { GraphEdge, GraphFrame, GraphNode } from './nodeTypes';

/**
 * 示例图。训练被拆成组件之后，「怎么接」本身就是知识——
 * 与其让人从空画布猜，不如直接给几张能跑通的接法当起点。
 *
 * 坐标是手排的：一列一个阶段，从左到右就是数据流方向。
 */
export interface Preset {
  key: string;
  label: string;
  desc: string;
  nodes: GraphNode[];
  edges: GraphEdge[];
  frames?: GraphFrame[];
}

const E = (from: string, fromPort: string, to: string, toPort: string): GraphEdge => ({
  id: `${from}.${fromPort}-${to}.${toPort}`,
  from,
  fromPort,
  to,
  toPort,
});

/**
 * GRPO：一个 prompt 采 N 条 → 打分 → 组内归一化成优势 → clip 目标 + KL → 训练循环。
 * 特点是没有 critic，优势完全靠组内比较得出。
 */
const GRPO: Preset = {
  key: 'grpo',
  label: 'GRPO',
  desc: '组内采样 + 规则奖励 + 组内归一化优势，无 critic',
  nodes: [
    { id: 'g_model', type: 'model', x: 0, y: 0, status: 'idle' },
    { id: 'g_data', type: 'dataset', x: 0, y: 150, status: 'idle' },
    { id: 'g_ref', type: 'ref_model', x: 0, y: 300, status: 'idle' },
    { id: 'g_rollout', type: 'rollout', x: 262, y: 90, status: 'idle' },
    { id: 'g_reward', type: 'reward_fn', x: 524, y: 0, status: 'idle' },
    { id: 'g_kl', type: 'kl_penalty', x: 524, y: 290, status: 'idle' },
    { id: 'g_adv', type: 'advantage', x: 786, y: 0, status: 'idle' },
    { id: 'g_metric', type: 'metric', x: 786, y: 180, status: 'idle' },
    { id: 'g_ploss', type: 'policy_loss', x: 1048, y: 20, status: 'idle' },
    { id: 'g_optim', type: 'optimizer', x: 1048, y: 330, status: 'idle' },
    { id: 'g_sum', type: 'loss_sum', x: 1310, y: 60, status: 'idle' },
    { id: 'g_sched', type: 'scheduler', x: 1310, y: 330, status: 'idle' },
    { id: 'g_trainer', type: 'trainer', x: 1572, y: 120, status: 'idle' },
    { id: 'g_eval', type: 'eval', x: 1834, y: 170, status: 'idle' },
  ],
  edges: [
    E('g_model', 'model', 'g_rollout', 'model'),
    E('g_data', 'data', 'g_rollout', 'data'),
    E('g_rollout', 'traj', 'g_reward', 'traj'),
    E('g_rollout', 'traj', 'g_kl', 'traj'),
    E('g_rollout', 'traj', 'g_ploss', 'traj'),
    E('g_rollout', 'traj', 'g_metric', 'traj'),
    E('g_ref', 'model', 'g_kl', 'ref'),
    E('g_reward', 'reward', 'g_adv', 'reward'),
    E('g_reward', 'reward', 'g_metric', 'reward'),
    E('g_adv', 'adv', 'g_ploss', 'adv'),
    E('g_ploss', 'loss', 'g_sum', 'a'),
    E('g_kl', 'loss', 'g_sum', 'b'),
    E('g_sum', 'loss', 'g_trainer', 'loss'),
    E('g_metric', 'metric', 'g_trainer', 'metric'),
    E('g_optim', 'optim', 'g_sched', 'optim'),
    E('g_sched', 'optim', 'g_trainer', 'optim'),
    E('g_model', 'model', 'g_trainer', 'model'),
    E('g_trainer', 'ckpt', 'g_eval', 'ckpt'),
  ],
};

/** SFT：最短的一条链，交叉熵 + 分通道监控 */
const SFT: Preset = {
  key: 'sft',
  label: 'SFT',
  desc: '交叉熵损失 + 分通道 loss 监控',
  nodes: [
    { id: 's_model', type: 'model', x: 0, y: 0, status: 'idle' },
    { id: 's_data', type: 'dataset', x: 0, y: 150, status: 'idle' },
    { id: 's_loss', type: 'ce_loss', x: 262, y: 30, status: 'idle' },
    { id: 's_optim', type: 'optimizer', x: 262, y: 300, status: 'idle' },
    { id: 's_chan', type: 'channel_metric', x: 524, y: 210, status: 'idle' },
    { id: 's_sched', type: 'scheduler', x: 524, y: 380, status: 'idle' },
    { id: 's_trainer', type: 'trainer', x: 786, y: 40, status: 'idle' },
    { id: 's_eval', type: 'eval', x: 1048, y: 90, status: 'idle' },
    { id: 's_merge', type: 'merge', x: 1048, y: 260, status: 'idle' },
    { id: 's_deploy', type: 'deploy', x: 1310, y: 260, status: 'idle' },
  ],
  edges: [
    E('s_model', 'model', 's_loss', 'model'),
    E('s_data', 'data', 's_loss', 'data'),
    E('s_loss', 'loss', 's_chan', 'loss'),
    E('s_loss', 'loss', 's_trainer', 'loss'),
    E('s_chan', 'metric', 's_trainer', 'metric'),
    E('s_optim', 'optim', 's_sched', 'optim'),
    E('s_sched', 'optim', 's_trainer', 'optim'),
    E('s_model', 'model', 's_trainer', 'model'),
    E('s_trainer', 'ckpt', 's_eval', 'ckpt'),
    E('s_trainer', 'ckpt', 's_merge', 'ckpt'),
    E('s_merge', 'model', 's_deploy', 'model'),
  ],
};

/** DPO：不采样，直接吃偏好对，损失里自带参考模型 */
const DPO: Preset = {
  key: 'dpo',
  label: 'DPO',
  desc: '偏好对直接优化，不需要采样和奖励',
  nodes: [
    { id: 'd_model', type: 'model', x: 0, y: 0, status: 'idle' },
    { id: 'd_ref', type: 'ref_model', x: 0, y: 150, status: 'idle' },
    { id: 'd_data', type: 'dataset', x: 0, y: 300, status: 'idle' },
    { id: 'd_loss', type: 'dpo_loss', x: 262, y: 60, status: 'idle' },
    { id: 'd_optim', type: 'optimizer', x: 262, y: 300, status: 'idle' },
    { id: 'd_chan', type: 'channel_metric', x: 524, y: 60, status: 'idle' },
    { id: 'd_sched', type: 'scheduler', x: 524, y: 300, status: 'idle' },
    { id: 'd_trainer', type: 'trainer', x: 786, y: 90, status: 'idle' },
    { id: 'd_eval', type: 'eval', x: 1048, y: 140, status: 'idle' },
  ],
  edges: [
    E('d_model', 'model', 'd_loss', 'model'),
    E('d_ref', 'model', 'd_loss', 'ref'),
    E('d_data', 'data', 'd_loss', 'data'),
    E('d_loss', 'loss', 'd_chan', 'loss'),
    E('d_loss', 'loss', 'd_trainer', 'loss'),
    E('d_chan', 'metric', 'd_trainer', 'metric'),
    E('d_optim', 'optim', 'd_sched', 'optim'),
    E('d_sched', 'optim', 'd_trainer', 'optim'),
    E('d_model', 'model', 'd_trainer', 'model'),
    E('d_trainer', 'ckpt', 'd_eval', 'ckpt'),
  ],
};

/**
 * 多轮 GRPO（对应 cookbook/rl/multi_turn/multi_turn_grpo.py）。
 *
 * 跟单轮 GRPO 的区别：不读数据集，prompt 来自环境的 observation；
 * 不写奖励函数，分数由环境给；采样是“模型发工具调用 → 环境回观测”
 * 来回跑 max_turns 轮。
 *
 * 连线做了三处合并，否则照搬脚本会多出七八根线：
 *  1. logprobs 不单拉线——它本来就在轨迹 dict 里（traj['logprobs']），
 *     所以 TRAJ 一根线就带着它走。
 *  2. tool_managers / env_tools_list 不单拉线——它们和 adapters 是同一批
 *     env slot 的三个视图，合成一根 ENV 线。回合奖励只接 TRAJ 就够，
 *     因为轨迹和 slot 按下标一一对应。
 *  3. sync_weights / reset_prefix_cache 不单独成节点——它们是副作用调用，
 *     不向下游传值，收成了推理引擎的 sync_weights 参数。
 */
const MULTI_TURN_GRPO: Preset = {
  key: 'multi_turn_grpo',
  label: '多轮 GRPO',
  desc: '环境交互 + 工具调用，奖励来自环境而非奖励函数',
  nodes: [
    /* 循环外：只构造一次的声明 */
    { id: 'm_mapper', type: 'action_mapper', x: 0, y: 0, status: 'idle' },
    { id: 'm_tools', type: 'tool_schema', x: 0, y: 170, status: 'idle' },
    { id: 'm_model', type: 'model', x: 0, y: 310, status: 'idle' },
    { id: 'm_optim', type: 'optimizer', x: 0, y: 450, status: 'idle' },
    { id: 'm_env', type: 'env_pool', x: 262, y: 0, status: 'idle' },
    { id: 'm_sampler', type: 'sampler', x: 262, y: 190, status: 'idle' },
    { id: 'm_loss', type: 'grpo_loss', x: 262, y: 380, status: 'idle' },
    { id: 'm_sched', type: 'scheduler', x: 262, y: 530, status: 'idle' },
    /* 循环内：每步重跑 */
    { id: 'm_rollout', type: 'multi_turn_rollout', x: 524, y: 60, status: 'idle' },
    { id: 'm_reward', type: 'episode_reward', x: 786, y: 40, status: 'idle' },
    { id: 'm_metric', type: 'metric', x: 786, y: 200, status: 'idle' },
    { id: 'm_adv', type: 'advantage', x: 1048, y: 40, status: 'idle' },
    { id: 'm_filter', type: 'traj_filter', x: 1310, y: 60, status: 'idle' },
    { id: 'm_trainer', type: 'trainer', x: 1572, y: 60, status: 'idle' },
  ],
  edges: [
    E('m_mapper', 'fn', 'm_env', 'mapper'),
    E('m_model', 'model', 'm_sampler', 'model'),
    E('m_model', 'model', 'm_trainer', 'model'),
    E('m_sampler', 'sampler', 'm_rollout', 'sampler'),
    E('m_env', 'env', 'm_rollout', 'env'),
    E('m_tools', 'tool', 'm_rollout', 'tool'),
    E('m_rollout', 'traj', 'm_reward', 'traj'),
    E('m_rollout', 'traj', 'm_metric', 'traj'),
    E('m_rollout', 'traj', 'm_filter', 'traj'),
    E('m_reward', 'reward', 'm_adv', 'reward'),
    E('m_reward', 'reward', 'm_metric', 'reward'),
    E('m_adv', 'adv', 'm_filter', 'adv'),
    E('m_filter', 'traj', 'm_trainer', 'traj'),
    E('m_filter', 'adv', 'm_trainer', 'adv'),
    E('m_loss', 'loss', 'm_trainer', 'loss'),
    E('m_optim', 'optim', 'm_sched', 'optim'),
    E('m_sched', 'optim', 'm_trainer', 'optim'),
    E('m_metric', 'metric', 'm_trainer', 'metric'),
  ],
  frames: [
    {
      id: 'm_loop',
      label: '训练循环',
      note: 'while optim_step < max_steps：环境重置 → 同步权重 → 采样 → 打分 → 优势 → 过滤 → 更新',
      x: 496,
      y: -44,
      w: 1316,
      h: 408,
    },
  ],
};

export const PRESETS: Preset[] = [MULTI_TURN_GRPO, GRPO, SFT, DPO];

/** 默认打开就是多轮 GRPO——组件拆得够细才能接出来的就是它 */
export const DEFAULT_PRESET = MULTI_TURN_GRPO;

/** 每次取副本，避免示例被画布上的拖动改掉 */
export function clonePreset(p: Preset): {
  nodes: GraphNode[];
  edges: GraphEdge[];
  frames: GraphFrame[];
} {
  return {
    nodes: p.nodes.map((n) => ({ ...n })),
    edges: p.edges.map((e) => ({ ...e })),
    frames: (p.frames ?? []).map((f) => ({ ...f })),
  };
}
