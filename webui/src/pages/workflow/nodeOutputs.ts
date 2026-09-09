import type { GraphNode } from './nodeTypes';
import { NODE_TYPES } from './nodeTypes';

/**
 * 节点运行后的输出（mock）。
 *
 * 每种节点该给什么，是按它在数据流里的位置定的：
 * 采样节点该给出一条真实的轨迹长什么样，奖励节点该给出分数分布，
 * 训练节点该给出日志。全都塞成一张 key-value 表反而看不出问题在哪。
 *
 * 这些数字是编的，但形状是真的——接后端时只要把 outputOf 换成读接口返回，
 * 上层的 NodeInspector 一行都不用改。
 */

/** 一段输出内容。故意分成几种块，让不同节点能长得不一样 */
export type OutputBlock =
  | { kind: 'kv'; title?: string; rows: [string, string][] }
  /** 等宽正文，用来放采样样本、schema、报错栈这类原文 */
  | { kind: 'text'; title?: string; body: string }
  /** 简易柱状分布，value 是 0~1 的占比 */
  | { kind: 'dist'; title?: string; bars: { label: string; value: number }[] }
  | { kind: 'log'; title?: string; lines: string[] };

export interface NodeOutput {
  /** 一句话结论。这是最该先被看到的东西 */
  summary: string;
  elapsed: string;
  blocks: OutputBlock[];
  /** 有值就说明这次跑挂了，展示成红色 */
  error?: string;
}

/** 一条多轮 blackjack 轨迹，跟 multi_turn_grpo 那个例子对得上 */
const TRAJ_SAMPLE = [
  '── turn 1 ──────────────────────────────',
  'user   : 你的手牌 [9, 5]，庄家明牌 [7]。请选择 hit 或 stand。',
  'assist : <tool_call>{"name":"play","arguments":{"action":"hit"}}</tool_call>',
  'tool   : {"obs":"你摸到 4，当前 18","done":false,"reward":0}',
  '',
  '── turn 2 ──────────────────────────────',
  'user   : 你的手牌 [9, 5, 4]，庄家明牌 [7]。',
  'assist : <tool_call>{"name":"play","arguments":{"action":"stand"}}</tool_call>',
  'tool   : {"obs":"庄家 17，你 18","done":true,"reward":1}',
  '',
  'turns=2  tokens=386  可训 tokens=41  episode_reward=1',
].join('\n');

/** 训练循环的日志。前缀跟 LogViewer 的着色规则对齐 */
const TRAIN_LOG = [
  '[INFO] resume from step 0, world_size=8, dp=4',
  '[INFO] step 1/1000 loss=0.6931 reward/mean=0.42 reward/std=0.49 clip_ratio=0.03',
  '[INFO] step 2/1000 loss=0.6802 reward/mean=0.45 reward/std=0.50 clip_ratio=0.04',
  '[WARN] step 3: 2 trajectories dropped (len > max_length)',
  '[INFO] step 3/1000 loss=0.6714 reward/mean=0.47 reward/std=0.50 clip_ratio=0.05',
  '[INFO] step 4/1000 loss=0.6598 reward/mean=0.51 reward/std=0.50 clip_ratio=0.06',
  '[INFO] step 5/1000 loss=0.6421 reward/mean=0.55 reward/std=0.49 clip_ratio=0.07',
  '[INFO] sync policy weights to sampler, 0.9s',
];

function pv(node: GraphNode, label: string, fallback = '—'): string {
  const ps = node.params ?? NODE_TYPES[node.type].params;
  return ps.find((p) => p.label === label)?.value ?? fallback;
}

/**
 * 生成某个节点的输出。
 *
 * 没写到的类型走兜底：至少把生效参数原样列出来。
 * 这比显示「暂无输出」有用——起码能确认跑的时候用的是哪套参数。
 */
export function outputOf(node: GraphNode): NodeOutput {
  const def = NODE_TYPES[node.type];

  switch (node.type) {
    case 'dataset':
      return {
        summary: `读到 7473 条样本，${pv(node, 'split')} 划分`,
        elapsed: '1.2s',
        blocks: [
          {
            kind: 'kv',
            rows: [
              ['dataset', pv(node, 'dataset')],
              ['rows', '7473'],
              ['字段', 'problem, solution'],
              ['prompt 长度 p50 / p99', '86 / 264 tokens'],
            ],
          },
          {
            kind: 'text',
            title: '第一条',
            body: '{\n  "problem": "求 x^2 - 5x + 6 = 0 的所有实根。",\n  "solution": "x = 2 或 x = 3"\n}',
          },
        ],
      };

    case 'model':
    case 'ref_model':
      return {
        summary: `权重已加载，${pv(node, 'train_type', 'frozen')}`,
        elapsed: '18.4s',
        blocks: [
          {
            kind: 'kv',
            rows: [
              ['参数量', '8.19 B'],
              ['可训参数', node.type === 'ref_model' ? '0（冻结）' : '20.2 M（0.25%）'],
              ['dtype', 'bfloat16'],
              ['显存占用', node.type === 'ref_model' ? '15.6 GB' : '16.3 GB'],
            ],
          },
        ],
      };

    case 'sampler':
      return {
        summary: `vLLM 起来了，KV cache 够 ${pv(node, 'max_model_len')} × 46 条并发`,
        elapsed: '42.7s',
        blocks: [
          {
            kind: 'kv',
            rows: [
              ['engine', pv(node, 'engine')],
              ['GPU', '4 卡独占'],
              ['KV cache', `18.4 GB（gpu_memory_util=${pv(node, 'gpu_memory_util')}）`],
              ['权重同步', pv(node, 'sync_weights')],
            ],
          },
          {
            kind: 'log',
            lines: [
              '[INFO] loading weights... 18.2s',
              '[INFO] capturing cudagraph shapes [1,2,4,8,16,32]',
              '[INFO] engine ready, throughput est. 2.1k tok/s',
            ],
          },
        ],
      };

    case 'rollout':
    case 'multi_turn_rollout': {
      const gen = pv(node, 'num_generations', '8');
      const isMulti = node.type === 'multi_turn_rollout';
      return {
        summary: `采出 ${Number(gen) * 4} 条轨迹（4 prompt × ${gen}）`,
        elapsed: isMulti ? '1m 34s' : '52.1s',
        blocks: [
          {
            kind: 'kv',
            rows: [
              ['轨迹数', String(Number(gen) * 4)],
              ...(isMulti
                ? ([
                    ['平均轮数', `2.6 / 上限 ${pv(node, 'max_turns')}`],
                    ['提前 done 的比例', '78%'],
                  ] as [string, string][])
                : ([['平均长度', '312 tokens']] as [string, string][])),
              ['生成 tokens', '12.4k'],
              ['吞吐', '2.0k tok/s'],
            ],
          },
          { kind: 'text', title: '第 1 条轨迹', body: TRAJ_SAMPLE },
        ],
      };
    }

    case 'env_pool':
      return {
        summary: `${pv(node, 'game_name')} 环境池就绪，32 个 slot`,
        elapsed: '0.8s',
        blocks: [
          {
            kind: 'kv',
            rows: [
              ['env', pv(node, 'env_name')],
              ['slot 数', '32（batch 4 × num_gen 8）'],
              ['放置', pv(node, 'placement')],
              ['reset 平均耗时', '1.4ms'],
            ],
          },
          {
            kind: 'text',
            title: 'slot 0 的首个 observation',
            body: '{\n  "player": [9, 5],\n  "dealer": [7],\n  "legal_actions": ["hit", "stand"]\n}',
          },
        ],
      };

    case 'tool_schema':
      return {
        summary: '1 个工具，模型侧可见',
        elapsed: '—',
        blocks: [
          {
            kind: 'text',
            title: '发给模型的 schema',
            body: [
              '{',
              '  "name": "play",',
              '  "parameters": {',
              '    "type": "object",',
              '    "properties": {',
              '      "action": {"enum": ["hit", "stand"]}',
              '    },',
              '    "required": ["action"]',
              '  }',
              '}',
            ].join('\n'),
          },
        ],
      };

    case 'action_mapper':
    case 'code':
      return {
        summary: '函数已注册，32 次调用无异常',
        elapsed: '0.1s',
        blocks: [
          {
            kind: 'kv',
            rows: [
              ['调用次数', '32'],
              ['异常', '0'],
              ['平均耗时', '0.03ms'],
            ],
          },
          {
            kind: 'text',
            title: '一次调用',
            body: "in : ('play', {'action': 'hit'})\nout: {'action_id': 1, 'game_name': 'blackjack'}",
          },
        ],
      };

    case 'reward_fn':
    case 'reward_model':
    case 'episode_reward':
      return {
        summary: '32 条轨迹打分完成，均值 0.44',
        elapsed: '2.3s',
        blocks: [
          {
            kind: 'kv',
            rows: [
              ['来源', pv(node, 'source', pv(node, 'reward_funcs', def.label))],
              ['mean / std', '0.44 / 0.50'],
              ['min / max', '0.00 / 1.00'],
            ],
          },
          {
            kind: 'dist',
            title: '分数分布',
            bars: [
              { label: '输 (0)', value: 0.56 },
              { label: '赢 (1)', value: 0.44 },
            ],
          },
          {
            kind: 'text',
            title: '注意',
            body: '这一批只有 0 和 1 两种取值。奖励太稀疏时组内优势容易整组为 0，\n可以看看优势节点的输出确认。',
          },
        ],
      };

    case 'advantage':
    case 'rloo':
    case 'gae': {
      const gen = pv(node, 'num_generations', '8');
      return {
        summary: `4 组归一化完成，其中 1 组方差为 0`,
        elapsed: '0.2s',
        blocks: [
          {
            kind: 'kv',
            rows: [
              ['算法', pv(node, 'advantage', def.label)],
              ['组数 × 组大小', `4 × ${gen}`],
              ['全零组', '1 / 4'],
              ['优势 |值| 均值', '0.83'],
            ],
          },
          {
            kind: 'dist',
            title: '每组的组内标准差',
            bars: [
              { label: 'g0', value: 0.5 },
              { label: 'g1', value: 0.48 },
              { label: 'g2', value: 0.0 },
              { label: 'g3', value: 0.5 },
            ],
          },
          {
            kind: 'text',
            title: 'g2 为什么是 0',
            body: '这组 8 条答案全错，减掉均值之后全是 0——这一组贡献不了梯度。\n要么把题目难度拉开，要么把 num_generations 提上去。',
          },
        ],
      };
    }

    case 'traj_filter':
      return {
        summary: '32 条进，30 条出，丢了 2 条',
        elapsed: '0.1s',
        blocks: [
          {
            kind: 'kv',
            rows: [
              ['输入 / 输出', '32 / 30'],
              ['超长丢弃', '2'],
              ['可训 tokens = 0 丢弃', '0'],
              ['优势同步删除', '2（下标对齐）'],
            ],
          },
          {
            kind: 'text',
            title: '为什么两路一起删',
            body: '轨迹和优势是两个等长列表，靠下标对应。\n只删一路不会报错，但训练学到的东西全部错位——这是这个节点连线被锁住的原因。',
          },
        ],
      };

    case 'policy_loss':
    case 'grpo_loss':
    case 'ce_loss':
    case 'dpo_loss':
    case 'kl_penalty':
    case 'loss_sum':
      return {
        summary: 'loss = 0.6421',
        elapsed: '0.4s',
        blocks: [
          {
            kind: 'kv',
            rows: [
              ['loss', '0.6421'],
              ['clip_ratio', '0.07'],
              ['参与 token 数', '1284'],
              ...(node.type === 'kl_penalty'
                ? ([['kl', `0.0031（beta=${pv(node, 'beta')}）`]] as [string, string][])
                : []),
            ],
          },
          {
            kind: 'text',
            title: '怎么看 clip_ratio',
            body: '0.07 属于正常范围。如果它一直贴在上限，说明策略一步走太远，\n采样分布跟不上了——该降学习率，而不是调 epsilon。',
          },
        ],
      };

    case 'metric':
    case 'channel_metric':
      return {
        summary: '5 步已上报，8 个 key',
        elapsed: '—',
        blocks: [
          {
            kind: 'kv',
            title: '最近一步',
            rows: [
              ['reward/mean', '0.55'],
              ['reward/std', '0.49'],
              ['completion/len', '312'],
              ['clip_ratio', '0.07'],
            ],
          },
          {
            kind: 'dist',
            title: 'reward/mean 走势（step 1-5）',
            bars: [
              { label: '1', value: 0.42 },
              { label: '2', value: 0.45 },
              { label: '3', value: 0.47 },
              { label: '4', value: 0.51 },
              { label: '5', value: 0.55 },
            ],
          },
        ],
      };

    case 'optimizer':
    case 'scheduler':
      return {
        summary: `当前 lr = ${pv(node, 'lr', '9.4e-7')}`,
        elapsed: '—',
        blocks: [
          {
            kind: 'kv',
            rows: [
              ['optim', pv(node, 'optim', 'adamw_torch')],
              ['当前 lr', '9.4e-7'],
              ['schedule', pv(node, 'lr_scheduler', 'cosine')],
              ['已 warmup', '5 / 50 step'],
            ],
          },
        ],
      };

    case 'trainer':
      return {
        summary: `已跑 5 / ${pv(node, 'max_steps')} step，reward/mean 从 0.42 升到 0.55`,
        elapsed: '4m 12s',
        blocks: [
          {
            kind: 'kv',
            rows: [
              ['step', `5 / ${pv(node, 'max_steps')}`],
              ['loss', '0.6421'],
              ['grad_norm', '0.81'],
              ['显存峰值', '68.2 / 80 GB'],
              ['checkpoint', '还没到 save_steps'],
            ],
          },
          { kind: 'log', title: '训练日志', lines: TRAIN_LOG },
        ],
      };

    case 'eval':
      return {
        summary: 'gsm8k 61.5，比训练前 +3.0',
        elapsed: '6m 08s',
        blocks: [
          {
            kind: 'kv',
            rows: [
              ['gsm8k', '61.5（基线 58.5）'],
              ['math500', '24.8（基线 23.4）'],
              ['样本数', pv(node, 'limit')],
            ],
          },
          {
            kind: 'text',
            title: '一句提醒',
            body: 'limit=200 的抽样，波动大概在 ±3 分。这个涨幅还不足以下结论，\n要判断真涨假涨得把 limit 放开跑全量。',
          },
        ],
      };

    default:
      return {
        summary: `${def.label} 执行完成`,
        elapsed: '0.3s',
        blocks: [
          {
            kind: 'kv',
            title: '本次生效的参数',
            rows: (node.params ?? def.params).map((p) => [p.label, p.value] as [string, string]),
          },
        ],
      };
  }
}

/** 跑失败时的输出。故意做成能看出「该去哪查」的样子 */
export function failedOutput(node: GraphNode): NodeOutput {
  const def = NODE_TYPES[node.type];
  return {
    summary: `${def.label} 挂了`,
    elapsed: '12.4s',
    error: 'torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.62 GiB',
    blocks: [
      {
        kind: 'text',
        title: '调用栈（截断）',
        body: [
          '  File "twinkle/trainer/grpo.py", line 214, in forward_backward',
          '    loss = self.loss_fn(logits, adv)',
          '  File "torch/nn/functional.py", line 3059, in log_softmax',
          'torch.OutOfMemoryError: CUDA out of memory.',
          '  Tried to allocate 2.62 GiB. GPU 0 has 79.1 GiB, 76.8 GiB in use.',
        ].join('\n'),
      },
      {
        kind: 'kv',
        title: '出事时的现场',
        rows: [
          ['micro_batch_size', pv(node, 'micro_batch_size', '2')],
          ['最长轨迹', '4096 tokens'],
          ['显存', '76.8 / 79.1 GB'],
        ],
      },
    ],
  };
}
