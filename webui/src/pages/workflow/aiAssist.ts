import type { AiMessage, ActionState, ProposalView } from '@/components/AiChatPanel';
import { messageId } from '@/components/AiChatPanel';
import type { GraphEdge, GraphNode, NodeParam } from './nodeTypes';
import { NODE_TYPES } from './nodeTypes';

/**
 * 编排页的 AI 协助层。
 *
 * 这里放两样东西：AI 能提出的动作类型，和一个假 AI。
 * UI 组件只认 FlowMessage，接真后端时把 askAi() 换成一次请求即可，界面不用动。
 *
 * 最重要的一条设计：AI 不直接改图。
 * 它只能"提议"——每条提议是一张动作卡片，得点了应用才会落到图上。
 * 理由是这份图形是流程的唯一事实来源，如果 AI 能静默改它，
 * 「谁把 num_generations 从 8 调成 32 的」就会变成查不出来的事。
 */

/** AI 能提议的动作。故意做得很窄——只有这几种能被安全地一键应用 */
export type AiAction =
  | {
      kind: 'param';
      /** 目标节点 id */
      nodeId: string;
      /** 参数名，对应 NodeParam.label */
      label: string;
      /** 改成什么 */
      to: string;
    }
  | { kind: 'addNode'; typeKey: string; note: string }
  | { kind: 'control'; op: 'start' | 'pause' | 'resume' | 'stop' };

/** 编排页的一条消息。消息的形状是通用的，只有提议的类型是这一页特有的 */
export type FlowMessage = AiMessage<AiAction>;

/** 问 AI 时带上的上下文。有 node 就是在问单个节点，没有就是在问整张图 */
export interface AiContext {
  node?: GraphNode;
  nodes: GraphNode[];
  edges: GraphEdge[];
  presetLabel: string;
  running: boolean;
}

/** 取节点当前生效的参数（节点自己改过的优先，否则用类型默认） */
export function paramsOf(node: GraphNode): NodeParam[] {
  return node.params ?? NODE_TYPES[node.type].params;
}

function paramValue(node: GraphNode, label: string): string | undefined {
  return paramsOf(node).find((p) => p.label === label)?.value;
}

function reply(text: string, actions?: AiAction[]): FlowMessage {
  return {
    id: messageId(),
    role: 'assistant',
    text,
    actions,
    states: actions?.map(() => 'pending' as ActionState),
  };
}

/** 单个节点视角的开场白：直接把这个节点在干什么、接了什么说清楚 */
export function nodeIntro(node: GraphNode, ctx: AiContext): FlowMessage {
  const def = NODE_TYPES[node.type];
  const upstream = ctx.edges
    .filter((e) => e.to === node.id)
    .map((e) => NODE_TYPES[ctx.nodes.find((n) => n.id === e.from)?.type ?? '']?.label)
    .filter(Boolean);
  const downstream = ctx.edges
    .filter((e) => e.from === node.id)
    .map((e) => NODE_TYPES[ctx.nodes.find((n) => n.id === e.to)?.type ?? '']?.label)
    .filter(Boolean);

  const lines = [`这是「${def.label}」。${def.hint ?? ''}`];
  lines.push(
    upstream.length ? `上游接了：${[...new Set(upstream)].join('、')}` : '上游没接东西。',
  );
  lines.push(
    downstream.length ? `输出去到：${[...new Set(downstream)].join('、')}` : '输出还没接下游。',
  );
  if (def.lockedPorts) {
    lines.push('这个节点的连线是锁定的——它的两路数据必须同进同出，接错了不会报错，所以先不开放改。');
  }
  lines.push('想问什么都可以，比如为什么这么设、这个参数调大调小会怎样。');
  return reply(lines.join('\n'));
}

/** 整图视角的开场白。跟整图兜底是同一套话，统一从这里出 */
export function flowIntro(ctx: AiContext): FlowMessage {
  return askAi('', ctx);
}

/**
 * 假 AI。按关键词匹配剧本。
 *
 * 剧本不是随便编的：每条都对应一个真实会被问到的问题，
 * 回答里引用的是图上当前的真实取值，所以看着才像真在读这张图。
 */
export function askAi(question: string, ctx: AiContext): FlowMessage {
  const q = question.toLowerCase();
  const node = ctx.node;

  /* ---- 控制类：启动 / 暂停 ---- */
  if (/暂停|停一下|停下|pause/.test(q)) {
    return reply(
      ctx.running
        ? '好，我停在当前 step。已经写盘的 checkpoint 会留着，继续的时候从最近一个接上——中间那半步的梯度会丢，这是正常的。'
        : '现在没在跑，没什么可暂停的。要我起一次吗？',
      ctx.running ? [{ kind: 'control', op: 'pause' }] : [{ kind: 'control', op: 'start' }],
    );
  }
  if (/继续|恢复|resume/.test(q)) {
    return reply('从最近一个 checkpoint 接着跑。', [{ kind: 'control', op: 'resume' }]);
  }
  if (/停止|终止|别跑了|stop|kill/.test(q)) {
    return reply('整个任务停掉，不保留进程。已存的 checkpoint 不动。', [
      { kind: 'control', op: 'stop' },
    ]);
  }
  if (/跑起来|启动|开始训|运行|开跑|start|run/.test(q)) {
    return reply(
      `这张图 ${ctx.nodes.length} 个节点、${ctx.edges.length} 根线，我按拓扑顺序提交。`,
      [{ kind: 'control', op: 'start' }],
    );
  }

  /* ---- 显存 ---- */
  if (/oom|显存|内存不够|out of memory|爆了/.test(q)) {
    const trainer = ctx.nodes.find((n) => n.type === 'trainer');
    const cur = trainer ? paramValue(trainer, 'micro_batch_size') : undefined;
    const actions: AiAction[] =
      trainer && cur && Number(cur) > 1
        ? [{ kind: 'param', nodeId: trainer.id, label: 'micro_batch_size', to: '1' }]
        : [];
    return reply(
      [
        '按代价从小到大来：',
        `1. 先降 micro_batch_size${cur ? `（现在是 ${cur}）` : ''}，这个最便宜，只影响单卡一次塞多少`,
        '2. 再靠梯度累积把等效 batch 补回来，效果基本等价，就是慢一点',
        '3. 还不够就开 gradient checkpointing，拿重算换显存',
        '4. 最后才考虑换卡或者上并行',
        '',
        '顺带一句：如果是采样阶段爆的，那得去调推理引擎的 gpu_memory_util，跟训练这边不是一回事。',
      ].join('\n'),
      actions,
    );
  }

  /* ---- 优势为 0 ---- */
  if (/优势.*0|advantage.*0|全对|全错|方差/.test(q)) {
    const adv = ctx.nodes.find((n) => n.type === 'advantage');
    const numGen = adv ? paramValue(adv, 'num_generations') : undefined;
    return reply(
      [
        'GRPO 的优势是组内比较出来的：一组里减掉均值再除标准差。',
        `所以一组${numGen ? ` ${numGen} 条` : ''}答案如果全对或者全错，方差就是 0，整组优势全是 0——这一组白跑了，梯度是零。`,
        '',
        '两个方向：把题目难度拉开，让一组里有对有错；或者把 num_generations 提上去，撞到不同结果的概率大一些。',
        '要注意后者是线性涨采样成本的。',
      ].join('\n'),
    );
  }

  /* ---- 速度 ---- */
  if (/慢|快一点|速度|太久|加速/.test(q)) {
    const rollout = ctx.nodes.find(
      (n) => n.type === 'multi_turn_rollout' || n.type === 'rollout',
    );
    const gen = rollout ? paramValue(rollout, 'num_generations') : undefined;
    const actions: AiAction[] =
      rollout && gen && Number(gen) > 4
        ? [{ kind: 'param', nodeId: rollout.id, label: 'num_generations', to: '4' }]
        : [];
    return reply(
      [
        'RL 的时间基本都花在采样上，不在反向传播上。所以先看这几个：',
        `· num_generations${gen ? `（现在 ${gen}）`: ''}：这是乘数，减半就快接近一倍`,
        '· max_tokens：模型爱写长文的时候，这个比什么都影响大',
        '· max_turns：多轮场景里每多一轮就多一次完整的生成',
        '',
        '调试阶段我建议先把这三个都压到最小，把流程跑通了再放开。',
      ].join('\n'),
      actions,
    );
  }

  /* ---- 轨迹对错位 ---- */
  if (/错位|对不上|对齐|过滤|丢样本|filter/.test(q)) {
    return reply(
      [
        '这是多轮 RL 里最难查的一类问题：轨迹、奖励、优势是三个等长的列表，靠下标对应。',
        '一旦某一路被过滤掉几条而另外两路没同步删，剩下的全部错位——',
        '而且它不报错，训练照跑，只是学的东西对不上。',
        '',
        '所以画布上「轨迹过滤」的连线是锁死的：两路必须同进同出，',
        '而且优势必须在过滤之前算好。先过滤再算，组内归一化的分母就变了。',
      ].join('\n'),
    );
  }

  /* ---- 学习率 ---- */
  if (/学习率|lr|learning rate/.test(q)) {
    return reply(
      [
        'RL 微调的学习率通常要比 SFT 小一个量级——策略更新一步走太远，采样分布就跟不上了。',
        'LoRA 的话可以稍大一些，因为动的参数少。',
        '',
        '判断依据不是数值本身，是看 clip_ratio：如果它一直贴在上限，说明步子太大了。',
      ].join('\n'),
    );
  }

  /* ---- 节点上下文相关的兜底 ---- */
  if (node) {
    const def = NODE_TYPES[node.type];
    const ps = paramsOf(node);
    return reply(
      [
        `关于「${def.label}」，我现在能看到的是：`,
        ...ps.map((p) => `· ${p.label} = ${p.value}`),
        '',
        def.hint ?? '',
        '',
        '想调哪个参数直接说，我会给出建议值，你点应用才会改到图上。',
      ]
        .filter(Boolean)
        .join('\n'),
    );
  }

  /* ---- 整图兜底 ---- */
  return reply(
    [
      `当前是「${ctx.presetLabel}」，${ctx.nodes.length} 个节点、${ctx.edges.length} 根连线，${ctx.running ? '正在运行' : '还没启动'}。`,
      '',
      '我能帮的几类事：',
      '· 解释某个节点或某个参数是干什么的',
      '· 显存不够、跑得太慢、优势为 0 这类具体问题的排查顺序',
      '· 提议改参数或加节点——我只提议，你点了应用才生效',
      '· 启动、暂停、继续、停止',
      '',
      '直接说你卡在哪就行。',
    ].join('\n'),
  );
}

/**
 * 把一条提议翻成卡片上的样子。
 *
 * 翻译放在这里而不是面板里：面板是通用的，它不知道 nodeId 对应哪个节点，
 * 也不该知道——新建页那边的提议改的是表单字段，压根没有节点这回事。
 */
export function describeAction(a: AiAction, nodes: GraphNode[]): ProposalView {
  /* 控制类（启停）比改个参数重得多，给它醒目色 */
  return { label: actionLabel(a, nodes), heavy: a.kind === 'control' };
}

function actionLabel(a: AiAction, nodes: GraphNode[]): string {
  if (a.kind === 'param') {
    const n = nodes.find((x) => x.id === a.nodeId);
    const label = n ? NODE_TYPES[n.type].label : a.nodeId;
    const cur = n ? paramValue(n, a.label) : undefined;
    return `${label}：${a.label} ${cur ? `${cur} → ` : '改为 '}${a.to}`;
  }
  if (a.kind === 'addNode') {
    return `加一个「${NODE_TYPES[a.typeKey]?.label ?? a.typeKey}」节点`;
  }
  const ops = { start: '启动整图', pause: '暂停', resume: '继续', stop: '停止' };
  return ops[a.op];
}
