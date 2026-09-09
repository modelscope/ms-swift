import { useEffect, useMemo, useState } from 'react';
import { Loader2, Play, RotateCw, X } from 'lucide-react';
import { cn } from 'cn';
import { Hint } from '@/components/Hint';
import { AiChatPanel, userMessage } from '@/components/AiChatPanel';
import { LogViewer } from '@/components/LogViewer';
import { SegmentedControl } from '@/components/FormField';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import type { GraphEdge, GraphNode } from './nodeTypes';
import { NODE_TYPES } from './nodeTypes';
import type { NodeOutput, OutputBlock } from './nodeOutputs';
import { failedOutput, outputOf } from './nodeOutputs';
import type { AiAction, FlowMessage } from './aiAssist';
import { askAi, describeAction, nodeIntro, paramsOf } from './aiAssist';

export type InspectorTab = 'params' | 'output' | 'ai';
type Tab = InspectorTab;

/**
 * 运行状态的说法和颜色。
 * 用 Tailwind 调色板里的 amber/emerald/red 而不是主题色：
 * 这几个是「跑得怎么样」的通用信号，跟品牌色换不换没关系。
 */
const STATUS: Record<GraphNode['status'], { text: string; tone: string }> = {
  idle: { text: '未运行', tone: 'text-muted-foreground' },
  running: { text: '正在运行', tone: 'text-amber-500' },
  done: { text: '已完成', tone: 'text-emerald-500' },
  failed: { text: '失败', tone: 'text-red-500' },
};

/**
 * 节点检查面板。功能 1（看输出）和功能 2（问 AI）都落在这里。
 *
 * 三个 tab 的顺序是有意的：参数 → 输出 → 问 AI，
 * 正好是「我设了什么 → 它跑出了什么 → 这跟我想的不一样，帮我看看」这条路径。
 *
 * AI 会话由这个面板自己持有，切到别的节点就重置——
 * 问「这个参数为什么这么设」时，上一个节点的上下文留着只会误导。
 */
export function NodeInspector({
  node,
  nodes,
  edges,
  presetLabel,
  running,
  aiEnabled,
  confirmBeforeApply,
  tab,
  onClose,
  onRun,
  onParamChange,
  onAction,
  onTabChange,
}: {
  node: GraphNode;
  nodes: GraphNode[];
  edges: GraphEdge[];
  presetLabel: string;
  running: boolean;
  aiEnabled: boolean;
  confirmBeforeApply: boolean;
  /**
   * 开在哪一页。它由外面持有，而不是这里自己存一份再跟着 prop 同步——
   * 同一个东西两处各存一份，就会出现「点节点上的问 AI，面板却停在参数页」这种事。
   */
  tab: Tab;
  onClose: () => void;
  onRun: (nodeId: string) => void;
  /**
   * 改参数。给的是下标而不是参数名——
   * traj_filter 那两个 drop_if 同名，按名字改会一次改掉两个。
   */
  onParamChange: (nodeId: string, index: number, value: string) => void;
  /** AI 提议被应用时上报。面板自己不改图 */
  onAction: (a: AiAction) => void;
  onTabChange: (t: Tab) => void;
}) {
  const def = NODE_TYPES[node.type];

  /* 设置里把 AI 关掉时，正停在 AI 页就退回参数页——不拦的话面板下半截是空的 */
  const active: Tab = tab === 'ai' && !aiEnabled ? 'params' : tab;

  const ctx = { node, nodes, edges, presetLabel, running };
  const [messages, setMessages] = useState<FlowMessage[]>(() => [nodeIntro(node, ctx)]);
  const [draft, setDraft] = useState('');

  // 换节点才重开会话。只认 node.id：拖动节点位置、切 tab 都不该把对话清掉
  useEffect(() => {
    setMessages([nodeIntro(node, { node, nodes, edges, presetLabel, running })]);
    setDraft('');
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [node.id]);

  const params = paramsOf(node);
  const output: NodeOutput | null = useMemo(() => {
    if (node.status === 'done') return outputOf(node);
    if (node.status === 'failed') return failedOutput(node);
    return null;
  }, [node]);

  const send = () => {
    const text = draft.trim();
    if (!text) return;
    setDraft('');
    setMessages((m) => [...m, userMessage(text), askAi(text, { node, nodes, edges, presetLabel, running })]);
  };

  const handleAction = (msgId: string, index: number, next: 'applied' | 'ignored') => {
    const msg = messages.find((m) => m.id === msgId);
    const action = msg?.actions?.[index];
    if (next === 'applied' && action) onAction(action);
    setMessages((ms) =>
      ms.map((m) =>
        m.id === msgId
          ? { ...m, states: m.states?.map((s, i) => (i === index ? next : s)) }
          : m,
      ),
    );
  };

  const tabs = [
    { label: '参数', value: 'params' },
    { label: '输出', value: 'output' },
    ...(aiEnabled ? [{ label: '问 AI', value: 'ai' }] : []),
  ];

  const status = STATUS[node.status];

  return (
    <div className="bg-background border-border flex h-full w-87 flex-none flex-col border-s">
      {/* 头部：节点是谁 + 现在什么状态 */}
      <div className="border-border/60 border-b px-3 pt-3 pb-2.5">
        <div className="flex items-center gap-2">
          <span
            className="inline-flex size-5.5 flex-none items-center justify-center rounded-[7px] text-white [&>svg]:size-3.5"
            style={{ background: def.color }}
          >
            {def.icon}
          </span>
          <span className="text-foreground min-w-0 flex-1 truncate text-sm font-semibold">
            {def.label}
          </span>
          <Button variant="ghost" size="icon" className="size-7" onClick={onClose}>
            <X />
          </Button>
        </div>

        <div className="mt-2 flex items-center gap-2">
          <span className={cn('inline-flex items-center gap-1.5 text-xs', status.tone)}>
            {/* bg-current：点跟着文字同色，状态色只写一处 */}
            <span className="size-1.5 rounded-full bg-current" />
            {status.text}
          </span>
          {/*
            elapsed 为 '—' 的意思是「这类节点没有单次耗时」——工具定义、指标上报、
            优化器都是声明式或常驻的。既然没有值，就别把「耗时 —」这行占着位置摆在那。
          */}
          {output && output.elapsed !== '—' && (
            <span className="text-muted-foreground text-xs">耗时 {output.elapsed}</span>
          )}

          {/*
            所有节点都能单独跑。声明式节点（模型、优化器）跑一次的意思是
            「把它准备好」——加载权重、把生效配置定下来，这也是有输出可看的。
          */}
          <Hint
            title={
              def.runnable
                ? undefined
                : '这个节点是声明式的，跑一次只是把它准备好，输出里能看到加载结果和生效配置'
            }
          >
            <Button
              size="sm"
              variant={node.status === 'done' ? 'outline' : 'default'}
              className="ms-auto h-7 text-xs"
              disabled={node.status === 'running'}
              onClick={() => onRun(node.id)}
            >
              {node.status === 'running' ? (
                <Loader2 className="animate-spin" />
              ) : node.status === 'idle' ? (
                <Play />
              ) : (
                <RotateCw />
              )}
              {node.status === 'idle' ? '运行' : node.status === 'running' ? '运行中' : '重跑'}
            </Button>
          </Hint>
        </div>
      </div>

      <div className="px-3 pt-2.5">
        <SegmentedControl block value={active} onChange={(v) => onTabChange(v as Tab)} options={tabs} />
      </div>

      {active === 'params' && (
        <div className="min-h-0 flex-1 overflow-auto px-3 pt-3 pb-4">
          {params.length === 0 ? (
            <EmptyHint>这个节点没有可调参数</EmptyHint>
          ) : (
            <div className="flex flex-col gap-2.5">
              {params.map((p, i) => (
                <label key={`${p.label}-${i}`} className="block">
                  <span className="text-muted-foreground mb-1 block text-xs">{p.label}</span>
                  <Input
                    className="h-8"
                    value={p.value}
                    onChange={(e) => onParamChange(node.id, i, e.target.value)}
                  />
                </label>
              ))}
            </div>
          )}
          {def.hint && (
            <div className="bg-secondary text-muted-foreground mt-3.5 rounded-[10px] px-3 py-2.5 text-xs leading-[19px]">
              {def.hint}
            </div>
          )}
        </div>
      )}

      {active === 'output' && (
        <div className="min-h-0 flex-1 overflow-auto px-3 pt-3 pb-4">
          {node.status === 'idle' && <EmptyHint>还没跑过，跑一次才有输出</EmptyHint>}
          {node.status === 'running' && (
            <div className="text-muted-foreground flex items-center justify-center gap-1.5 py-5 text-xs">
              <Loader2 className="size-3.5 animate-spin" /> 正在跑，跑完这里会有输出
            </div>
          )}
          {output && <OutputView output={output} />}
        </div>
      )}

      {active === 'ai' && (
        <AiChatPanel
          messages={messages}
          describe={(a) => describeAction(a, nodes)}
          draft={draft}
          onDraftChange={setDraft}
          onSend={send}
          onAction={handleAction}
          confirmBeforeApply={confirmBeforeApply}
          accent={def.color}
          suggestions={suggestionsFor(node.type)}
          placeholder={`问问「${def.label}」……`}
        />
      )}
    </div>
  );
}

/** 空状态就一行字。这里的空是「还没到时候」，不值得画一张插图去强调 */
function EmptyHint({ children }: { children: React.ReactNode }) {
  return <div className="text-muted-foreground py-6 text-center text-xs">{children}</div>;
}

/** 按节点类型给几个像样的追问。问不出问题的时候，这几个按钮就是入口 */
function suggestionsFor(type: string): string[] {
  if (type === 'trainer') return ['显存不够怎么办', '为什么这么慢', '学习率怎么定'];
  if (type === 'advantage' || type === 'rloo' || type === 'gae') return ['优势为什么是 0', '组要开多大'];
  if (type === 'traj_filter') return ['为什么连线锁着', '会不会错位'];
  if (type === 'multi_turn_rollout' || type === 'rollout') return ['为什么这么慢', '轮数怎么定'];
  return ['这个节点是干什么的', '参数怎么调'];
}

/** 输出块渲染。分块的意义就在这里——每种块长得不一样，扫一眼就知道在看什么 */
function OutputView({ output }: { output: NodeOutput }) {
  return (
    <div className="flex flex-col gap-3.5">
      <div
        className={cn(
          'rounded-[10px] border px-3 py-2.5 text-[12.5px] leading-[19px]',
          output.error
            ? 'border-red-200 bg-red-50 text-red-700'
            : 'bg-secondary border-border text-foreground',
        )}
      >
        {output.error ?? output.summary}
      </div>

      {output.blocks.map((b, i) => (
        <Block key={i} block={b} />
      ))}
    </div>
  );
}

function Block({ block }: { block: OutputBlock }) {
  const title = block.title && (
    <div className="text-muted-foreground mb-1.5 text-xs">{block.title}</div>
  );

  if (block.kind === 'kv') {
    return (
      <div>
        {title}
        <div className="flex flex-col">
          {block.rows.map(([k, v], i) => (
            <div
              key={`${k}-${i}`}
              className={cn(
                'flex gap-2.5 py-[5px] text-[12.5px]',
                i !== block.rows.length - 1 && 'border-border/60 border-b',
              )}
            >
              <span className="text-muted-foreground min-w-30 flex-none">{k}</span>
              <span className="text-foreground min-w-0 flex-1 break-all">{v}</span>
            </div>
          ))}
        </div>
      </div>
    );
  }

  if (block.kind === 'text') {
    return (
      <div>
        {title}
        <pre className="bg-muted text-foreground overflow-x-auto rounded-[10px] px-3 py-2.5 font-mono text-[11.5px] leading-[18px]">
          {block.body}
        </pre>
      </div>
    );
  }

  if (block.kind === 'dist') {
    const max = Math.max(...block.bars.map((b) => b.value), 0.0001);
    return (
      <div>
        {title}
        <div className="flex flex-col gap-1.5">
          {block.bars.map((b) => (
            <div key={b.label} className="flex items-center gap-2">
              <span className="text-muted-foreground min-w-14 flex-none text-[11.5px]">
                {b.label}
              </span>
              <span className="bg-border h-2 flex-1 overflow-hidden rounded-full">
                <span
                  className={cn(
                    'block h-full rounded-full',
                    /* 方差为 0 的那种情况要一眼看出来，所以零值给个红点宽度 */
                    b.value === 0 ? 'min-w-[3px] bg-red-500' : 'bg-primary',
                  )}
                  style={{ width: `${(b.value / max) * 100}%` }}
                />
              </span>
              <span className="text-muted-foreground min-w-8 flex-none text-end text-[11.5px]">
                {b.value.toFixed(2)}
              </span>
            </div>
          ))}
        </div>
      </div>
    );
  }

  return (
    <div>
      {title}
      <LogViewer lines={block.lines} height={196} follow={false} />
    </div>
  );
}
