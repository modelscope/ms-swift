import { useEffect, useMemo, useState } from 'react';
import { Button, Empty, Input, Segmented, Tooltip } from 'antd';
import {
  CaretRightOutlined,
  CloseOutlined,
  LoadingOutlined,
  ReloadOutlined,
} from '@ant-design/icons';
import { LogViewer } from '@/components/LogViewer';
import { brand, neutral } from '@/theme/theme';
import type { GraphEdge, GraphNode } from './nodeTypes';
import { NODE_TYPES } from './nodeTypes';
import type { NodeOutput, OutputBlock } from './nodeOutputs';
import { failedOutput, outputOf } from './nodeOutputs';
import type { AiAction, AiMessage } from './aiAssist';
import { askAi, nodeIntro, paramsOf, userMessage } from './aiAssist';
import { AiChatPanel } from './AiChatPanel';

export type InspectorTab = 'params' | 'output' | 'ai';
type Tab = InspectorTab;

const STATUS_TEXT: Record<string, string> = {
  idle: '未运行',
  running: '正在运行',
  done: '已完成',
  failed: '失败',
};
const STATUS_COLOR: Record<string, string> = {
  idle: neutral.textTertiary,
  running: '#F59E0B',
  done: '#10B981',
  failed: '#EF4444',
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
  defaultTab = 'params',
  onClose,
  onRun,
  onParamChange,
  onAction,
}: {
  node: GraphNode;
  nodes: GraphNode[];
  edges: GraphEdge[];
  presetLabel: string;
  running: boolean;
  aiEnabled: boolean;
  confirmBeforeApply: boolean;
  defaultTab?: Tab;
  onClose: () => void;
  onRun: (nodeId: string) => void;
  /**
   * 改参数。给的是下标而不是参数名——
   * traj_filter 那两个 drop_if 同名，按名字改会一次改掉两个。
   */
  onParamChange: (nodeId: string, index: number, value: string) => void;
  /** AI 提议被应用时上报。面板自己不改图 */
  onAction: (a: AiAction) => void;
}) {
  const def = NODE_TYPES[node.type];
  const [tab, setTab] = useState<Tab>(defaultTab);

  // 换节点时：回到默认 tab，AI 会话重开
  const ctx = { node, nodes, edges, presetLabel, running };
  const [messages, setMessages] = useState<AiMessage[]>(() => [nodeIntro(node, ctx)]);
  const [draft, setDraft] = useState('');
  useEffect(() => {
    setTab(defaultTab);
    setMessages([nodeIntro(node, { node, nodes, edges, presetLabel, running })]);
    setDraft('');
    // 只认 node.id：拖动节点位置不该把对话清掉
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [node.id, defaultTab]);

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

  const tabs: { label: string; value: Tab }[] = [
    { label: '参数', value: 'params' },
    { label: '输出', value: 'output' },
    ...(aiEnabled ? [{ label: '问 AI', value: 'ai' as Tab }] : []),
  ];

  return (
    <div
      style={{
        width: 348,
        flex: 'none',
        height: '100%',
        display: 'flex',
        flexDirection: 'column',
        background: '#fff',
        borderInlineStart: `1px solid ${neutral.border}`,
      }}
    >
      {/* 头部：节点是谁 + 现在什么状态 */}
      <div style={{ padding: '12px 12px 10px', borderBottom: `1px solid ${neutral.borderLight}` }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <span
            style={{
              width: 22,
              height: 22,
              borderRadius: 7,
              flex: 'none',
              background: def.color,
              color: '#fff',
              fontSize: 12,
              display: 'inline-flex',
              alignItems: 'center',
              justifyContent: 'center',
            }}
          >
            {def.icon}
          </span>
          <span style={{ fontSize: 14, fontWeight: 600, color: neutral.text, flex: 1, minWidth: 0 }}>
            {def.label}
          </span>
          <Button type="text" size="small" icon={<CloseOutlined />} onClick={onClose} />
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginTop: 9 }}>
          <span
            style={{
              display: 'inline-flex',
              alignItems: 'center',
              gap: 5,
              fontSize: 12,
              color: STATUS_COLOR[node.status],
            }}
          >
            <span
              style={{
                width: 6,
                height: 6,
                borderRadius: '50%',
                background: STATUS_COLOR[node.status],
              }}
            />
            {STATUS_TEXT[node.status]}
          </span>
          {output && (
            <span style={{ fontSize: 12, color: neutral.textTertiary }}>耗时 {output.elapsed}</span>
          )}
          <span style={{ marginInlineStart: 'auto' }}>
            {/*
              所有节点都能单独跑。声明式节点（模型、优化器）跑一次的意思是
              「把它准备好」——加载权重、把生效配置定下来，这也是有输出可看的。
            */}
            <Tooltip
              title={
                def.runnable
                  ? undefined
                  : '这个节点是声明式的，跑一次只是把它准备好，输出里能看到加载结果和生效配置'
              }
            >
              <Button
                size="small"
                type={node.status === 'done' ? 'default' : 'primary'}
                icon={
                  node.status === 'running' ? (
                    <LoadingOutlined />
                  ) : node.status === 'idle' ? (
                    <CaretRightOutlined />
                  ) : (
                    <ReloadOutlined />
                  )
                }
                disabled={node.status === 'running'}
                onClick={() => onRun(node.id)}
                style={{ fontSize: 12 }}
              >
                {node.status === 'idle' ? '运行' : node.status === 'running' ? '运行中' : '重跑'}
              </Button>
            </Tooltip>
          </span>
        </div>
      </div>

      <div style={{ padding: '10px 12px 0' }}>
        <Segmented
          size="small"
          block
          value={tab}
          onChange={(v) => setTab(v as Tab)}
          options={tabs}
        />
      </div>

      {tab === 'params' && (
        <div style={{ flex: 1, minHeight: 0, overflow: 'auto', padding: '12px 12px 16px' }}>
          {params.length === 0 ? (
            <Empty
              image={Empty.PRESENTED_IMAGE_SIMPLE}
              description={<span style={{ fontSize: 12 }}>这个节点没有可调参数</span>}
            />
          ) : (
            <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
              {params.map((p, i) => (
                <div key={`${p.label}-${i}`}>
                  <div style={{ fontSize: 12, color: neutral.textSecondary, marginBottom: 4 }}>
                    {p.label}
                  </div>
                  <Input
                    size="small"
                    value={p.value}
                    onChange={(e) => onParamChange(node.id, i, e.target.value)}
                  />
                </div>
              ))}
            </div>
          )}
          {def.hint && (
            <div
              style={{
                marginTop: 14,
                padding: '9px 11px',
                borderRadius: 10,
                background: neutral.bgSubtle,
                fontSize: 12,
                lineHeight: '19px',
                color: neutral.textSecondary,
              }}
            >
              {def.hint}
            </div>
          )}
        </div>
      )}

      {tab === 'output' && (
        <div style={{ flex: 1, minHeight: 0, overflow: 'auto', padding: '12px 12px 16px' }}>
          {node.status === 'idle' && (
            <Empty
              image={Empty.PRESENTED_IMAGE_SIMPLE}
              description={<span style={{ fontSize: 12 }}>还没跑过，跑一次才有输出</span>}
            />
          )}
          {node.status === 'running' && (
            <div style={{ fontSize: 12, color: neutral.textSecondary, padding: '20px 0', textAlign: 'center' }}>
              <LoadingOutlined /> 正在跑，跑完这里会有输出
            </div>
          )}
          {output && <OutputView output={output} />}
        </div>
      )}

      {tab === 'ai' && (
        <AiChatPanel
          messages={messages}
          nodes={nodes}
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
    <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
      <div
        style={{
          padding: '9px 11px',
          borderRadius: 10,
          fontSize: 12.5,
          lineHeight: '19px',
          background: output.error ? '#FEF2F2' : brand.soft,
          border: `1px solid ${output.error ? '#FECACA' : brand.softBorder}`,
          color: output.error ? '#B91C1C' : neutral.text,
        }}
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
    <div style={{ fontSize: 12, color: neutral.textTertiary, marginBottom: 6 }}>{block.title}</div>
  );

  if (block.kind === 'kv') {
    return (
      <div>
        {title}
        <div style={{ display: 'flex', flexDirection: 'column' }}>
          {block.rows.map(([k, v], i) => (
            <div
              key={`${k}-${i}`}
              style={{
                display: 'flex',
                gap: 10,
                padding: '5px 0',
                fontSize: 12.5,
                borderBottom: i === block.rows.length - 1 ? undefined : `1px solid ${neutral.borderLight}`,
              }}
            >
              <span style={{ color: neutral.textSecondary, flex: 'none', minWidth: 118 }}>{k}</span>
              <span style={{ color: neutral.text, flex: 1, minWidth: 0, wordBreak: 'break-all' }}>{v}</span>
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
        <pre
          style={{
            margin: 0,
            padding: '9px 11px',
            borderRadius: 10,
            background: neutral.bgCode,
            fontSize: 11.5,
            lineHeight: '18px',
            color: neutral.text,
            fontFamily: 'ui-monospace, SFMono-Regular, Menlo, monospace',
            whiteSpace: 'pre',
            overflowX: 'auto',
          }}
        >
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
        <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
          {block.bars.map((b) => (
            <div key={b.label} style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              <span style={{ fontSize: 11.5, color: neutral.textSecondary, flex: 'none', minWidth: 54 }}>
                {b.label}
              </span>
              <span
                style={{
                  flex: 1,
                  height: 8,
                  borderRadius: 4,
                  background: neutral.borderLight,
                  overflow: 'hidden',
                }}
              >
                <span
                  style={{
                    display: 'block',
                    width: `${(b.value / max) * 100}%`,
                    height: '100%',
                    borderRadius: 4,
                    /** 方差为 0 的那种情况要一眼看出来，所以零值给个红点宽度 */
                    background: b.value === 0 ? '#EF4444' : brand.primary,
                    minWidth: b.value === 0 ? 3 : undefined,
                  }}
                />
              </span>
              <span style={{ fontSize: 11.5, color: neutral.textTertiary, flex: 'none', minWidth: 32, textAlign: 'right' }}>
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
