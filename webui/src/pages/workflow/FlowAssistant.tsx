import { useEffect, useState } from 'react';
import { Button, Drawer, Tooltip } from 'antd';
import {
  CaretRightOutlined,
  PauseOutlined,
  RobotOutlined,
  StopOutlined,
} from '@ant-design/icons';
import { brand, neutral } from '@/theme/theme';
import type { GraphEdge, GraphNode } from './nodeTypes';
import type { AiAction, AiMessage } from './aiAssist';
import { askAi, flowIntro, userMessage } from './aiAssist';
import { AiChatPanel } from './AiChatPanel';

export type RunState = 'idle' | 'running' | 'paused';

const RUN_TEXT: Record<RunState, string> = {
  idle: '未启动',
  running: '正在运行',
  paused: '已暂停',
};
const RUN_COLOR: Record<RunState, string> = {
  idle: neutral.textTertiary,
  running: '#10B981',
  paused: '#F59E0B',
};

/**
 * 整图 AI 助手抽屉（功能 3）。
 *
 * 跟节点面板的区别不在于界面，而在于上下文：这里的 AiContext 不带 node，
 * 所以 AI 回答的是「这张图」的事——启动、暂停、整体为什么慢。
 *
 * 顶上那排启停按钮是有意放在这里的：AI 提议「暂停」的时候，
 * 用户可以选择点卡片让它代劳，也可以自己按按钮。
 * 不给手动出口的话，AI 就从助手变成了唯一通道。
 */
export function FlowAssistant({
  open,
  onClose,
  nodes,
  edges,
  presetLabel,
  runState,
  confirmBeforeApply,
  onControl,
  onAction,
}: {
  open: boolean;
  onClose: () => void;
  nodes: GraphNode[];
  edges: GraphEdge[];
  presetLabel: string;
  runState: RunState;
  confirmBeforeApply: boolean;
  onControl: (op: 'start' | 'pause' | 'resume' | 'stop') => void;
  /** 非控制类提议（改参数、加节点）交给页面处理 */
  onAction: (a: AiAction) => void;
}) {
  const [messages, setMessages] = useState<AiMessage[]>([]);
  const [draft, setDraft] = useState('');
  const running = runState === 'running';

  // 第一次打开时给一句开场白。之后再开保留上次的对话
  useEffect(() => {
    if (open && messages.length === 0) {
      setMessages([flowIntro({ nodes, edges, presetLabel, running })]);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open]);

  const send = () => {
    const text = draft.trim();
    if (!text) return;
    setDraft('');
    setMessages((m) => [...m, userMessage(text), askAi(text, { nodes, edges, presetLabel, running })]);
  };

  const handleAction = (msgId: string, index: number, next: 'applied' | 'ignored') => {
    const action = messages.find((m) => m.id === msgId)?.actions?.[index];
    if (next === 'applied' && action) {
      if (action.kind === 'control') onControl(action.op);
      else onAction(action);
    }
    setMessages((ms) =>
      ms.map((m) =>
        m.id === msgId ? { ...m, states: m.states?.map((s, i) => (i === index ? next : s)) } : m,
      ),
    );
  };

  const done = nodes.filter((n) => n.status === 'done').length;
  const failed = nodes.filter((n) => n.status === 'failed').length;

  return (
    <Drawer
      open={open}
      onClose={onClose}
      width={420}
      mask={false}
      title={
        <span style={{ display: 'inline-flex', alignItems: 'center', gap: 8, fontSize: 14 }}>
          <RobotOutlined style={{ color: brand.primary }} />
          流程助手
        </span>
      }
      styles={{
        body: { padding: 0, display: 'flex', flexDirection: 'column' },
        header: { padding: '12px 16px' },
      }}
    >
      {/* 状态条 + 手动启停。AI 能做的事，用户自己也得能做 */}
      <div
        style={{
          padding: '10px 14px',
          borderBottom: `1px solid ${neutral.borderLight}`,
          background: neutral.bgSubtle,
          display: 'flex',
          alignItems: 'center',
          gap: 8,
          flexWrap: 'wrap',
        }}
      >
        <span
          style={{
            display: 'inline-flex',
            alignItems: 'center',
            gap: 5,
            fontSize: 12.5,
            color: RUN_COLOR[runState],
          }}
        >
          <span
            className={running ? 'status-dot-live' : undefined}
            style={{ width: 7, height: 7, borderRadius: '50%', background: RUN_COLOR[runState] }}
          />
          {RUN_TEXT[runState]}
        </span>
        <span style={{ fontSize: 12, color: neutral.textTertiary }}>
          {done}/{nodes.length} 已完成
          {failed > 0 && <span style={{ color: '#EF4444' }}>　{failed} 失败</span>}
        </span>

        <span style={{ marginInlineStart: 'auto', display: 'flex', gap: 6 }}>
          {runState === 'running' ? (
            <Tooltip title="停在当前 step，已写盘的 checkpoint 留着">
              <Button size="small" icon={<PauseOutlined />} onClick={() => onControl('pause')}>
                暂停
              </Button>
            </Tooltip>
          ) : (
            <Button
              size="small"
              type="primary"
              icon={<CaretRightOutlined />}
              onClick={() => onControl(runState === 'paused' ? 'resume' : 'start')}
            >
              {runState === 'paused' ? '继续' : '启动'}
            </Button>
          )}
          <Tooltip title="结束这次运行，进程不保留">
            <Button
              size="small"
              danger
              icon={<StopOutlined />}
              disabled={runState === 'idle'}
              onClick={() => onControl('stop')}
            />
          </Tooltip>
        </span>
      </div>

      <AiChatPanel
        messages={messages}
        nodes={nodes}
        draft={draft}
        onDraftChange={setDraft}
        onSend={send}
        onAction={handleAction}
        confirmBeforeApply={confirmBeforeApply}
        suggestions={
          running
            ? ['先暂停一下', '为什么这么慢', '显存不够怎么办']
            : ['帮我跑起来', '这张图在做什么', '优势为什么是 0']
        }
        placeholder="问问这张流程图……"
      />
    </Drawer>
  );
}
