import { useEffect, useRef } from 'react';
import { Avatar, Button, Input, Tooltip } from 'antd';
import {
  ArrowUpOutlined,
  CheckOutlined,
  CloseOutlined,
  RobotOutlined,
  ThunderboltOutlined,
  UserOutlined,
} from '@ant-design/icons';
import { brand, neutral } from '@/theme/theme';
import type { GraphNode } from './nodeTypes';
import type { ActionState, AiAction, AiMessage } from './aiAssist';
import { actionLabel } from './aiAssist';

/**
 * 编排页的 AI 对话面板。功能 2（问单个节点）和功能 3（问整张图）共用这一个组件。
 *
 * 气泡样式跟对话页保持一致：右侧用户淡紫、左侧助手浅灰、28px 头像。
 * 不重新设计一套是有意的——同一个产品里两处对话长得不一样，
 * 用户会以为它们是两个不同的东西。
 *
 * 完全受控：自己不存消息、不调 askAi。谁用它谁负责推进对话，
 * 这样节点面板和整图抽屉可以各存一份互不干扰的会话。
 */
export function AiChatPanel({
  messages,
  nodes,
  onSend,
  onAction,
  draft,
  onDraftChange,
  suggestions,
  confirmBeforeApply,
  accent = brand.primary,
  placeholder = '问点什么……',
}: {
  messages: AiMessage[];
  /** 用来把动作里的 nodeId 翻成人看得懂的节点名 */
  nodes: GraphNode[];
  onSend: () => void;
  onAction: (msgId: string, index: number, next: Exclude<ActionState, 'pending'>) => void;
  draft: string;
  onDraftChange: (v: string) => void;
  /** 快捷追问。空着就不显示那一排 */
  suggestions?: string[];
  /** 来自偏好设置。关掉时动作卡片不再等确认，直接呈现为已应用 */
  confirmBeforeApply: boolean;
  accent?: string;
  placeholder?: string;
}) {
  const tail = useRef<HTMLDivElement>(null);

  // 新消息进来滚到底。依赖里带上最后一条的 id，避免只改 states 时也滚
  const lastId = messages[messages.length - 1]?.id;
  useEffect(() => {
    tail.current?.scrollIntoView({ behavior: 'smooth', block: 'end' });
  }, [lastId, messages.length]);

  const send = () => {
    if (draft.trim()) onSend();
  };

  return (
    <div style={{ flex: 1, minHeight: 0, display: 'flex', flexDirection: 'column' }}>
      <div
        style={{
          flex: 1,
          minHeight: 0,
          overflow: 'auto',
          padding: '14px 14px 4px',
          display: 'flex',
          flexDirection: 'column',
          gap: 14,
        }}
      >
        {messages.map((m) => (
          <div key={m.id}>
            <div
              style={{
                display: 'flex',
                gap: 9,
                flexDirection: m.role === 'user' ? 'row-reverse' : 'row',
              }}
            >
              <Avatar
                size={26}
                style={{ flex: 'none', background: m.role === 'user' ? '#e5e7eb' : accent }}
                icon={m.role === 'user' ? <UserOutlined /> : <RobotOutlined />}
              />
              <div
                style={{
                  maxWidth: '86%',
                  padding: '8px 12px',
                  borderRadius: 12,
                  fontSize: 13,
                  lineHeight: '21px',
                  background: m.role === 'user' ? brand.soft : neutral.bgSubtle,
                  color: neutral.text,
                  /** 剧本里是带换行的多行文本，得让它保留 */
                  whiteSpace: 'pre-wrap',
                }}
              >
                {m.pending ? <span style={{ color: neutral.textTertiary }}>正在想……</span> : m.text}
              </div>
            </div>

            {/* 动作卡片。缩进到跟气泡左边缘对齐 */}
            {!!m.actions?.length && (
              <div
                style={{
                  marginTop: 8,
                  marginInlineStart: 35,
                  display: 'flex',
                  flexDirection: 'column',
                  gap: 6,
                }}
              >
                {m.actions.map((a, i) => (
                  <ActionCard
                    key={i}
                    action={a}
                    nodes={nodes}
                    state={confirmBeforeApply ? (m.states?.[i] ?? 'pending') : 'applied'}
                    onApply={() => onAction(m.id, i, 'applied')}
                    onIgnore={() => onAction(m.id, i, 'ignored')}
                  />
                ))}
              </div>
            )}
          </div>
        ))}
        <div ref={tail} />
      </div>

      {!!suggestions?.length && (
        <div
          style={{
            display: 'flex',
            flexWrap: 'wrap',
            gap: 6,
            padding: '6px 14px 0',
          }}
        >
          {suggestions.map((s) => (
            <Button
              key={s}
              size="small"
              shape="round"
              onClick={() => {
                onDraftChange(s);
                // 让 onSend 拿到的是这条，而不是上一帧的 draft
                requestAnimationFrame(onSend);
              }}
              style={{
                fontSize: 12,
                color: neutral.textSecondary,
                borderColor: neutral.border,
              }}
            >
              {s}
            </Button>
          ))}
        </div>
      )}

      <div style={{ padding: 12, borderTop: `1px solid ${neutral.borderLight}` }}>
        <div
          style={{
            background: '#fff',
            border: `1px solid ${neutral.border}`,
            borderRadius: 14,
            padding: '8px 10px',
            display: 'flex',
            alignItems: 'flex-end',
            gap: 8,
          }}
        >
          <Input.TextArea
            variant="borderless"
            value={draft}
            onChange={(e) => onDraftChange(e.target.value)}
            onPressEnter={(e) => {
              if (!e.shiftKey) {
                e.preventDefault();
                send();
              }
            }}
            placeholder={placeholder}
            autoSize={{ minRows: 1, maxRows: 5 }}
            style={{ fontSize: 13, padding: 0, resize: 'none' }}
          />
          <Button
            type="primary"
            size="small"
            shape="circle"
            icon={<ArrowUpOutlined />}
            disabled={!draft.trim()}
            onClick={send}
          />
        </div>
        <div style={{ fontSize: 11, color: neutral.textTertiary, marginTop: 6 }}>
          回答是本地示意，未接后端模型
        </div>
      </div>
    </div>
  );
}

/**
 * 一条提议。
 *
 * 关键在于它长得不像一句话，而像一个待办：有边框、有明确的两个出口。
 * AI 说「建议把 micro_batch_size 降到 1」和真把它降到 1 之间，
 * 必须隔着一次点击——图是流程的唯一事实来源，不能被静默改写。
 */
function ActionCard({
  action,
  nodes,
  state,
  onApply,
  onIgnore,
}: {
  action: AiAction;
  nodes: GraphNode[];
  state: ActionState;
  onApply: () => void;
  onIgnore: () => void;
}) {
  const label = actionLabel(action, nodes);
  const isControl = action.kind === 'control';
  const done = state !== 'pending';

  return (
    <div
      style={{
        display: 'flex',
        alignItems: 'center',
        gap: 8,
        padding: '7px 10px',
        borderRadius: 10,
        fontSize: 12,
        border: `1px solid ${done ? neutral.borderLight : brand.softBorder}`,
        background: done ? neutral.bgSubtle : brand.soft,
        color: state === 'ignored' ? neutral.textTertiary : neutral.text,
      }}
    >
      <ThunderboltOutlined
        style={{ flex: 'none', fontSize: 12, color: done ? neutral.textTertiary : accentOf(isControl) }}
      />
      <span
        style={{
          flex: 1,
          minWidth: 0,
          textDecoration: state === 'ignored' ? 'line-through' : undefined,
        }}
      >
        {label}
      </span>

      {state === 'pending' ? (
        <span style={{ flex: 'none', display: 'flex', gap: 4 }}>
          <Button size="small" type="primary" style={{ fontSize: 12 }} onClick={onApply}>
            应用
          </Button>
          <Tooltip title="不采纳这条建议">
            <Button size="small" type="text" icon={<CloseOutlined />} onClick={onIgnore} />
          </Tooltip>
        </span>
      ) : (
        <span
          style={{
            flex: 'none',
            fontSize: 11,
            color: neutral.textTertiary,
            display: 'inline-flex',
            alignItems: 'center',
            gap: 3,
          }}
        >
          {state === 'applied' ? (
            <>
              <CheckOutlined style={{ fontSize: 10 }} />
              已应用
            </>
          ) : (
            '已忽略'
          )}
        </span>
      )}
    </div>
  );
}

/** 控制类动作（启动/暂停）比改参数更重，给个更醒目的颜色 */
function accentOf(isControl: boolean): string {
  return isControl ? '#D97706' : brand.primary;
}
