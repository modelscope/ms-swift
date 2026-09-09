import { useEffect, useRef } from 'react';
import { ArrowUp, Bot, Check, User, X, Zap } from 'lucide-react';
import { cn } from 'cn';
import { Hint } from '@/components/Hint';
import { Button } from '@/components/ui/button';
import { Textarea } from '@/components/ui/textarea';
import type { GraphNode } from './nodeTypes';
import type { ActionState, AiAction, AiMessage } from './aiAssist';
import { actionLabel } from './aiAssist';

/**
 * 编排页的 AI 对话面板。功能 2（问单个节点）和功能 3（问整张图）共用这一个组件。
 *
 * 气泡样式跟对话页保持一致：右侧用户淡紫、左侧助手浅灰、26px 头像。
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
  accent = 'var(--primary)',
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
  /** 头像颜色。节点面板里传节点主色，认得出这段对话是在问谁 */
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
    <div className="flex min-h-0 flex-1 flex-col">
      <div className="flex min-h-0 flex-1 flex-col gap-3.5 overflow-auto px-3.5 pt-3.5 pb-1">
        {messages.map((m) => (
          <div key={m.id}>
            <div className={cn('flex gap-2.5', m.role === 'user' && 'flex-row-reverse')}>
              <span
                className="inline-flex size-6.5 flex-none items-center justify-center rounded-full text-white [&>svg]:size-3.5"
                style={{ background: m.role === 'user' ? '#e5e7eb' : accent }}
              >
                {m.role === 'user' ? <User className="text-neutral-500" /> : <Bot />}
              </span>
              {/* pre-wrap：剧本里是带换行的多行文本，得让它保留 */}
              <div
                className={cn(
                  'text-foreground max-w-[86%] rounded-xl px-3 py-2 text-[13px] leading-[21px] whitespace-pre-wrap',
                  m.role === 'user' ? 'bg-secondary' : 'bg-muted',
                )}
              >
                {m.pending ? <span className="text-muted-foreground">正在想……</span> : m.text}
              </div>
            </div>

            {/* 动作卡片。缩进到跟气泡左边缘对齐 */}
            {!!m.actions?.length && (
              <div className="mt-2 ms-9 flex flex-col gap-1.5">
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
        <div className="flex flex-wrap gap-1.5 px-3.5 pt-1.5">
          {suggestions.map((s) => (
            <Button
              key={s}
              variant="outline"
              size="sm"
              className="text-muted-foreground h-7 rounded-full text-xs"
              onClick={() => {
                onDraftChange(s);
                // 让 onSend 拿到的是这条，而不是上一帧的 draft
                requestAnimationFrame(onSend);
              }}
            >
              {s}
            </Button>
          ))}
        </div>
      )}

      <div className="border-border/60 border-t p-3">
        <div className="bg-background border-border flex items-end gap-2 rounded-[14px] border px-2.5 py-2">
          {/* 高度靠 textarea 自带的 field-sizing-content 长，max-h 兜住上限 */}
          <Textarea
            value={draft}
            onChange={(e) => onDraftChange(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                send();
              }
            }}
            placeholder={placeholder}
            className="max-h-28 min-h-6 resize-none border-0 bg-transparent p-0 text-[13px] shadow-none focus-visible:ring-0"
          />
          <Button size="icon" className="size-7 flex-none rounded-full" disabled={!draft.trim()} onClick={send}>
            <ArrowUp />
          </Button>
        </div>
        <div className="text-muted-foreground mt-1.5 text-[11px]">回答是本地示意，未接后端模型</div>
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
  const done = state !== 'pending';

  return (
    <div
      className={cn(
        'flex items-center gap-2 rounded-[10px] border px-2.5 py-1.5 text-xs',
        done ? 'bg-muted border-border/60 text-muted-foreground' : 'bg-secondary border-border text-foreground',
      )}
    >
      <Zap
        className={cn(
          'size-3 flex-none',
          done
            ? 'text-muted-foreground'
            : /* 控制类动作（启动/暂停）比改参数更重，给个更醒目的颜色 */
              action.kind === 'control'
              ? 'text-amber-600'
              : 'text-primary',
        )}
      />
      <span className={cn('min-w-0 flex-1', state === 'ignored' && 'line-through')}>
        {actionLabel(action, nodes)}
      </span>

      {state === 'pending' ? (
        <span className="flex flex-none gap-1">
          <Button size="sm" className="h-6 px-2.5 text-xs" onClick={onApply}>
            应用
          </Button>
          <Hint title="不采纳这条建议">
            <Button variant="ghost" size="icon" className="size-6" onClick={onIgnore}>
              <X />
            </Button>
          </Hint>
        </span>
      ) : (
        <span className="text-muted-foreground inline-flex flex-none items-center gap-1 text-[11px]">
          {state === 'applied' ? (
            <>
              <Check className="size-2.5" />
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
