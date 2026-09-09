import { useEffect, useState } from 'react';
import { Bot, Pause, Play, Square } from 'lucide-react';
import { cn } from 'cn';
import { Hint } from '@/components/Hint';
import { Button } from '@/components/ui/button';
import { Sheet, SheetContent, SheetHeader, SheetTitle } from '@/components/ui/sheet';
import type { GraphEdge, GraphNode } from './nodeTypes';
import type { AiAction, AiMessage } from './aiAssist';
import { askAi, flowIntro, userMessage } from './aiAssist';
import { AiChatPanel } from './AiChatPanel';

export type RunState = 'idle' | 'running' | 'paused';

const RUN: Record<RunState, { text: string; tone: string }> = {
  idle: { text: '未启动', tone: 'text-muted-foreground' },
  running: { text: '正在运行', tone: 'text-emerald-500' },
  paused: { text: '已暂停', tone: 'text-amber-500' },
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
  const state = RUN[runState];

  return (
    /*
      不盖遮罩、不夺焦点：开着助手还要能看着画布上的节点一个个点亮。
      onInteractOutside 也要拦下来，不然点一下画布抽屉就自己关了。
    */
    <Sheet modal={false} open={open} onOpenChange={(o) => !o && onClose()}>
      <SheetContent
        showOverlay={false}
        className="w-105 gap-0 sm:max-w-none"
        onInteractOutside={(e) => e.preventDefault()}
      >
        <SheetHeader className="px-4 py-3">
          <SheetTitle className="flex items-center gap-2 text-sm">
            <Bot className="text-primary size-4" />
            流程助手
          </SheetTitle>
        </SheetHeader>

        {/* 状态条 + 手动启停。AI 能做的事，用户自己也得能做 */}
        <div className="border-border/60 bg-secondary flex flex-wrap items-center gap-2 border-y px-3.5 py-2.5">
          <span className={cn('inline-flex items-center gap-1.5 text-[12.5px]', state.tone)}>
            <span
              className={cn('size-[7px] rounded-full bg-current', running && 'status-dot-live')}
            />
            {state.text}
          </span>
          <span className="text-muted-foreground text-xs">
            {done}/{nodes.length} 已完成
            {failed > 0 && <span className="text-red-500">　{failed} 失败</span>}
          </span>

          <span className="ms-auto flex gap-1.5">
            {running ? (
              <Hint title="停在当前 step，已写盘的 checkpoint 留着">
                <Button variant="outline" size="sm" className="h-7 text-xs" onClick={() => onControl('pause')}>
                  <Pause />
                  暂停
                </Button>
              </Hint>
            ) : (
              <Button
                size="sm"
                className="h-7 text-xs"
                onClick={() => onControl(runState === 'paused' ? 'resume' : 'start')}
              >
                <Play />
                {runState === 'paused' ? '继续' : '启动'}
              </Button>
            )}
            <Hint title="结束这次运行，进程不保留">
              <Button
                variant="outline"
                size="icon"
                className="text-destructive hover:text-destructive size-7"
                disabled={runState === 'idle'}
                onClick={() => onControl('stop')}
              >
                <Square />
              </Button>
            </Hint>
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
      </SheetContent>
    </Sheet>
  );
}
