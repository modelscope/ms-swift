import { useEffect, useState } from 'react';
import { Bot } from 'lucide-react';
import { AiChatPanel, userMessage } from './AiChatPanel';
import { Sheet, SheetContent, SheetHeader, SheetTitle } from './ui/sheet';
import type { ConfigField, ConfigMessage } from './configAssist';
import { askConfigAi, configIntro, configSuggestions, describeConfigAction } from './configAssist';
import type { ModuleKey } from '@/theme/modules';
import { MODULES } from '@/theme/modules';

/**
 * 新建页的配置助手抽屉。四个新建页共用，入口在 ConfigFormShell 的页头上。
 *
 * 跟编排页的流程助手是同一个形态（不盖遮罩、不夺焦点、右侧 420 宽），
 * 因为要做的是同一件事：一边看着左边的表单，一边问。
 * 盖上遮罩就变成了「先关掉助手才能改表单」，那这个助手就没用了。
 *
 * 它不持有配置。fields 每次渲染都由页面重新构造，所以助手读到的永远是
 * 表单的当下状态——用户自己在左边改了一个值，下一句话里 AI 就知道。
 */
export function ConfigAssistant({
  open,
  onClose,
  module,
  fields,
  confirmBeforeApply,
}: {
  open: boolean;
  onClose: () => void;
  module: ModuleKey;
  fields: ConfigField[];
  confirmBeforeApply: boolean;
}) {
  const [messages, setMessages] = useState<ConfigMessage[]>([]);
  const [draft, setDraft] = useState('');

  // 第一次打开时给一句开场白（里面带着配置体检的结果）。之后再开保留上次的对话
  useEffect(() => {
    if (open && messages.length === 0) setMessages([configIntro({ module, fields })]);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open]);

  const send = () => {
    const text = draft.trim();
    if (!text) return;
    setDraft('');
    setMessages((m) => [...m, userMessage(text), askConfigAi(text, { module, fields })]);
  };

  const handleAction = (msgId: string, index: number, next: 'applied' | 'ignored') => {
    const action = messages.find((m) => m.id === msgId)?.actions?.[index];
    /* 回写走字段自带的 setter，助手这边不需要知道那个值最终存成了什么类型 */
    if (next === 'applied' && action) fields.find((f) => f.key === action.key)?.apply?.(action.to);
    setMessages((ms) =>
      ms.map((m) =>
        m.id === msgId ? { ...m, states: m.states?.map((s, i) => (i === index ? next : s)) } : m,
      ),
    );
  };

  const label = MODULES[module].label;

  return (
    <Sheet modal={false} open={open} onOpenChange={(o) => !o && onClose()}>
      <SheetContent
        showOverlay={false}
        className="w-105 gap-0 sm:max-w-none"
        onInteractOutside={(e) => e.preventDefault()}
      >
        <SheetHeader className="px-4 py-3">
          <SheetTitle className="flex items-center gap-2 text-sm">
            <Bot className="text-primary size-4" />
            配置助手
            <span className="text-muted-foreground text-xs font-normal">新建{label}</span>
          </SheetTitle>
        </SheetHeader>

        <AiChatPanel
          messages={messages}
          describe={(a) => describeConfigAction(a, fields)}
          draft={draft}
          onDraftChange={setDraft}
          onSend={send}
          onAction={handleAction}
          confirmBeforeApply={confirmBeforeApply}
          suggestions={configSuggestions(module)}
          placeholder={`问问这份${label}配置……`}
        />
      </SheetContent>
    </Sheet>
  );
}
