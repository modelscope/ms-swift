import { useState } from 'react';
import { Check, Copy } from 'lucide-react';
import { cn } from 'cn';
import { Tooltip, TooltipContent, TooltipTrigger } from '@/components/ui/tooltip';

/**
 * 复制按钮。取代 antd 的 Typography.Text copyable。
 *
 * 复制成功不弹 toast，只把图标换成对勾 1.2 秒——这个动作一页里可能连点好几次
 * （逐个复制服务地址），每次都弹一条全局提示会很吵。
 *
 * 用 navigator.clipboard 而不是老的 execCommand：它只在 https 或 localhost 下可用，
 * WebUI 是本机起的，localhost 满足条件。失败时静默，不至于把页面搞崩。
 *
 * 单独导出是因为复制这个动作有两种长相：跟在文本后面（CopyText），
 * 和单独浮在代码块右上角（CodeBlock）。行为只应该有一份实现。
 */
export function CopyButton({ text, label = '复制' }: { text: string; label?: string }) {
  const [done, setDone] = useState(false);

  const copy = async (e: React.MouseEvent) => {
    /* 这个按钮经常落在整行可点的列表行里，不拦住就会连带跳转 */
    e.preventDefault();
    e.stopPropagation();
    try {
      await navigator.clipboard.writeText(text);
      setDone(true);
      window.setTimeout(() => setDone(false), 1200);
    } catch {
      /* 剪贴板被浏览器策略挡掉时什么也不做，文本本身仍然可以手选 */
    }
  };

  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <button
          type="button"
          aria-label={label}
          onClick={copy}
          className="text-muted-foreground/60 hover:text-foreground shrink-0 cursor-pointer transition-colors"
        >
          {done ? <Check size={12} className="text-state-done" /> : <Copy size={12} />}
        </button>
      </TooltipTrigger>
      <TooltipContent>{done ? '已复制' : label}</TooltipContent>
    </Tooltip>
  );
}

/** 文本 + 跟在后面的复制按钮。文本过长时截断，按钮不被挤走 */
export function CopyText({
  text,
  children,
  mono,
  className,
  label,
}: {
  text: string;
  children?: React.ReactNode;
  /** 服务地址、任务 id 这类要等宽 */
  mono?: boolean;
  className?: string;
  label?: string;
}) {
  return (
    <span className={cn('inline-flex min-w-0 items-center gap-1.5', mono && 'font-mono', className)}>
      <span className="truncate">{children ?? text}</span>
      <CopyButton text={text} label={label} />
    </span>
  );
}
