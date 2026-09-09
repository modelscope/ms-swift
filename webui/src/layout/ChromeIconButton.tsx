import type { ReactNode } from 'react';
import { cn } from 'cn';
import { Tooltip, TooltipContent, TooltipTrigger } from '@/components/ui/tooltip';

/**
 * 深色外壳（标题栏 / 侧栏 / tips 条）上的无框图标按钮。
 *
 * 不直接用 shadcn 的 Button variant="ghost"：那个 variant 的 hover 是 bg-accent，
 * 而 accent 属于内容区那套浅色语义，落在深色外壳上会变成一块白斑。
 * 外壳区域必须走 sidebar-* 这组变量，所以单独收一个组件，
 * 让「深色底上的 hover 反馈长什么样」只有这一处定义。
 */
export function ChromeIconButton({
  icon,
  label,
  onClick,
  href,
  size = 26,
  side = 'bottom',
  className,
}: {
  icon: ReactNode;
  /** 同时作为 tooltip 文案和无障碍名称 */
  label: string;
  onClick?: () => void;
  /** 给了就渲染成新窗口打开的链接 */
  href?: string;
  size?: number;
  /** 贴着 tips 条那一排要往上弹，否则气泡会掉到视口外 */
  side?: 'top' | 'bottom' | 'left' | 'right';
  className?: string;
}) {
  const shared = cn(
    'inline-flex shrink-0 items-center justify-center transition-colors',
    'text-sidebar-foreground/45 hover:bg-sidebar-accent hover:text-sidebar-foreground',
    'focus-visible:ring-sidebar-ring cursor-pointer focus-visible:ring-2 focus-visible:outline-none',
    className,
  );
  const box = { width: size, height: size };

  return (
    <Tooltip>
      <TooltipTrigger asChild>
        {href ? (
          <a
            href={href}
            target="_blank"
            rel="noreferrer"
            aria-label={label}
            className={shared}
            style={box}
          >
            {icon}
          </a>
        ) : (
          <button
            type="button"
            aria-label={label}
            onClick={(e) => {
              /* 侧栏里这个按钮套在 <Link> 内部，不拦住就会连带跳一次路由 */
              e.preventDefault();
              e.stopPropagation();
              onClick?.();
            }}
            className={shared}
            style={box}
          >
            {icon}
          </button>
        )}
      </TooltipTrigger>
      <TooltipContent side={side}>{label}</TooltipContent>
    </Tooltip>
  );
}
