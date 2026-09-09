import type { ReactNode } from 'react';
import { Tooltip, TooltipContent, TooltipTrigger } from '@/components/ui/tooltip';

/**
 * 悬浮提示。
 *
 * Radix 的提示要写三层（Root / Trigger asChild / Content），一屏上有二十几处提示时，
 * 这三层套下来真正的内容会被压得看不见。这里收成一层，用法和原来 antd 的 `title` 一致。
 *
 * children 必须是单个能接 ref 的元素：asChild 把事件和定位挂在它身上，
 * 传字符串或 Fragment 会拿不到节点、气泡飘到页面左上角。
 *
 * 延迟不在这里配，统一由 main.tsx 的 TooltipProvider 定（400ms）——
 * 密排的图标按钮上，每个提示各有各的延迟会让人觉得界面时快时慢。
 */
export function Hint({
  title,
  side = 'top',
  children,
}: {
  /** 空值时退化成什么都不包，省掉调用方写条件渲染 */
  title?: ReactNode;
  side?: 'top' | 'right' | 'bottom' | 'left';
  children: ReactNode;
}) {
  if (!title) return <>{children}</>;
  return (
    <Tooltip>
      <TooltipTrigger asChild>{children}</TooltipTrigger>
      <TooltipContent side={side}>{title}</TooltipContent>
    </Tooltip>
  );
}
