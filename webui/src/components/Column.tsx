import type { ReactNode } from 'react';
import { cn } from 'cn';

/**
 * 内容列。
 *
 * 内容不铺满整个宽度，而是居中收在一条列里——这是「应用」而不是「后台管理」
 * 观感的来源之一，铺满边到边就会变回控制台。
 *
 * 原来这是 theme.ts 里的三个常量（column / columnStyle / wideColumnStyle）+ 四处
 * inline style。收成组件的理由不是少写几个字，而是宽度这件事必须只有一个出处：
 * 页头和正文是两个相邻的块，它们的左边缘必须严格对齐，差 1px 都看得出来。
 */
export function Column({
  wide,
  className,
  children,
}: {
  /** 详情页用宽列：要放曲线和日志，880 太窄 */
  wide?: boolean;
  className?: string;
  children: ReactNode;
}) {
  return (
    <div
      className={cn('mx-auto w-full px-8', wide ? 'max-w-[1240px]' : 'max-w-[880px]', className)}
    >
      {children}
    </div>
  );
}
