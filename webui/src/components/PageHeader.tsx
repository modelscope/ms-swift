import type { ReactNode } from 'react';
import type { ModuleMeta } from '@/theme/modules';
import { Column } from './Column';

/**
 * 页头。操作区不排成工具栏，而是浮到内容区右上角——「非线性」的关键：
 * 主操作单独占一个角，不与标题挤在同一条基线上。
 *
 * 字号比之前压了一档（30/25 → 24/19）。原来那套「很大很轻的标题 + 大留白」
 * 是 Codex 的路子，跟这套皮肤不是一回事：这里靠细边框和紧凑间距立层级，
 * 标题一大就散，撑不住下面密集的列表和曲线。
 */
export function PageHeader({
  module: _module,
  title,
  desc,
  extra,
  back,
  titleExtra,
  wide,
  size = 'lg',
}: {
  /** 目前没被用到，保留是为了将来按模块调页头配色；设置页这类无模块页面可不传 */
  module?: ModuleMeta;
  title: string;
  desc?: ReactNode;
  extra?: ReactNode;
  back?: ReactNode;
  /** 跟标题同一行的附属内容，如状态标 */
  titleExtra?: ReactNode;
  /** 详情页用宽列，与下方图表/日志左边缘对齐 */
  wide?: boolean;
  size?: 'lg' | 'md';
}) {
  return (
    <>
      {extra && (
        /* sticky + pointer-events-none：贴住窗口右上角，脱离标题的排版流，
           同时不挡住底下内容的鼠标事件，只有按钮本身可点 */
        <div className="pointer-events-none sticky top-0 z-3 flex justify-end gap-2 px-5 pt-3.5">
          <div className="pointer-events-auto flex items-center gap-2">{extra}</div>
        </div>
      )}

      <Column wide={wide} className={extra ? 'pt-4 pb-5' : 'pt-9 pb-5'}>
        {back}
        <div className="flex flex-wrap items-center gap-3">
          <h1
            className={
              size === 'lg'
                ? 'text-foreground min-w-0 text-2xl leading-8 font-semibold tracking-[-0.02em]'
                : 'text-foreground min-w-0 text-[19px] leading-7 font-semibold tracking-[-0.01em]'
            }
          >
            {title}
          </h1>
          {titleExtra}
        </div>
        {desc && <div className="text-muted-foreground mt-1.5 text-[13.5px] leading-5">{desc}</div>}
      </Column>
    </>
  );
}
