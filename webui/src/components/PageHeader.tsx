import type { ReactNode } from 'react';
import type { ModuleMeta } from '@/theme/modules';
import { columnStyle, neutral, wideColumnStyle } from '@/theme/theme';

/**
 * 页头。字号层级参考 Codex：标题很大很轻、说明是一行灰字，靠留白建立层级。
 *
 * 操作区不排成工具栏，而是浮到内容区右上角——「非线性」的关键：
 * 主操作单独占一个角，不与标题挤在同一条基线上。
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
  const col = wide ? wideColumnStyle : columnStyle;

  return (
    <>
      {extra && (
        <div
          style={{
            display: 'flex',
            justifyContent: 'flex-end',
            gap: 8,
            padding: '14px 20px 0',
            /** 贴住窗口右上角，脱离标题的排版流 */
            position: 'sticky',
            top: 0,
            zIndex: 3,
            pointerEvents: 'none',
          }}
        >
          <div style={{ display: 'flex', gap: 8, alignItems: 'center', pointerEvents: 'auto' }}>
            {extra}
          </div>
        </div>
      )}

      <div style={{ ...col, paddingTop: extra ? 18 : 46, paddingBottom: 22 }}>
        {back}
        <div style={{ display: 'flex', alignItems: 'center', gap: 12, flexWrap: 'wrap' }}>
          <div
            style={{
              fontSize: size === 'lg' ? 30 : 25,
              fontWeight: 600,
              color: neutral.text,
              lineHeight: size === 'lg' ? '38px' : '33px',
              letterSpacing: -0.4,
              minWidth: 0,
            }}
          >
            {title}
          </div>
          {titleExtra}
        </div>
        {desc && (
          <div style={{ fontSize: 15, color: neutral.textTertiary, marginTop: 7, lineHeight: '22px' }}>
            {desc}
          </div>
        )}
      </div>
    </>
  );
}
