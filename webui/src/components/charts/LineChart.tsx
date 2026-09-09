import { useId } from 'react';
import { neutral } from '@/theme/theme';

/**
 * 极简折线图。刻意不引图表库：原型阶段只要能看出趋势，
 * 后续要换 echarts 只替换这一个组件，调用方 props 不变。
 */
export function LineChart({
  points,
  color,
  height = 240,
  yLabel,
  xLabel = 'step',
}: {
  points: Array<{ x: number; y: number }>;
  color: string;
  height?: number;
  yLabel?: string;
  xLabel?: string;
}) {
  const W = 720;
  const H = height;
  const pad = { top: 16, right: 16, bottom: 28, left: 48 };
  /**
   * 渐变的 id 必须是 useId 算出来的，不能拿颜色拼。
   *
   * 原来写的是 `grad-${color.replace('#','')}`，在色值是十六进制时没问题；
   * 但现在色值是 var(--chart-1) 这种 CSS 函数，id 就变成了 grad-var(--chart-1)，
   * fill="url(#grad-var(--chart-1))" 引用不到任何节点——SVG 里 fill 引用失败会退回初始值
   * 也就是纯黑，于是整张图被填成一块黑疴。这类错误 tsc 和 build 都查不出来。
   */
  const gradId = `chart-grad-${useId().replace(/:/g, '')}`;

  if (points.length === 0) return null;

  const xs = points.map((p) => p.x);
  const ys = points.map((p) => p.y);
  const xMin = Math.min(...xs);
  const xMax = Math.max(...xs);
  const yMin = Math.min(...ys);
  const yMax = Math.max(...ys);
  const yPad = (yMax - yMin) * 0.12 || 0.1;

  const sx = (x: number) =>
    pad.left + ((x - xMin) / (xMax - xMin || 1)) * (W - pad.left - pad.right);
  const sy = (y: number) =>
    H - pad.bottom - ((y - (yMin - yPad)) / (yMax + yPad - (yMin - yPad) || 1)) * (H - pad.top - pad.bottom);

  const path = points.map((p, i) => `${i === 0 ? 'M' : 'L'}${sx(p.x)},${sy(p.y)}`).join(' ');
  const area = `${path} L${sx(xMax)},${H - pad.bottom} L${sx(xMin)},${H - pad.bottom} Z`;

  const yTicks = Array.from({ length: 5 }, (_, i) => yMin - yPad + ((yMax + yPad - (yMin - yPad)) / 4) * i);
  const xTicks = Array.from({ length: 5 }, (_, i) => xMin + ((xMax - xMin) / 4) * i);

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: '100%', height: 'auto', display: 'block' }}>
      <defs>
        {/*
          颜色走 style 而不是 stopColor 属性：stop-color 作为属性写时对 var() 的支持
          各引擎不一致，当成 CSS 声明写则是确定生效的。
        */}
        <linearGradient id={gradId} x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" style={{ stopColor: color, stopOpacity: 0.16 }} />
          <stop offset="100%" style={{ stopColor: color, stopOpacity: 0 }} />
        </linearGradient>
      </defs>

      {yTicks.map((t, i) => (
        <g key={i}>
          <line
            x1={pad.left}
            y1={sy(t)}
            x2={W - pad.right}
            y2={sy(t)}
            stroke={neutral.borderLight}
            strokeWidth={1}
          />
          <text x={pad.left - 8} y={sy(t) + 4} textAnchor="end" fontSize={11} fill={neutral.textTertiary}>
            {t.toFixed(2)}
          </text>
        </g>
      ))}

      {xTicks.map((t, i) => (
        <text
          key={i}
          x={sx(t)}
          y={H - pad.bottom + 18}
          textAnchor="middle"
          fontSize={11}
          fill={neutral.textTertiary}
        >
          {Math.round(t)}
        </text>
      ))}

      <path d={area} fill={`url(#${gradId})`} />
      <path d={path} fill="none" style={{ stroke: color }} strokeWidth={1.8} strokeLinejoin="round" />
      <circle
        cx={sx(points[points.length - 1].x)}
        cy={sy(points[points.length - 1].y)}
        r={3.5}
        style={{ fill: color }}
      />

      {yLabel && (
        <text x={pad.left} y={12} fontSize={11} fill={neutral.textSecondary}>
          {yLabel}
        </text>
      )}
      <text x={W - pad.right} y={H - 4} textAnchor="end" fontSize={11} fill={neutral.textTertiary}>
        {xLabel}
      </text>
    </svg>
  );
}
