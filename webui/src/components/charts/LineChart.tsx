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
  const gradId = `grad-${color.replace('#', '')}`;

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: '100%', height: 'auto', display: 'block' }}>
      <defs>
        <linearGradient id={gradId} x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor={color} stopOpacity={0.18} />
          <stop offset="100%" stopColor={color} stopOpacity={0} />
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
      <path d={path} fill="none" stroke={color} strokeWidth={1.8} strokeLinejoin="round" />
      <circle
        cx={sx(points[points.length - 1].x)}
        cy={sy(points[points.length - 1].y)}
        r={3.5}
        fill={color}
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
