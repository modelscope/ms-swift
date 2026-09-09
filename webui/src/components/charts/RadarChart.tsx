import { neutral } from '@/theme/theme';

/** 评测对比用的雷达图，同样是零依赖实现，后续可整体替换 */
export function RadarChart({
  axes,
  series,
  size = 320,
  max = 100,
}: {
  axes: string[];
  series: Array<{ name: string; values: number[]; color: string }>;
  size?: number;
  max?: number;
}) {
  const cx = size / 2;
  const cy = size / 2;
  const r = size / 2 - 42;
  const n = axes.length;

  const pt = (i: number, ratio: number) => {
    const angle = (Math.PI * 2 * i) / n - Math.PI / 2;
    return [cx + Math.cos(angle) * r * ratio, cy + Math.sin(angle) * r * ratio] as const;
  };

  return (
    <svg viewBox={`0 0 ${size} ${size}`} style={{ width: '100%', maxWidth: size, height: 'auto' }}>
      {[0.25, 0.5, 0.75, 1].map((ratio) => (
        <polygon
          key={ratio}
          points={axes.map((_, i) => pt(i, ratio).join(',')).join(' ')}
          fill="none"
          stroke={neutral.borderLight}
          strokeWidth={1}
        />
      ))}

      {axes.map((label, i) => {
        const [x, y] = pt(i, 1);
        const [lx, ly] = pt(i, 1.16);
        return (
          <g key={label}>
            <line x1={cx} y1={cy} x2={x} y2={y} stroke={neutral.borderLight} strokeWidth={1} />
            <text
              x={lx}
              y={ly + 4}
              textAnchor={Math.abs(lx - cx) < 6 ? 'middle' : lx > cx ? 'start' : 'end'}
              fontSize={11}
              fill={neutral.textSecondary}
            >
              {label}
            </text>
          </g>
        );
      })}

      {series.map((s) => (
        <g key={s.name}>
          <polygon
            points={s.values.map((v, i) => pt(i, v / max).join(',')).join(' ')}
            fill={s.color}
            fillOpacity={0.14}
            stroke={s.color}
            strokeWidth={1.8}
          />
          {s.values.map((v, i) => {
            const [x, y] = pt(i, v / max);
            return <circle key={i} cx={x} cy={y} r={2.8} fill={s.color} />;
          })}
        </g>
      ))}
    </svg>
  );
}
