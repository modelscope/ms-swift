import { useEffect, useRef } from 'react';
import { contentHeight } from '@/layout/metrics';
import { neutral } from '@/theme/theme';

/**
 * 日志视图。真实实现是按字节 offset 增量拉 output.log（SSE），
 * 这里只渲染给定的行数组，滚动与着色逻辑先定下来。
 */
export function LogViewer({
  lines,
  follow = true,
  /* 192px 是详情页自己的页头和 Tab 条；底部 tips 条开关时高度自动跟着变 */
  height = contentHeight(192),
}: {
  lines: string[];
  follow?: boolean;
  height?: string | number;
}) {
  const boxRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (follow && boxRef.current) {
      boxRef.current.scrollTop = boxRef.current.scrollHeight;
    }
  }, [lines, follow]);

  const colorOf = (line: string) => {
    if (/\b(ERROR|Traceback|CUDA out of memory|FAILED)\b/.test(line)) return '#cf1322';
    if (/\bWARN(ING)?\b/.test(line)) return '#d46b08';
    if (line.startsWith('{')) return '#0958d9';
    if (/^\[\d{4}-/.test(line) || line.includes('run.sh')) return neutral.textTertiary;
    return neutral.text;
  };

  return (
    <div
      ref={boxRef}
      style={{
        height,
        overflow: 'auto',
        background: neutral.bgCode,
        border: `1px solid ${neutral.borderLight}`,
        borderRadius: 8,
        padding: '12px 0',
        fontFamily: 'ui-monospace, SFMono-Regular, Menlo, Consolas, monospace',
        fontSize: 12.5,
        lineHeight: '20px',
      }}
    >
      {lines.map((line, i) => (
        <div
          key={i}
          style={{ display: 'flex', padding: '0 14px', whiteSpace: 'pre-wrap', wordBreak: 'break-all' }}
        >
          <span
            style={{
              width: 40,
              flex: 'none',
              textAlign: 'right',
              paddingRight: 14,
              color: '#c9d1d9',
              userSelect: 'none',
            }}
          >
            {i + 1}
          </span>
          <span style={{ color: colorOf(line) }}>{line}</span>
        </div>
      ))}
    </div>
  );
}
