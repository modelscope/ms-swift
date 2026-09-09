import { useEffect, useRef } from 'react';
import { contentHeight } from '@/layout/metrics';

/**
 * 日志视图。真实实现是按字节 offset 增量拉 output.log（SSE），
 * 这里只渲染给定的行数组，滚动与着色逻辑先定下来。
 *
 * 着色改成取 --state-* 和 --chart-*：原来四个色值是写死的十六进制，
 * 深色模式下 ERROR 那个红会发暗、行号那个 #c9d1d9 会白得刺眼。
 * 走变量之后深浅两套自动成立。
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

  /**
   * 判定顺序有讲究：ERROR 要先于 WARN，否则 "WARN: ... ERROR" 这类行会被判成警告。
   * 时间戳行放最后，因为它只是想把噪音压灰，不该盖掉前面几类。
   */
  const colorOf = (line: string) => {
    if (/\b(ERROR|Traceback|CUDA out of memory|FAILED)\b/.test(line)) return 'var(--state-failed)';
    if (/\bWARN(ING)?\b/.test(line)) return 'var(--state-paused)';
    /* 整行 JSON 一般是 metrics 落盘，跟正文区分开好扫 */
    if (line.startsWith('{')) return 'var(--chart-3)';
    if (/^\[\d{4}-/.test(line) || line.includes('run.sh')) return 'var(--muted-foreground)';
    return 'var(--foreground)';
  };

  return (
    <div
      ref={boxRef}
      className="bg-muted border-border overflow-auto border py-3 font-mono text-[12.5px] leading-5"
      style={{ height }}
    >
      {lines.map((line, i) => (
        <div key={i} className="flex px-3.5 whitespace-pre-wrap break-all">
          <span className="text-muted-foreground/45 w-10 flex-none pr-3.5 text-right select-none tabular">
            {i + 1}
          </span>
          <span style={{ color: colorOf(line) }}>{line}</span>
        </div>
      ))}
    </div>
  );
}
