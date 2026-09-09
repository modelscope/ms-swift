import { Tooltip } from 'antd';
import type { TaskStatus } from '@/mock/types';

/**
 * 状态标签。状态集合与设计文档一致：RUNNING / DONE / FAILED / PAUSED，没有第五种。
 * 样式走 agentic 风：小圆点 + 文字的软胶囊，无硬边框；RUNNING 的点会呼吸。
 * tooltip 写清每种状态的判定来源，避免用户把 PAUSED 当成已结束。
 */
const STATUS_STYLE: Record<
  TaskStatus,
  { text: string; color: string; bg: string; live?: boolean; hint: string }
> = {
  RUNNING: {
    text: '运行中',
    color: '#3E4585',
    bg: '#EEEFF7',
    live: true,
    hint: '进程存活，且尚未写出 exit_code',
  },
  DONE: {
    text: '已完成',
    color: '#1F6F4A',
    bg: '#EDF7F1',
    hint: 'exit_code = 0，或被用户主动停止',
  },
  FAILED: {
    text: '失败',
    color: '#B03A3A',
    bg: '#FBF0F0',
    hint: 'exit_code ≠ 0，或进程已死但没留下 exit_code（如被 OOM kill）',
  },
  PAUSED: {
    text: '已暂停',
    color: '#8A6320',
    bg: '#FAF5EA',
    hint: '本地进程已退出，但服务端仍保留训练状态，显存未释放，可继续',
  },
};

export function StatusTag({ status }: { status: TaskStatus }) {
  const s = STATUS_STYLE[status];
  return (
    <Tooltip title={s.hint}>
      <span
        style={{
          display: 'inline-flex',
          alignItems: 'center',
          gap: 6,
          padding: '1px 10px',
          borderRadius: 999,
          background: s.bg,
          color: s.color,
          fontSize: 12,
          fontWeight: 600,
          lineHeight: '20px',
          whiteSpace: 'nowrap',
        }}
      >
        <span
          className={s.live ? 'status-dot-live' : undefined}
          style={{ width: 6, height: 6, borderRadius: '50%', background: s.color, display: 'inline-block' }}
        />
        {s.text}
      </span>
    </Tooltip>
  );
}
