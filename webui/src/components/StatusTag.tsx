import { Tooltip, TooltipContent, TooltipTrigger } from '@/components/ui/tooltip';
import type { TaskStatus } from '@/mock/types';

/**
 * 状态标签。状态集合与设计文档一致：RUNNING / DONE / FAILED / PAUSED，没有第五种。
 *
 * 原来是无边框的软胶囊（圆角 999 + 浅底色），现在是细边框 + 极淡底色的标记。
 * 圆角走 rounded-sm 而不是不写：以前靠「不写圆角类」就能得到直角，那是因为当时
 * --radius 是 0；现在皮肤的 radius 是 0.5rem，不写的话全站都圆了只剩它是方的。
 * 这里取比默认小一档的 sm：标签只有 18px 行高，用 8px 圆角会显得胖。
 *
 * 颜色不再写死在这个文件里，而是取 index.css 的 --state-*，
 * 好处是画布里的节点边框、连线动画和这里天然同色——状态色只有一处定义。
 * 底色和边框由同一个变量 color-mix 出来，深色模式下会跟着一起变，不用写两套。
 *
 * tooltip 写清每种状态的判定来源，避免用户把 PAUSED 当成已结束。
 */
const STATUS_META: Record<TaskStatus, { text: string; token: string; live?: boolean; hint: string }> =
  {
    RUNNING: {
      text: '运行中',
      token: '--state-running',
      live: true,
      hint: '进程存活，且尚未写出 exit_code',
    },
    DONE: {
      text: '已完成',
      token: '--state-done',
      hint: 'exit_code = 0，或被用户主动停止',
    },
    FAILED: {
      text: '失败',
      token: '--state-failed',
      hint: 'exit_code ≠ 0，或进程已死但没留下 exit_code（如被 OOM kill）',
    },
    PAUSED: {
      text: '已暂停',
      token: '--state-paused',
      hint: '本地进程已退出，但服务端仍保留训练状态，显存未释放，可继续',
    },
  };

export function StatusTag({ status }: { status: TaskStatus }) {
  const s = STATUS_META[status];
  const c = `var(${s.token})`;

  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <span
          className="inline-flex cursor-default items-center gap-1.5 rounded-sm border px-2 text-[11.5px] leading-[18px] font-medium whitespace-nowrap"
          style={{
            color: c,
            /* 8% / 28% 是试出来的：底色要能看出色相但不能盖过文字，边框要能勾出形状 */
            background: `color-mix(in oklab, ${c} 8%, var(--background))`,
            borderColor: `color-mix(in oklab, ${c} 28%, var(--background))`,
          }}
        >
          <span
            className={s.live ? 'status-dot-live' : undefined}
            style={{ width: 5, height: 5, borderRadius: '50%', background: c, flex: 'none' }}
          />
          {s.text}
        </span>
      </TooltipTrigger>
      <TooltipContent>{s.hint}</TooltipContent>
    </Tooltip>
  );
}
