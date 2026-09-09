import { forwardRef, type ComponentProps, type ReactNode } from 'react';
import { Link, useNavigate, useParams } from 'react-router-dom';
import {
  ChevronLeft,
  Check,
  MoreHorizontal,
  Pause,
  Play,
  RotateCw,
  Square,
  Trash2,
  TriangleAlert,
} from 'lucide-react';
import { toast } from 'sonner';
import { cn } from 'cn';
import { Button } from '@/components/ui/button';
import { Tabs, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Tooltip, TooltipContent, TooltipTrigger } from '@/components/ui/tooltip';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu';
import { PageHeader } from './PageHeader';
import { Column } from './Column';
import { StatusTag } from './StatusTag';
import { CopyText } from './CopyText';
import type { ModuleMeta } from '@/theme/modules';
import { features } from '@/config/features';
import type { TaskItem } from '@/mock/types';

export interface DetailTab {
  key: string;
  label: string;
  content: ReactNode;
}

/** 一枚元信息：标签在上、值在下。取代 antd Descriptions 那种「标签：值」的表格观感 */
export function MetaCell({ label, value }: { label: string; value: ReactNode }) {
  return (
    <div className="min-w-0">
      <div className="text-muted-foreground mb-1 text-xs">{label}</div>
      <div className="text-foreground truncate text-[13.5px]">{value}</div>
    </div>
  );
}

/** 元信息条：一排无边框的键值对，横向铺开 */
export function MetaStrip({ children }: { children: ReactNode }) {
  return (
    <div className="border-border flex flex-wrap gap-x-9 gap-y-4 border-t pt-3.5 pb-4">
      {children}
    </div>
  );
}

/**
 * 提示条。不用 antd Alert 那种带框带图标底的样式。
 * 配色取 --state-*，跟 StatusTag 同源：同一件事在标签上和提示条上不该是两个红。
 */
export function Notice({
  tone,
  title,
  children,
}: {
  tone: 'warn' | 'error';
  title: string;
  children?: ReactNode;
}) {
  const c = tone === 'warn' ? 'var(--state-paused)' : 'var(--state-failed)';
  return (
    <div
      className="mb-4 rounded-md border px-4 py-3"
      style={{
        background: `color-mix(in oklab, ${c} 7%, var(--background))`,
        borderColor: `color-mix(in oklab, ${c} 25%, var(--background))`,
      }}
    >
      <div className="flex items-center gap-2 text-[13.5px] font-medium" style={{ color: c }}>
        <TriangleAlert size={14} /> {title}
      </div>
      {children && (
        <div className="text-muted-foreground mt-1.5 text-[13px] leading-5">{children}</div>
      )}
    </div>
  );
}

/**
 * 内容区的操作按钮。
 *
 * 原来叫 PillButton，圆角写死 999。写死就意味着换皮时它不跟着变，名字也把
 * 外观钉死了，所以改名 ActionButton，实现直接委托给 shadcn 的 Button——
 * 圆角跟着 --radius 走，这里只保留 tone 那层语义映射，页面代码不需要知道 variant 叫什么。
 *
 * 两处不能省的细节：
 *
 * 1. props 必须开放透传（...rest）而不是只列 icon/children/tone/onClick。Radix 的
 *    `<XxxTrigger asChild>` 是把开关逻辑当 props 塞给子元素的（onPointerDown、
 *    onKeyDown、aria-expanded、data-state），子元素不透传就等于把开关拆了——
 *    之前详情页的「更多」下拉点了没反应就是这个原因。
 * 2. 必须 forwardRef，理由见 ui/button.tsx 顶部那段。
 *
 * tooltip 直接做成一个 prop：这个按钮九成场合都要配一句说明，让调用方各自去拼
 * Tooltip/TooltipTrigger/TooltipContent 三层，只会把 asChild 那两个坑重新踩一遍。
 */
export const ActionButton = forwardRef<
  HTMLButtonElement,
  ComponentProps<'button'> & {
    icon?: ReactNode;
    tone?: 'primary' | 'ghost' | 'danger';
    tooltip?: ReactNode;
  }
>(({ icon, children, tone = 'ghost', tooltip, className, ...rest }, ref) => {
  const variant = tone === 'primary' ? 'default' : tone === 'danger' ? 'outline' : 'secondary';
  const btn = (
    <Button
      ref={ref}
      variant={variant}
      /* 没有文字时收成正方形图标按钮，跟有文字的那些保持同一个高度 */
      size={children ? 'default' : 'icon'}
      className={cn(tone === 'danger' && 'text-state-failed hover:text-state-failed', className)}
      {...rest}
    >
      {icon}
      {children}
    </Button>
  );

  if (!tooltip) return btn;
  return (
    <Tooltip>
      <TooltipTrigger asChild>{btn}</TooltipTrigger>
      <TooltipContent>{tooltip}</TooltipContent>
    </Tooltip>
  );
});
ActionButton.displayName = 'ActionButton';

/**
 * Tab 条。用 shadcn 的 Tabs 只取它的 TabsList——内容不放在 TabsContent 里，
 * 因为切 Tab 是走路由的（刷新后能停在同一个 Tab），内容由外层按当前路由渲染。
 * 这样做还能白拿 Radix 的左右键导航。
 */
export function TabStrip({
  tabs,
  activeKey,
  onChange,
}: {
  tabs: Array<{ key: string; label: string }>;
  activeKey: string;
  onChange: (k: string) => void;
}) {
  return (
    <Tabs value={activeKey} onValueChange={onChange} className="mb-4">
      <TabsList>
        {tabs.map((t) => (
          <TabsTrigger key={t.key} value={t.key}>
            {t.label}
          </TabsTrigger>
        ))}
      </TabsList>
    </Tabs>
  );
}

/**
 * 任务详情外壳：返回、状态、元信息、操作、Tab 切换都在这里，
 * 各模块只负责提供 Tab 内容。切 Tab 走路由，刷新后停在同一个 Tab。
 */
export function TaskDetailShell({
  module,
  tasks,
  tabs,
  activeTab,
  extraMeta,
  actions,
}: {
  module: ModuleMeta;
  tasks: TaskItem[];
  tabs: DetailTab[];
  activeTab: string;
  /** 模块特有的元信息格 */
  extraMeta?: ReactNode;
  /** 模块特有的操作按钮，放在通用操作左侧 */
  actions?: ReactNode;
}) {
  const { id } = useParams();
  const navigate = useNavigate();
  const task = tasks.find((t) => t.id === id);

  if (!task) {
    return (
      <div className="flex flex-col items-center gap-4 py-24">
        <div className="text-muted-foreground text-sm">
          找不到任务 <span className="font-mono">{id}</span>
        </div>
        <ActionButton onClick={() => navigate(module.path)}>返回{module.label}</ActionButton>
      </div>
    );
  }

  const canPause = features.pauseResume && task.type === 'train' && Boolean(task.serverUrl);
  const current = tabs.find((t) => t.key === activeTab) ?? tabs[0];

  return (
    <>
      <PageHeader
        module={module}
        wide
        size="md"
        title={task.label}
        titleExtra={<StatusTag status={task.status} />}
        back={
          <Link
            to={module.path}
            className="text-muted-foreground hover:text-foreground mb-2 inline-flex items-center gap-1 text-[12.5px] transition-colors"
          >
            <ChevronLeft size={12} /> {module.label}
          </Link>
        }
        extra={
          <>
            {actions}
            <ActionButton
              icon={<RotateCw />}
              tooltip="刷新"
              onClick={() => toast.success('已刷新（示意）')}
            />
            {task.status === 'RUNNING' && canPause && (
              <ActionButton
                icon={<Pause />}
                tooltip="暂停后服务端仍保留训练状态，显存不释放"
                onClick={() => toast.info('暂停（示意）')}
              >
                暂停
              </ActionButton>
            )}
            {task.status === 'PAUSED' && (
              <ActionButton tone="primary" icon={<Play />} onClick={() => toast.info('继续（示意）')}>
                继续
              </ActionButton>
            )}
            {task.status === 'RUNNING' && (
              <ActionButton tone="danger" icon={<Square />} onClick={() => toast.info('停止（示意）')}>
                停止
              </ActionButton>
            )}
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <ActionButton icon={<MoreHorizontal />} />
              </DropdownMenuTrigger>
              <DropdownMenuContent align="end">
                <DropdownMenuItem onClick={() => toast.info('示意，未接后端')}>
                  重命名
                </DropdownMenuItem>
                <DropdownMenuItem onClick={() => toast.info('示意，未接后端')}>
                  以此为模板新建
                </DropdownMenuItem>
                <DropdownMenuSeparator />
                <DropdownMenuItem variant="destructive" onClick={() => toast.info('示意，未接后端')}>
                  <Trash2 />
                  删除
                </DropdownMenuItem>
              </DropdownMenuContent>
            </DropdownMenu>
          </>
        }
      />

      <Column wide className="pb-10">
        {task.status === 'PAUSED' && (
          <Notice tone="warn" title="任务已暂停，但服务端仍占用显存">
            本地客户端进程已退出，服务端仍在 GPU 中保留模型与优化器状态。点「继续」可无损续跑；确定不再需要请点「停止」，那会真正释放服务端资源。
          </Notice>
        )}
        {task.status === 'FAILED' && task.error && (
          <Notice tone="error" title="任务失败">
            {task.error}
          </Notice>
        )}

        {/* 元信息条：取代 Descriptions */}
        <MetaStrip>
          <MetaCell label="模型 / 来源" value={task.model} />
          <MetaCell label="运行方式" value={task.runner} />
          <MetaCell
            label="服务端"
            value={task.serverUrl ?? <span className="text-muted-foreground">本地</span>}
          />
          <MetaCell label="创建时间" value={<span className="tabular">{task.createdAt}</span>} />
          {task.endpoint && (
            <MetaCell label="服务地址" value={<CopyText text={task.endpoint} mono />} />
          )}
          <MetaCell
            label="任务 id"
            value={<span className="font-mono text-[12.5px]">{task.id}</span>}
          />
          {extraMeta}
        </MetaStrip>

        <TabStrip
          tabs={tabs}
          activeKey={current.key}
          onChange={(k) => navigate(`${module.path}/${task.id}/${k}`)}
        />

        {current.content}
      </Column>
    </>
  );
}

/** 详情页里的内容块。细边框 + 无阴影，取代到处都是的 antd Card */
export function Panel({
  title,
  extra,
  children,
  padded = true,
}: {
  title?: ReactNode;
  extra?: ReactNode;
  children: ReactNode;
  padded?: boolean;
}) {
  return (
    <div className="bg-card border-border overflow-hidden rounded-lg border">
      {title && (
        <div className="flex items-center justify-between gap-3 px-4 pt-3">
          <span className="text-foreground text-[13.5px] font-medium">{title}</span>
          {extra}
        </div>
      )}
      <div className={padded ? 'px-4 pt-3 pb-4' : ''}>{children}</div>
    </div>
  );
}

/** 一枚大数字指标。取代 antd Card + Statistic 的组合 */
export function StatCell({
  label,
  value,
  suffix,
  hint,
}: {
  label: string;
  value: ReactNode;
  suffix?: ReactNode;
  hint?: string;
}) {
  const body = (
    <div className="border-border min-w-0 flex-1 rounded-lg border px-4 py-3">
      <div className="text-muted-foreground mb-1 text-xs">{label}</div>
      {/* tabular：指标是会跳的，数字不等宽整个卡片宽度会跟着抖 */}
      <div className="text-foreground tabular text-[23px] leading-7 font-semibold tracking-[-0.02em]">
        {value}
        {suffix && (
          <span className="text-muted-foreground ms-1.5 text-[13px] font-normal">{suffix}</span>
        )}
      </div>
    </div>
  );
  return hint ? (
    <Tooltip>
      <TooltipTrigger asChild>{body}</TooltipTrigger>
      <TooltipContent>{hint}</TooltipContent>
    </Tooltip>
  ) : (
    body
  );
}

/** 一行「已完成」式的产物条目 */
export function ArtifactRow({
  title,
  desc,
  done,
  extra,
}: {
  title: ReactNode;
  desc?: ReactNode;
  done?: boolean;
  extra?: ReactNode;
}) {
  return (
    <div className="task-row hover:bg-secondary/50 -mx-3 flex items-center gap-3 rounded-md px-3 py-3 transition-colors">
      <span
        className={cn(
          'inline-flex size-[22px] flex-none items-center justify-center rounded-sm border',
          done ? 'text-state-done' : 'border-border text-muted-foreground',
        )}
        style={
          done
            ? {
                background: 'color-mix(in oklab, var(--state-done) 8%, var(--background))',
                borderColor: 'color-mix(in oklab, var(--state-done) 28%, var(--background))',
              }
            : undefined
        }
      >
        {done ? <Check size={12} /> : ''}
      </span>
      <div className="min-w-0 flex-1">
        <div className="text-foreground text-sm">{title}</div>
        {desc && <div className="text-muted-foreground mt-0.5 text-[13px]">{desc}</div>}
      </div>
      {extra}
    </div>
  );
}
