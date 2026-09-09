import { useMemo, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import type { ReactNode } from 'react';
import {
  CloudCog,
  FileText,
  LineChart,
  MoreHorizontal,
  Network,
  Package,
  Pause,
  Play,
  Plus,
  RotateCw,
  Rocket,
  Search,
  Square,
  Trash2,
  Workflow,
  Zap,
} from 'lucide-react';
import { toast } from 'sonner';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Tooltip, TooltipContent, TooltipTrigger } from '@/components/ui/tooltip';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu';
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
  AlertDialogTrigger,
} from '@/components/ui/alert-dialog';
import { PageHeader } from './PageHeader';
import { StatusTag } from './StatusTag';
import { CopyText } from './CopyText';
import { Column } from './Column';
import type { ModuleKey, ModuleMeta } from '@/theme/modules';
import { taskDetailPath } from '@/theme/modules';
import { features } from '@/config/features';
import type { TaskItem, TaskStatus } from '@/mock/types';

/** 列表页可选列，各模块按需开启，避免为每个模块复制一份列表页 */
export interface TaskListOptions {
  /** train：显示 step 进度 */
  showProgress?: boolean;
  /** deploy：显示 endpoint */
  showEndpoint?: boolean;
  /** train：显示指标入口 */
  showMetrics?: boolean;
  /** 「新建」按钮文案 */
  createText: string;
  /** 「模型 / 来源」的语义标签，各模块不同 */
  sourceLabel: string;
}

const STATUS_FILTERS: Array<{ label: string; value: TaskStatus | 'ALL' }> = [
  { label: '全部', value: 'ALL' },
  { label: '运行中', value: 'RUNNING' },
  { label: '已完成', value: 'DONE' },
  { label: '失败', value: 'FAILED' },
  { label: '已暂停', value: 'PAUSED' },
];

const MODULE_ICON: Partial<Record<ModuleKey, ReactNode>> = {
  train: <Zap size={16} />,
  eval: <Network size={16} />,
  export: <Package size={16} />,
  deploy: <Rocket size={16} />,
  workflow: <Workflow size={16} />,
};

/** 次要信息之间的分隔点 */
function Dot() {
  return <span className="text-border">·</span>;
}

/**
 * 行内的小图标按钮。平时藏起来（靠 index.css 的 .task-row .row-actions），
 * hover 整行才浮现，所以不需要每个按钮各自挂 group-hover。
 */
function RowAction({
  icon,
  label,
  onClick,
  danger,
}: {
  icon: ReactNode;
  label: string;
  onClick?: () => void;
  danger?: boolean;
}) {
  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <Button
          variant="ghost"
          size="icon-sm"
          onClick={onClick}
          aria-label={label}
          className={danger ? 'text-state-failed hover:text-state-failed' : 'text-muted-foreground'}
        >
          {icon}
        </Button>
      </TooltipTrigger>
      <TooltipContent>{label}</TooltipContent>
    </Tooltip>
  );
}

export function TaskListPage({
  module,
  tasks,
  options,
}: {
  module: ModuleMeta;
  tasks: TaskItem[];
  options: TaskListOptions;
}) {
  const navigate = useNavigate();
  const [keyword, setKeyword] = useState('');
  const [status, setStatus] = useState<TaskStatus | 'ALL'>('ALL');

  const filtered = useMemo(
    () =>
      tasks.filter(
        (t) =>
          (status === 'ALL' || t.status === status) &&
          (keyword === '' ||
            t.label.includes(keyword) ||
            t.id.includes(keyword) ||
            t.model.includes(keyword)),
      ),
    [tasks, keyword, status],
  );

  /** 只有连了 twinkle-server 的训练才能暂停/继续，本地训练杀了状态就没了 */
  const canPause = (t: TaskItem) => features.pauseResume && t.type === 'train' && Boolean(t.serverUrl);

  const stop = (e: React.MouseEvent) => e.stopPropagation();

  return (
    <>
      <PageHeader
        module={module}
        title={module.label}
        desc={module.desc}
        extra={
          <>
            <Tooltip>
              <TooltipTrigger asChild>
                <Button
                  variant="ghost"
                  size="icon"
                  aria-label="刷新"
                  className="text-muted-foreground"
                  onClick={() => toast.success('已刷新（示意）')}
                >
                  <RotateCw />
                </Button>
              </TooltipTrigger>
              <TooltipContent>刷新</TooltipContent>
            </Tooltip>
            <Button onClick={() => navigate(`${module.path}/new`)}>
              <Plus />
              {options.createText}
            </Button>
          </>
        }
      />

      <Column className="pb-10">
        {/* 整宽搜索，是这一页唯一的输入框。图标绝对定位在框内，输入区留出左内边距 */}
        <div className="relative">
          <Search
            size={15}
            className="text-muted-foreground pointer-events-none absolute top-1/2 left-3 -translate-y-1/2"
          />
          <Input
            placeholder={`搜索名称、id、${options.sourceLabel}`}
            className="h-10 ps-9"
            value={keyword}
            onChange={(e) => setKeyword(e.target.value)}
          />
        </div>

        {/* 状态筛选：小字文本按钮，选中才有一点底色 */}
        <div className="mt-4 mb-1.5 flex flex-wrap gap-0.5">
          {STATUS_FILTERS.map((f) => {
            const active = status === f.value;
            const count =
              f.value === 'ALL' ? tasks.length : tasks.filter((t) => t.status === f.value).length;
            return (
              <button
                key={f.value}
                type="button"
                onClick={() => setStatus(f.value)}
                className={
                  active
                    ? 'bg-secondary text-foreground inline-flex h-7 cursor-pointer items-center gap-1.5 px-2.5 text-[13px] font-medium'
                    : 'text-muted-foreground hover:text-foreground hover:bg-secondary/60 inline-flex h-7 cursor-pointer items-center gap-1.5 px-2.5 text-[13px]'
                }
              >
                {f.label}
                <span className="text-muted-foreground tabular text-xs">{count}</span>
              </button>
            );
          })}
        </div>

        {/* 列表：无边框、无卡片，靠留白分行 */}
        {filtered.length === 0 ? (
          <div className="text-muted-foreground border-border mt-2 border border-dashed py-18 text-center text-sm">
            {tasks.length === 0 ? '还没有任务' : '没有匹配的任务'}
          </div>
        ) : (
          <div className="flex flex-col">
            {filtered.map((t) => (
              <div
                key={t.id}
                className="task-row hover:bg-secondary/50 -mx-3 flex cursor-pointer items-start gap-3.5 px-3 py-3.5 transition-colors"
                onClick={() => navigate(taskDetailPath(t))}
              >
                {/* 裸图标，不套底色方块 */}
                <span className="text-muted-foreground mt-0.5 flex-none">
                  {MODULE_ICON[module.key] ?? <Workflow size={16} />}
                </span>

                <div className="min-w-0 flex-1">
                  {/* 标题行：名字 + 行内灰时间 + 状态 */}
                  <div className="flex flex-wrap items-center gap-2.5">
                    <span className="text-foreground max-w-full truncate text-[15px] font-medium">
                      {t.label}
                    </span>
                    <span className="text-muted-foreground tabular text-[13px]">{t.createdAt}</span>
                    <StatusTag status={t.status} />
                  </div>

                  {/* 描述行：一行灰字串起所有元信息，不再堆胶囊 */}
                  <div className="text-muted-foreground mt-1 flex flex-wrap items-center gap-[7px] text-[13.5px] leading-5">
                    <span>{t.model}</span>
                    <Dot />
                    <span>{t.runner}</span>
                    {t.serverUrl && (
                      <>
                        <Dot />
                        <Tooltip>
                          <TooltipTrigger asChild>
                            <span className="text-foreground inline-flex cursor-default items-center gap-1">
                              <CloudCog size={13} /> server
                            </span>
                          </TooltipTrigger>
                          <TooltipContent>
                            训练循环在本地，算子下沉到 {t.serverUrl}
                          </TooltipContent>
                        </Tooltip>
                      </>
                    )}
                    {options.showProgress && t.totalSteps ? (
                      <>
                        <Dot />
                        <span className="tabular">
                          {t.step ?? 0} / {t.totalSteps} step ·{' '}
                          {Math.round(((t.step ?? 0) / t.totalSteps) * 100)}%
                        </span>
                      </>
                    ) : null}
                    {options.showEndpoint ? (
                      <>
                        <Dot />
                        {t.endpoint ? (
                          <CopyText text={t.endpoint} mono className="text-[12.5px]" />
                        ) : (
                          <Tooltip>
                            <TooltipTrigger asChild>
                              <span className="cursor-default">端口分配中…</span>
                            </TooltipTrigger>
                            <TooltipContent>
                              进程启动后自选端口并写回 runtime.json，此处等它出现
                            </TooltipContent>
                          </Tooltip>
                        )}
                      </>
                    ) : null}
                    <Dot />
                    <span className="font-mono text-[12.5px]">{t.id}</span>
                  </div>
                </div>

                <div className="row-actions flex flex-none items-center gap-0.5" onClick={stop}>
                  {options.showMetrics && (
                    <RowAction
                      icon={<LineChart />}
                      label="指标曲线"
                      onClick={() => navigate(`${module.path}/${t.id}/metrics`)}
                    />
                  )}
                  <RowAction
                    icon={<FileText />}
                    label="日志"
                    onClick={() => navigate(taskDetailPath(t))}
                  />
                  {t.status === 'RUNNING' && canPause(t) && (
                    <RowAction
                      icon={<Pause />}
                      label="暂停后服务端仍保留训练状态，显存不释放"
                      onClick={() => toast.info('暂停（示意，未接后端）')}
                    />
                  )}
                  {t.status === 'PAUSED' && (
                    <RowAction
                      icon={<Play />}
                      label="用同一个 adapter_name 起新客户端，无损续跑"
                      onClick={() => toast.info('继续（示意，未接后端）')}
                    />
                  )}
                  {t.status === 'RUNNING' && (
                    /* 停止是不可逆的，换成需要二次确认的弹窗；
                       antd 的 Popconfirm 在 shadcn 里对应 AlertDialog */
                    <AlertDialog>
                      <AlertDialogTrigger asChild>
                        <Button
                          variant="ghost"
                          size="icon-sm"
                          aria-label="停止"
                          className="text-state-failed hover:text-state-failed"
                        >
                          <Square />
                        </Button>
                      </AlertDialogTrigger>
                      <AlertDialogContent>
                        <AlertDialogHeader>
                          <AlertDialogTitle>停止「{t.label}」？</AlertDialogTitle>
                          <AlertDialogDescription>
                            先发 SIGTERM 让它有机会保存 checkpoint，超时未退再 SIGKILL。
                          </AlertDialogDescription>
                        </AlertDialogHeader>
                        <AlertDialogFooter>
                          <AlertDialogCancel>取消</AlertDialogCancel>
                          <AlertDialogAction onClick={() => toast.info('停止（示意，未接后端）')}>
                            停止
                          </AlertDialogAction>
                        </AlertDialogFooter>
                      </AlertDialogContent>
                    </AlertDialog>
                  )}
                  <DropdownMenu>
                    <DropdownMenuTrigger asChild>
                      <Button
                        variant="ghost"
                        size="icon-sm"
                        aria-label="更多"
                        className="text-muted-foreground"
                      >
                        <MoreHorizontal />
                      </Button>
                    </DropdownMenuTrigger>
                    <DropdownMenuContent align="end">
                      <DropdownMenuItem onClick={() => toast.info('示意，未接后端')}>
                        重命名
                      </DropdownMenuItem>
                      <DropdownMenuItem onClick={() => toast.info('示意，未接后端')}>
                        以此为模板新建
                      </DropdownMenuItem>
                      <DropdownMenuSeparator />
                      <DropdownMenuItem
                        variant="destructive"
                        onClick={() => toast.info('示意，未接后端')}
                      >
                        <Trash2 />
                        删除
                      </DropdownMenuItem>
                    </DropdownMenuContent>
                  </DropdownMenu>
                </div>
              </div>
            ))}
          </div>
        )}
      </Column>
    </>
  );
}
