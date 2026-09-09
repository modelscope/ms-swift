import { useMemo } from 'react';
import type { ReactNode } from 'react';
import { Link, useLocation, useNavigate } from 'react-router-dom';
import {
  Bell,
  ChevronDown,
  CircleHelp,
  MessageSquare,
  Network,
  Package,
  Plus,
  Rocket,
  Search,
  Workflow,
  Zap,
} from 'lucide-react';
import { cn } from 'cn';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu';
import { Tooltip, TooltipContent, TooltipTrigger } from '@/components/ui/tooltip';
import type { ModuleKey } from '@/theme/modules';
import { MODULES, MODULE_ORDER, moduleOf, taskDetailPath } from '@/theme/modules';
import { features } from '@/config/features';
import { tasksByType } from '@/mock/data';
import { ChromeIconButton } from './ChromeIconButton';

const ICONS: Record<ModuleKey, ReactNode> = {
  chat: <MessageSquare size={14} />,
  train: <Zap size={14} />,
  eval: <Network size={14} />,
  export: <Package size={14} />,
  deploy: <Rocket size={14} />,
  workflow: <Workflow size={14} />,
};

/**
 * 深色侧栏里弹出的菜单，配色改成跟侧栏同一套。
 *
 * shadcn 的 DropdownMenuContent 默认是 bg-popover——那是内容区那套浅色语义，
 * 从一块近黑的侧栏里弹出一张白卡片会很割裂。这里换成 sidebar-* 那组，
 * 菜单看起来就像从侧栏本身撑开的，跟 VS Code 一致。
 */
const MENU_CLS = 'bg-sidebar border-sidebar-border text-sidebar-foreground min-w-[168px]';
const MENU_ITEM_CLS =
  'text-sidebar-foreground/80 focus:bg-sidebar-accent focus:text-sidebar-foreground text-[13px]';

/** 分组小标签，用来打断长列表，而不是一长条等距菜单 */
function SectionLabel({ children }: { children: ReactNode }) {
  return (
    <div className="text-sidebar-foreground/40 mt-[18px] mb-1.5 px-[11px] text-xs select-none">
      {children}
    </div>
  );
}

/**
 * 左侧导航。信息设计参考 Codex：小字号、紧凑行高、细图标，
 * 选中态只是一块低对比底色——导航里不出现品牌色，让内容区去承担强调。
 * 操作按钮走「非线性」：贴边的无框幽灵按钮，而非并排的按钮组。
 */
export function Sidebar({ collapsed }: { collapsed: boolean }) {
  const { pathname } = useLocation();
  const navigate = useNavigate();
  const current = moduleOf(pathname);

  const items = useMemo(
    () => MODULE_ORDER.filter((k) => features.modules.includes(k)).map((k) => MODULES[k]),
    [],
  );

  /** 「最近」分组：跨模块取最新几条，给侧栏一点活气 */
  const recent = useMemo(
    () =>
      Object.values(tasksByType)
        .flat()
        .sort((a, b) => b.createdAt.localeCompare(a.createdAt))
        .slice(0, 5),
    [],
  );

  /* 导航行的公共形状。选中/未选中只差一层底色和字重 */
  const row = (active: boolean, extra?: string) =>
    cn(
      'flex h-8 items-center gap-2.5 text-[13px] no-underline transition-colors',
      collapsed ? 'justify-center px-0' : 'justify-start px-[11px]',
      active
        ? 'bg-sidebar-accent text-sidebar-foreground font-medium'
        : 'text-sidebar-foreground/60 hover:bg-sidebar-accent/60 hover:text-sidebar-foreground',
      extra,
    );

  return (
    <div
      className="bg-sidebar border-sidebar-border flex h-full shrink-0 flex-col overflow-hidden border-r transition-[width] duration-200"
      style={{ width: collapsed ? 60 : 240 }}
    >
      {/* 品牌区：纯文字 + 小箭头，右侧贴边的无框图标 */}
      <div
        className={cn(
          'flex h-[46px] shrink-0 items-center gap-1.5',
          collapsed ? 'px-4' : 'pr-3 pl-3.5',
        )}
      >
        {collapsed ? (
          <span className="text-sidebar-foreground text-sm font-bold">S</span>
        ) : (
          <>
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <button
                  type="button"
                  className="text-sidebar-foreground hover:text-sidebar-foreground/80 inline-flex cursor-pointer items-center gap-1.5 text-[15px] font-semibold tracking-wide outline-none"
                >
                  SWIFT
                  <ChevronDown size={10} className="text-sidebar-foreground/40" />
                </button>
              </DropdownMenuTrigger>
              <DropdownMenuContent align="start" className={MENU_CLS}>
                <DropdownMenuItem className={MENU_ITEM_CLS}>关于 SWIFT WebUI</DropdownMenuItem>
                <DropdownMenuItem className={MENU_ITEM_CLS} onClick={() => navigate('/settings')}>
                  偏好设置
                </DropdownMenuItem>
                <DropdownMenuSeparator className="bg-sidebar-border" />
                <DropdownMenuItem className={MENU_ITEM_CLS} asChild>
                  <a href="https://swift.readthedocs.io" target="_blank" rel="noreferrer">
                    文档
                  </a>
                </DropdownMenuItem>
              </DropdownMenuContent>
            </DropdownMenu>

            <div className="ms-auto flex gap-0.5">
              <ChromeIconButton icon={<Search size={13} />} label="搜索  ⌘K" />
              <ChromeIconButton icon={<Bell size={13} />} label="通知" />
            </div>
          </>
        )}
      </div>

      <div className="flex-1 overflow-y-auto px-2 py-0.5">
        {/* 模块导航 */}
        {items.map((m) => {
          const active = current.key === m.key;
          const link = (
            <Link
              to={m.path}
              className={row(
                active,
                /* 对话行右侧留出脱离行外的 + 按钮位置 */
                !collapsed && m.key === 'chat' ? 'pe-1' : undefined,
              )}
            >
              <span className="inline-flex text-current opacity-90">{ICONS[m.key]}</span>
              {!collapsed && m.label}
              {!collapsed && m.key === 'chat' && (
                <span className="ms-auto">
                  <ChromeIconButton
                    icon={<Plus size={13} />}
                    label="新建对话"
                    size={24}
                    onClick={() => navigate('/chat')}
                  />
                </span>
              )}
            </Link>
          );
          return collapsed ? (
            <Tooltip key={m.key}>
              <TooltipTrigger asChild>{link}</TooltipTrigger>
              <TooltipContent side="right">{m.label}</TooltipContent>
            </Tooltip>
          ) : (
            <div key={m.key}>{link}</div>
          );
        })}

        {/* 最近：跨模块的近期任务 */}
        {!collapsed && (
          <>
            <SectionLabel>最近</SectionLabel>
            {recent.map((t) => (
              <Link key={t.id} to={taskDetailPath(t)} className={row(pathname.includes(t.id), 'h-[30px]')}>
                <span className="truncate">{t.label}</span>
              </Link>
            ))}
          </>
        )}
      </div>

      {/* 底部：空间切换 + 用户，右侧贴边帮助 */}
      <div className="border-sidebar-border shrink-0 border-t p-2">
        {features.multiTenant && !collapsed && (
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <button type="button" className={cn(row(false), 'w-full cursor-pointer outline-none')}>
                <span className="text-sidebar-foreground/40">空间</span>
                <span className="text-sidebar-foreground">default</span>
                <ChevronDown size={10} className="text-sidebar-foreground/40 ms-auto" />
              </button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="start" side="top" className={MENU_CLS}>
              <DropdownMenuItem className={MENU_ITEM_CLS}>default（默认空间）</DropdownMenuItem>
              <DropdownMenuItem className={MENU_ITEM_CLS}>team-a</DropdownMenuItem>
              <DropdownMenuItem className={MENU_ITEM_CLS}>team-b</DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>
        )}

        <div
          className={cn(
            'flex h-[34px] items-center gap-2',
            collapsed ? 'justify-center px-0' : 'justify-start px-[11px]',
          )}
        >
          <div className="bg-sidebar-accent text-sidebar-foreground flex size-[22px] shrink-0 items-center justify-center rounded-full text-[11px] font-semibold">
            U
          </div>
          {!collapsed && (
            <>
              <span className="text-sidebar-foreground/60 truncate text-[13px]">user@swift</span>
              <span className="ms-auto">
                <ChromeIconButton icon={<CircleHelp size={13} />} label="帮助" side="top" />
              </span>
            </>
          )}
        </div>
      </div>
    </div>
  );
}
