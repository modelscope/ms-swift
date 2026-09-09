import { useLocation, useNavigate } from 'react-router-dom';
import { ArrowLeft, ArrowRight, BookOpen, PanelLeftClose, PanelLeftOpen } from 'lucide-react';
import { ChromeIconButton } from './ChromeIconButton';
import { TITLEBAR_H } from './metrics';
import { moduleOf } from '@/theme/modules';

/**
 * 窗口标题栏：侧栏开合 + 前进后退在左，当前位置居中，全局链接贴右。
 *
 * 原来左边有一组 macOS 交通灯，配合「浮在深色桌面上的圆角窗口」才成立。
 * 现在界面铺满视口，那三个点点了也不会关窗口，纯装饰且跟工具型界面的克制冲突，去掉了。
 *
 * 刻意不做面包屑，也不放居中的大搜索条——搜索入口在侧栏。
 */

/**
 * lucide 1.x 把品牌 logo 全部移出了图标库（品牌资产有各自的使用规范，
 * 不适合当通用图标维护），所以 GitHub 这个 mark 只能自己内联。
 * 用通用图标代替会丢掉识别度，这是少数值得手写 SVG 的地方。
 */
function GithubMark({ size = 14 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 16 16" fill="currentColor" aria-hidden="true">
      <path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-2.91-.88-2.91-2.9 0-.86.31-1.57.82-2.12-.08-.2-.36-1.01.08-2.1 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.09.16 1.9.08 2.1.51.55.82 1.25.82 2.12 0 2.03-1.14 2.7-2.92 2.9.3.26.56.76.56 1.54 0 1.11-.01 2.01-.01 2.29 0 .21.15.46.55.38A7.995 7.995 0 0 0 16 8c0-4.42-3.58-8-8-8Z" />
    </svg>
  );
}

export function Titlebar({
  collapsed,
  onToggleSidebar,
}: {
  collapsed: boolean;
  onToggleSidebar: () => void;
}) {
  const navigate = useNavigate();
  const { pathname } = useLocation();
  const current = moduleOf(pathname);
  const rest = pathname.slice(current.path.length).split('/').filter(Boolean);

  const tail: Record<string, string> = {
    new: '新建',
    log: '日志',
    metrics: '指标',
    result: '结果',
    edit: '编辑',
  };
  const sub = rest.map((s) => tail[s] ?? s).join(' · ');

  return (
    <div
      className="bg-titlebar border-sidebar-border flex shrink-0 items-center gap-0.5 border-b px-2.5 select-none"
      style={{ height: TITLEBAR_H }}
    >
      <ChromeIconButton
        icon={collapsed ? <PanelLeftOpen size={15} /> : <PanelLeftClose size={15} />}
        label={collapsed ? '展开侧栏' : '收起侧栏'}
        onClick={onToggleSidebar}
      />
      <ChromeIconButton icon={<ArrowLeft size={15} />} label="后退" onClick={() => navigate(-1)} />
      <ChromeIconButton icon={<ArrowRight size={15} />} label="前进" onClick={() => navigate(1)} />

      <div className="flex min-w-0 flex-1 items-center justify-center gap-2">
        <span className="text-sidebar-foreground/60 text-xs whitespace-nowrap">{current.label}</span>
        {sub && (
          <span className="text-sidebar-foreground/40 text-xs whitespace-nowrap">{sub}</span>
        )}
      </div>

      <ChromeIconButton
        icon={<BookOpen size={14} />}
        label="文档"
        href="https://swift.readthedocs.io"
      />
      <ChromeIconButton
        icon={<GithubMark />}
        label="GitHub"
        href="https://github.com/modelscope/ms-swift"
      />
    </div>
  );
}
