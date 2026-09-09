import { useState } from 'react';
import { Outlet } from 'react-router-dom';
import { Sidebar } from './Sidebar';
import { Titlebar } from './Titlebar';
import { TipsBar } from './TipsBar';
import { CONTENT_H_VAR, contentHeightValue } from './metrics';
import { useSettings } from '@/settings/SettingsContext';

/**
 * 应用外壳：标题栏（深）+ 侧栏（深）+ 内容区（浅）+ 底部 tips 条（深）。
 * 深外壳夹浅工作区是 IDE 的常见做法，也是这套界面「工具感」的主要来源。
 *
 * 原来外面还套了一层深色桌面、窗口带圆角和大阴影，现在去掉了，
 * 理由写在 metrics.ts 的 WINDOW_PAD 上。
 */
export function AppLayout() {
  const [collapsed, setCollapsed] = useState(false);
  const { settings, tipsDismissed } = useSettings();

  const tipsVisible = settings.tipsEnabled && !tipsDismissed;

  return (
    <div
      className="flex h-screen flex-col overflow-hidden bg-background"
      style={
        {
          /*
           * 整屏页面（对话、编排画布）用这个变量拿可用高度。
           * tips 条能被叉掉，高度是运行时变的，只有这里知道到底减不减那 26px。
           */
          [CONTENT_H_VAR]: contentHeightValue(tipsVisible),
        } as React.CSSProperties
      }
    >
      <Titlebar collapsed={collapsed} onToggleSidebar={() => setCollapsed((v) => !v)} />
      <div className="flex min-h-0 flex-1">
        <Sidebar collapsed={collapsed} />
        <main className="min-w-0 flex-1 overflow-auto bg-background">
          <Outlet />
        </main>
      </div>
      {tipsVisible && <TipsBar />}
    </div>
  );
}
