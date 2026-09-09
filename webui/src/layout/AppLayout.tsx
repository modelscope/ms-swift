import { useState } from 'react';
import { Outlet, useLocation } from 'react-router-dom';
import { Sidebar } from './Sidebar';
import { Titlebar } from './Titlebar';
import { TipsBar } from './TipsBar';
import { CONTENT_H_VAR, contentHeightValue } from './metrics';
import { ModuleTheme } from '@/components/ModuleTheme';
import { useSettings } from '@/settings/SettingsContext';
import { moduleOf } from '@/theme/modules';
import { ambientBackground } from '@/theme/theme';

/**
 * 应用外壳：做成一个桌面 App 窗口——深色「桌面」背景上浮着一扇圆角窗口，
 * 窗口内是标题栏（深）+ 侧栏（深）+ 内容区（浅）+ 底部 tips 条（深）。
 * 这种 chrome 深 / 内容浅的对比，是 App 感的主要来源。
 */
export function AppLayout() {
  const { pathname } = useLocation();
  const current = moduleOf(pathname);
  const [collapsed, setCollapsed] = useState(false);
  const { settings, tipsDismissed } = useSettings();

  const tipsVisible = settings.tipsEnabled && !tipsDismissed;

  return (
    <div
      style={{
        height: '100vh',
        padding: 14,
        background: ambientBackground,
        display: 'flex',
      }}
    >
      <div
        style={{
          flex: 1,
          minWidth: 0,
          display: 'flex',
          flexDirection: 'column',
          borderRadius: 14,
          overflow: 'hidden',
          background: '#fff',
          boxShadow: '0 24px 70px rgba(0,0,0,0.45), 0 0 0 1px rgba(255,255,255,0.06)',
          /*
           * 整屏页面（对话、编排画布）用这个变量拿可用高度。
           * tips 条能被叉掉，高度是运行时变的，只有这里知道到底减不减那 26px。
           */
          [CONTENT_H_VAR]: contentHeightValue(tipsVisible),
        } as React.CSSProperties}
      >
        <Titlebar collapsed={collapsed} onToggleSidebar={() => setCollapsed((v) => !v)} />
        <div style={{ flex: 1, display: 'flex', minHeight: 0 }}>
          <Sidebar collapsed={collapsed} />
          <div style={{ flex: 1, minWidth: 0, overflow: 'auto', background: '#fff' }}>
            <ModuleTheme module={current}>
              <Outlet />
            </ModuleTheme>
          </div>
        </div>
        {tipsVisible && <TipsBar />}
      </div>
    </div>
  );
}
