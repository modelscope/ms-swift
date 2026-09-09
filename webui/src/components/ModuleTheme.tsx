import { ConfigProvider } from 'antd';
import type { ReactNode } from 'react';
import type { ModuleMeta } from '@/theme/modules';

/**
 * 按模块注入强调色。
 * 页面内部所有 antd 组件的 primary 色都跟着当前模块走，
 * 页面代码里不需要出现任何色值——这是「每个功能不同色调」的唯一实现点。
 */
export function ModuleTheme({ module, children }: { module: ModuleMeta; children: ReactNode }) {
  return (
    <ConfigProvider
      theme={{
        token: {
          colorPrimary: module.accent,
          colorLink: module.accent,
        },
      }}
    >
      {children}
    </ConfigProvider>
  );
}
