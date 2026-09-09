import React from 'react';
import ReactDOM from 'react-dom/client';
import { RouterProvider } from 'react-router-dom';
import { App as AntApp, ConfigProvider } from 'antd';
import zhCN from 'antd/locale/zh_CN';
import { router } from '@/router/routes';
import { SettingsProvider } from '@/settings/SettingsContext';
import { baseTheme } from '@/theme/theme';
import './index.css';

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <ConfigProvider locale={zhCN} theme={baseTheme}>
      <AntApp>
        {/* 偏好设置包在路由外：tips 条在外壳上，开关在设置页里，两边都要读到 */}
        <SettingsProvider>
          <RouterProvider router={router} />
        </SettingsProvider>
      </AntApp>
    </ConfigProvider>
  </React.StrictMode>,
);
