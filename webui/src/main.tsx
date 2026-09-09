import React from 'react';
import ReactDOM from 'react-dom/client';
import { RouterProvider } from 'react-router-dom';
import { TooltipProvider } from '@/components/ui/tooltip';
import { Toaster } from '@/components/ui/sonner';
import { router } from '@/router/routes';
import { SettingsProvider } from '@/settings/SettingsContext';
import './index.css';

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    {/*
      delayDuration 给 400ms。shadcn 默认是 0，鼠标扫过工具栏就会连着弹几个气泡，
      而这个界面的图标按钮排得很密。放在最外层是因为提示要跨路由用，
      每个页面各包一个 Provider 的话延迟就各是各的。
    */}
    <TooltipProvider delayDuration={400}>
      {/* 偏好设置包在路由外：tips 条在外壳上，开关在设置页里，两边都要读到 */}
      <SettingsProvider>
        <RouterProvider router={router} />
      </SettingsProvider>
    </TooltipProvider>
    <Toaster position="bottom-center" />
  </React.StrictMode>,
);
