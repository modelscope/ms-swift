import { createBrowserRouter, Navigate } from 'react-router-dom';
import { AppLayout } from '@/layout/AppLayout';
import { ChatPage } from '@/pages/chat/ChatPage';
import { TrainListPage } from '@/pages/train/TrainListPage';
import { TrainNewPage } from '@/pages/train/TrainNewPage';
import { TrainDetailPage } from '@/pages/train/TrainDetailPage';
import { EvalListPage } from '@/pages/eval/EvalListPage';
import { EvalNewPage } from '@/pages/eval/EvalNewPage';
import { EvalDetailPage } from '@/pages/eval/EvalDetailPage';
import { ExportListPage } from '@/pages/export/ExportListPage';
import { ExportNewPage } from '@/pages/export/ExportNewPage';
import { ExportDetailPage } from '@/pages/export/ExportDetailPage';
import { DeployListPage } from '@/pages/deploy/DeployListPage';
import { DeployNewPage } from '@/pages/deploy/DeployNewPage';
import { DeployDetailPage } from '@/pages/deploy/DeployDetailPage';
import { WorkflowListPage } from '@/pages/workflow/WorkflowListPage';
import { WorkflowEditorPage } from '@/pages/workflow/WorkflowEditorPage';
import { SettingsPage } from '@/pages/settings/SettingsPage';

/**
 * 路由表。列表 / 新建 / 详情三段式；详情用 :tab 段承载 Tab，
 * 这样刷新或分享链接都能停在同一个 Tab。加模块时在此挂节点即可。
 */
export const router = createBrowserRouter([
  {
    path: '/',
    element: <AppLayout />,
    children: [
      { index: true, element: <Navigate to="/chat" replace /> },

      { path: 'chat', element: <ChatPage /> },

      { path: 'train', element: <TrainListPage /> },
      { path: 'train/new', element: <TrainNewPage /> },
      { path: 'train/:id/metrics', element: <TrainDetailPage tab="metrics" /> },
      { path: 'train/:id/log', element: <TrainDetailPage tab="log" /> },
      { path: 'train/:id', element: <Navigate to="metrics" replace /> },

      { path: 'eval', element: <EvalListPage /> },
      { path: 'eval/new', element: <EvalNewPage /> },
      { path: 'eval/:id/result', element: <EvalDetailPage tab="result" /> },
      { path: 'eval/:id/log', element: <EvalDetailPage tab="log" /> },
      { path: 'eval/:id', element: <Navigate to="result" replace /> },

      { path: 'export', element: <ExportListPage /> },
      { path: 'export/new', element: <ExportNewPage /> },
      { path: 'export/:id/artifact', element: <ExportDetailPage tab="artifact" /> },
      { path: 'export/:id/log', element: <ExportDetailPage tab="log" /> },
      { path: 'export/:id', element: <Navigate to="artifact" replace /> },

      { path: 'deploy', element: <DeployListPage /> },
      { path: 'deploy/new', element: <DeployNewPage /> },
      { path: 'deploy/:id/overview', element: <DeployDetailPage tab="overview" /> },
      { path: 'deploy/:id/log', element: <DeployDetailPage tab="log" /> },
      { path: 'deploy/:id', element: <Navigate to="overview" replace /> },

      { path: 'workflow', element: <WorkflowListPage /> },
      { path: 'workflow/new', element: <WorkflowEditorPage /> },
      { path: 'workflow/:id/edit', element: <WorkflowEditorPage /> },

      /* 不属于任何模块，不进侧边导航，从品牌下拉菜单进 */
      { path: 'settings', element: <SettingsPage /> },

      { path: '*', element: <Navigate to="/chat" replace /> },
    ],
  },
]);
