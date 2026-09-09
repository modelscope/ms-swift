/**
 * 模块注册表：全站唯一的「有哪些功能模块 / 挂在哪个路由」的定义源。
 *
 * 加一个新模块 = 在这里加一项 + 在 router/routes.tsx 挂上页面，
 * 侧边导航、面包屑会自动跟随，不需要改其它文件。
 *
 * 配色学 unsloth：整站单一强调色（brand 淡紫），模块不再各用不同色；
 * accent/accentSoft 保留字段以兼容现有组件，但统一指向 brand。
 */
import { brand } from './theme';
export type ModuleKey = 'chat' | 'train' | 'eval' | 'export' | 'deploy' | 'workflow';

export interface ModuleMeta {
  key: ModuleKey;
  /** 侧边栏与标题上的显示名 */
  label: string;
  /** 路由前缀，如 /train */
  path: string;
  /** 强调色（整站统一走 brand，保留字段兼容组件） */
  accent: string;
  /** 浅色底，用于选中态、标签底色 */
  accentSoft: string;
  /** 一句话说明，显示在页头 */
  desc: string;
}

export const MODULES: Record<ModuleKey, ModuleMeta> = {
  chat: {
    key: 'chat',
    label: '对话',
    path: '/chat',
    accent: brand.primary,
    accentSoft: brand.soft,
    desc: '与本地或远端模型对话，可多模型并排对比',
  },
  train: {
    key: 'train',
    label: '训练',
    path: '/train',
    accent: brand.primary,
    accentSoft: brand.soft,
    desc: '发起训练任务，实时查看指标曲线与日志',
  },
  eval: {
    key: 'eval',
    label: '评测',
    path: '/eval',
    accent: brand.primary,
    accentSoft: brand.soft,
    desc: '对 checkpoint 跑评测，支持多结果对比',
  },
  export: {
    key: 'export',
    label: '导出',
    path: '/export',
    accent: brand.primary,
    accentSoft: brand.soft,
    desc: 'LoRA 合并、量化、推送到 Hub',
  },
  deploy: {
    key: 'deploy',
    label: '部署',
    path: '/deploy',
    accent: brand.primary,
    accentSoft: brand.soft,
    desc: '把模型拉起成常驻推理服务',
  },
  workflow: {
    key: 'workflow',
    label: '编排',
    path: '/workflow',
    accent: brand.primary,
    accentSoft: brand.soft,
    desc: '把训练、评测、合并、部署串成一条流程',
  },
};

/** 侧边导航顺序 */
export const MODULE_ORDER: ModuleKey[] = ['chat', 'train', 'eval', 'export', 'deploy', 'workflow'];

/** 根据当前 pathname 反查所属模块，找不到时回落到 chat */
export function moduleOf(pathname: string): ModuleMeta {
  const hit = MODULE_ORDER.map((k) => MODULES[k]).find(
    (m) => pathname === m.path || pathname.startsWith(m.path + '/'),
  );
  return hit ?? MODULES.chat;
}

/**
 * 任务详情页路径。各模块默认 Tab 不同，统一交给路由里的 Navigate 决定，
 * 这里只给到 /{module}/{id}；编排没有详情页，点进去是编辑器。
 *
 * 列表行、侧栏「最近」都走这个函数，避免各处各拼一份路径拼错。
 */
export function taskDetailPath(task: { type: ModuleKey; id: string }): string {
  if (task.type === 'workflow') return `/workflow/${task.id}/edit`;
  return `${MODULES[task.type].path}/${task.id}`;
}
