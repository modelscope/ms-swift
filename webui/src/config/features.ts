import type { ModuleKey } from '@/theme/modules';

/**
 * 功能开关清单。
 * 设计里「界面风格可变、功能可增减」的落点：不同交付场景（单租户 / 多租户 /
 * 只给推理不给训练）通过改这一份配置裁剪界面，而不是在页面里到处写 if。
 *
 * 后续接后端时，这份配置改为启动时从 /api/features 拉取即可，页面无需改动。
 */
export interface FeatureFlags {
  /** 侧边栏里显示哪些模块，顺序仍由 MODULE_ORDER 决定 */
  modules: ModuleKey[];
  /** 多租户模式：显示租户切换、当前用户 */
  multiTenant: boolean;
  /** 对话页开启多模型并排对比 */
  chatCompare: boolean;
  /** 训练页显示「连 twinkle-server」这一类远端任务的暂停/继续 */
  pauseResume: boolean;
  /** 编排页显示 yaml/代码 双向同步面板 */
  workflowCodeSync: boolean;
}

export const features: FeatureFlags = {
  modules: ['chat', 'train', 'eval', 'export', 'deploy', 'workflow'],
  multiTenant: true,
  chatCompare: true,
  pauseResume: true,
  workflowCodeSync: true,
};
