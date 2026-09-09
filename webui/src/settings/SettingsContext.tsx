import { createContext, useCallback, useContext, useEffect, useMemo, useState } from 'react';

/**
 * 全局偏好设置。
 *
 * 这是项目里第一个全局状态——在此之前所有状态都是页面自己的 useState。
 * 引入它的理由很具体：tips 条的开关要在设置页里改、在 footbar 上生效，
 * 两个组件隔着整个组件树，靠 props 传不过去。
 *
 * 持久化用 localStorage 而不是后端：这些是「这台机器上这个人」的偏好，
 * 换了浏览器重新选一次没什么损失，不值得为它加一张表。
 */
export interface Settings {
  /** 底部显示轮换 tips */
  tipsEnabled: boolean;
  /** tips 轮换间隔（秒） */
  tipsInterval: number;
  /** tips 只显示自己看得懂的：关掉后不出现涉及具体算法细节的条目 */
  tipsBeginnerOnly: boolean;
  /** 节点和整图上的 AI 入口 */
  aiEnabled: boolean;
  /**
   * AI 改动是否需要人工确认。
   * 默认开，且强烈建议别关——图形是唯一事实来源，
   * AI 静默改图会让「谁把 lr 调成 1e-3 的」变成查不出来的事。
   */
  aiConfirmBeforeApply: boolean;
  /** 节点跑完自动弹出输出面板 */
  autoOpenOutput: boolean;
}

const DEFAULTS: Settings = {
  tipsEnabled: true,
  tipsInterval: 12,
  tipsBeginnerOnly: false,
  aiEnabled: true,
  aiConfirmBeforeApply: true,
  autoOpenOutput: false,
};

const STORAGE_KEY = 'swift-webui.settings.v1';

/** 读 localStorage。坏数据、隐私模式下抛错都当没设置过处理 */
function loadSettings(): Settings {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return DEFAULTS;
    const saved = JSON.parse(raw) as Partial<Settings>;
    /* 逐字段合并而不是整体替换：将来加字段时老用户不会缺键 */
    return { ...DEFAULTS, ...saved };
  } catch {
    return DEFAULTS;
  }
}

interface SettingsCtx {
  settings: Settings;
  /** 改一个字段 */
  update: <K extends keyof Settings>(key: K, value: Settings[K]) => void;
  /** 恢复默认 */
  reset: () => void;
  /**
   * 本次会话里临时把 tips 关掉（footbar 上那个叉）。
   * 跟 settings.tipsEnabled 分开：叉掉是「这次别烦我」，
   * 设置里关掉是「以后都别出现」，刷新页面后前者恢复、后者保持。
   */
  tipsDismissed: boolean;
  dismissTips: () => void;
}

const Ctx = createContext<SettingsCtx | null>(null);

export function SettingsProvider({ children }: { children: React.ReactNode }) {
  const [settings, setSettings] = useState<Settings>(loadSettings);
  const [tipsDismissed, setTipsDismissed] = useState(false);

  /* 写回磁盘。隐私模式下 setItem 会抛，静默忽略即可，不该因此白屏 */
  useEffect(() => {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(settings));
    } catch {
      /* 存不下就只在本次会话生效 */
    }
  }, [settings]);

  const update = useCallback(
    <K extends keyof Settings>(key: K, value: Settings[K]) =>
      setSettings((s) => ({ ...s, [key]: value })),
    [],
  );

  const reset = useCallback(() => setSettings(DEFAULTS), []);
  const dismissTips = useCallback(() => setTipsDismissed(true), []);

  const value = useMemo(
    () => ({ settings, update, reset, tipsDismissed, dismissTips }),
    [settings, update, reset, tipsDismissed, dismissTips],
  );

  return <Ctx.Provider value={value}>{children}</Ctx.Provider>;
}

export function useSettings(): SettingsCtx {
  const v = useContext(Ctx);
  if (!v) throw new Error('useSettings 必须在 SettingsProvider 里用');
  return v;
}

export { DEFAULTS as DEFAULT_SETTINGS };
