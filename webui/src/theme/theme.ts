import type { CSSProperties } from 'react';
import type { ThemeConfig } from 'antd';

/**
 * 设计 token。整体走「桌面 App」的路子：
 * 深色 chrome（标题栏 + 侧栏）+ 浅色内容区。
 *
 * 配色取自 SWIFT logo（燕子）：左尾钢蓝 → 中段靛蓝 → 头部深梅紫，
 * 低饱和、偏冷、沉稳。换配色只改这个文件，页面不动。
 */

/** logo 三段取色，渐变与图标都用它 */
export const logo = {
  blue: '#3A6EA5',
  indigo: '#3E4585',
  plum: '#573E79',
};

/** logo 同向的渐变，用于品牌标识、头像、强调块 */
export const logoGradient = `linear-gradient(135deg, ${logo.blue} 0%, ${logo.indigo} 52%, ${logo.plum} 100%)`;

/** 强调色：取 logo 中段靛蓝，比之前的亮紫沉稳得多 */
export const brand = {
  primary: '#45478F',
  primaryHover: '#3B3D7E',
  primaryActive: '#31336B',
  soft: '#EEEFF7',
  softBorder: '#CFD2E8',
  /** 深色 chrome 上用的亮化版本，保证对比度 */
  onDark: '#A8AEE0',
};

/** 浅色内容区的中性色，带一丝冷调以与 chrome 呼应 */
export const neutral = {
  text: '#16171F',
  textSecondary: '#525565',
  textTertiary: '#868A9C',
  border: '#E2E4EC',
  borderLight: '#ECEDF3',
  bgSubtle: '#F6F7FA',
  bgCode: '#F3F4F8',
};

/**
 * 内容列。内容不铺满整个宽度，而是居中收在一条列里——
 * 这是 Codex 那种「应用」观感的来源之一，铺满边到边就会变回后台管理。
 */
export const column = {
  maxWidth: 880,
  gutter: 32,
} as const;

export const columnStyle: CSSProperties = {
  width: '100%',
  maxWidth: column.maxWidth,
  marginInline: 'auto',
  paddingInline: column.gutter,
};

/** 详情页用的宽列：要放曲线和日志，880 太窄 */
export const wideColumnStyle: CSSProperties = {
  width: '100%',
  maxWidth: 1240,
  marginInline: 'auto',
  paddingInline: column.gutter,
};

/** 深色 chrome：标题栏与侧栏。色相偏 logo 的靛/梅紫 */
export const chrome = {
  titlebar: '#15161F',
  sidebar: '#1A1B26',
  border: 'rgba(255,255,255,0.07)',
  text: '#E6E7F0',
  textDim: '#969AAE',
  textFaint: '#6A6E82',
  hover: 'rgba(255,255,255,0.06)',
  activeBg: 'rgba(90,100,180,0.22)',
  activeText: '#C3C8F0',
};

/** 窗口外的「桌面」底色：深、低饱和，让白窗浮起来 */
export const ambientBackground = `
  radial-gradient(1200px 620px at 10% 0%, #26304A 0%, rgba(38,48,74,0) 60%),
  radial-gradient(1000px 560px at 100% 100%, #2E2544 0%, rgba(46,37,68,0) 58%),
  radial-gradient(800px 500px at 100% 0%, #1F2A40 0%, rgba(31,42,64,0) 55%),
  #14151D
`;

export const baseTheme: ThemeConfig = {
  token: {
    colorPrimary: brand.primary,
    colorInfo: brand.primary,
    colorLink: brand.primary,
    colorLinkHover: brand.primaryHover,
    colorBgLayout: 'transparent',
    colorBgContainer: '#ffffff',
    colorBorderSecondary: neutral.borderLight,
    colorBorder: neutral.border,
    colorText: neutral.text,
    colorTextSecondary: neutral.textSecondary,
    borderRadius: 10,
    borderRadiusLG: 14,
    fontSize: 14,
    fontFamily:
      'Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC", "Hiragino Sans GB", "Microsoft YaHei", sans-serif',
    controlHeight: 36,
  },
  components: {
    Layout: {
      bodyBg: 'transparent',
      headerBg: 'transparent',
      siderBg: 'transparent',
      headerHeight: 52,
    },
    Menu: {
      itemBg: 'transparent',
      subMenuItemBg: 'transparent',
      itemHeight: 38,
      itemBorderRadius: 9,
      itemSelectedBg: brand.soft,
      itemSelectedColor: brand.primaryActive,
      itemHoverBg: neutral.bgSubtle,
    },
    Button: {
      borderRadius: 9,
      controlHeight: 36,
      primaryShadow: 'none',
      defaultShadow: 'none',
    },
    Table: {
      headerBg: 'transparent',
      headerColor: neutral.textTertiary,
      rowHoverBg: neutral.bgSubtle,
      borderColor: neutral.borderLight,
    },
    Card: {
      headerFontSize: 15,
      borderRadiusLG: 14,
    },
    Input: {
      borderRadius: 9,
    },
    Segmented: {
      borderRadius: 9,
      itemSelectedColor: brand.primaryActive,
    },
    Tabs: {
      titleFontSize: 14,
      inkBarColor: brand.primary,
      itemSelectedColor: brand.primaryActive,
    },
  },
};
