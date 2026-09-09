/**
 * 外壳尺寸。
 *
 * 之前几个整屏页面各自硬写 `calc(100vh - 28px - 40px)`，那两个数字分别是
 * 窗口内边距（14 上 + 14 下）和标题栏高度——一旦底部多一条 footbar，
 * 每个页面都得跟着改，改漏一个就多出 26px 的滚动条。
 *
 * 所以把高度收成一个 CSS 变量：AppLayout 算一次写在窗口节点上，
 * 页面只引用 contentHeight()，footbar 开关时高度自动跟着变。
 */

/** 标题栏高度，见 Titlebar.tsx */
export const TITLEBAR_H = 40;
/** 窗口离屏幕边缘的留白，见 AppLayout 外层 padding */
export const WINDOW_PAD = 14;
/** 底部 tips 条高度 */
export const TIPSBAR_H = 26;

/** 内容区可用高度的 CSS 变量名，由 AppLayout 写入 */
export const CONTENT_H_VAR = '--app-content-h';

/**
 * 整屏页面的高度。extra 用来再减去页面自己的页头之类。
 *
 * 注意用的是 CSS 变量而不是直接算 100vh：tips 条能被叉掉，
 * 高度是运行时变的，只有 AppLayout 知道当前到底减不减那 26px。
 */
export function contentHeight(extra = 0): string {
  const base = `var(${CONTENT_H_VAR})`;
  return extra ? `calc(${base} - ${extra}px)` : base;
}

/** AppLayout 用：把当前可用高度写成 CSS 变量的值 */
export function contentHeightValue(tipsVisible: boolean): string {
  const fixed = WINDOW_PAD * 2 + TITLEBAR_H + (tipsVisible ? TIPSBAR_H : 0);
  return `calc(100vh - ${fixed}px)`;
}
