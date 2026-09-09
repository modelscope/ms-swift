/**
 * 给 JS 用的色值。
 *
 * 配色的定义在 index.css 的 CSS 变量里（Nature 的纸 + SWIFT logo 的墨），
 * 页面一律用 Tailwind 的语义 class。这个文件只剩一件事：
 * 把其中几个值以 JS 字符串的形式暴露出来，供手绘 SVG 用。
 *
 * 为什么 SVG 需要它：折线图和雷达图是自己画的 `<path stroke={...}>`、
 * `<text fill={...}>`，颜色是当参数传进组件的（同一个 LineChart 要画 loss 和 lr 两条不同色的线），
 * 没法换成 class。值本身仍然是 var()，所以深浅模式切换依旧自动生效。
 *
 * 这里不要再加新东西——需要颜色的地方先想 Tailwind class。
 */

/**
 * 图表用色。这三个名字本来就是从 logo 上取的三段色，现在名实相符了——
 * chart-1/2/3 就是 logo 渐变上采样出来的靛、蓝、紫（见 index.css 文件头）。
 */
export const logo = {
  blue: 'var(--chart-2)',
  indigo: 'var(--chart-1)',
  plum: 'var(--chart-3)',
};

/** 图表的网格线与坐标文字。只有 LineChart / RadarChart 读它 */
export const neutral = {
  /**
   * 皮肤只给了 foreground 和 muted-foreground 两档，中间这档要自己调。
   * 用 color-mix 而不是写死灰值，深浅模式切换时才不会失效。
   */
  textSecondary: 'color-mix(in oklab, var(--foreground) 68%, var(--background))',
  textTertiary: 'var(--muted-foreground)',
  /** 网格线要比 --border 更淡，否则一屏十几条横线会盖过数据本身 */
  borderLight: 'color-mix(in oklab, var(--border) 60%, var(--background))',
};
