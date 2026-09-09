import { useEffect, useMemo, useState } from 'react';
import { Link } from 'react-router-dom';
import { Tooltip } from 'antd';
import { BulbOutlined, CloseOutlined, ReloadOutlined } from '@ant-design/icons';
import { visibleTips } from '@/config/tips';
import { useSettings } from '@/settings/SettingsContext';
import { TIPSBAR_H } from './metrics';
import { chrome } from '@/theme/theme';

/**
 * 底部 tips 条。
 *
 * 放在窗口最下面一条 26px 的深色带里，跟标题栏呼应，把浅色内容区夹在中间。
 * 文案每隔一段时间换一条，淡入淡出。
 *
 * 两个退出口分开设计：
 *  - 叉掉 = 这次别烦我，刷新后还会回来
 *  - 设置里关掉 = 以后都别出现
 * 所以叉的 tooltip 里要写清楚「去设置里可以永久关闭」，否则用户会反复叉。
 */
export function TipsBar() {
  const { settings, dismissTips } = useSettings();
  const pool = useMemo(() => visibleTips(settings.tipsBeginnerOnly), [settings.tipsBeginnerOnly]);

  /* 每次进来从随机一条开始，否则每次打开都是同一句 */
  const [idx, setIdx] = useState(() => Math.floor(Math.random() * Math.max(1, pool.length)));
  /* 切换时先淡出再换字，避免文字直接跳变 */
  const [visible, setVisible] = useState(true);

  const next = () => {
    setVisible(false);
    window.setTimeout(() => {
      setIdx((i) => (i + 1) % Math.max(1, pool.length));
      setVisible(true);
    }, 220);
  };

  useEffect(() => {
    const ms = Math.max(4, settings.tipsInterval) * 1000;
    const timer = window.setInterval(next, ms);
    return () => window.clearInterval(timer);
  }, [settings.tipsInterval, pool.length]);

  /* 池子变小后当前下标可能越界 */
  const tip = pool[idx % Math.max(1, pool.length)];
  if (!tip) return null;

  return (
    <div
      style={{
        height: TIPSBAR_H,
        flex: 'none',
        display: 'flex',
        alignItems: 'center',
        gap: 8,
        padding: '0 10px 0 12px',
        background: chrome.titlebar,
        borderTop: `1px solid ${chrome.border}`,
        fontSize: 11.5,
        color: chrome.textDim,
        overflow: 'hidden',
      }}
    >
      <BulbOutlined style={{ fontSize: 11, color: '#C9A227', flex: 'none' }} />
      <span
        style={{
          flex: 1,
          minWidth: 0,
          overflow: 'hidden',
          whiteSpace: 'nowrap',
          textOverflow: 'ellipsis',
          opacity: visible ? 1 : 0,
          transform: visible ? 'translateY(0)' : 'translateY(3px)',
          transition: 'opacity 0.22s ease, transform 0.22s ease',
        }}
      >
        {tip.text}
      </span>

      <Tooltip title="换一条">
        <span
          className="chrome-icon-btn"
          onClick={next}
          style={tipsBtnStyle}
        >
          <ReloadOutlined style={{ fontSize: 10 }} />
        </span>
      </Tooltip>

      <Tooltip
        title={
          <span style={{ fontSize: 12 }}>
            本次关闭。想以后都不显示，去{' '}
            <Link to="/settings" style={{ color: '#A8AEE0' }}>
              偏好设置
            </Link>{' '}
            里关掉
          </span>
        }
      >
        <span className="chrome-icon-btn" onClick={dismissTips} style={tipsBtnStyle}>
          <CloseOutlined style={{ fontSize: 10 }} />
        </span>
      </Tooltip>
    </div>
  );
}

const tipsBtnStyle: React.CSSProperties = {
  width: 18,
  height: 18,
  borderRadius: 5,
  display: 'inline-flex',
  alignItems: 'center',
  justifyContent: 'center',
  color: chrome.textFaint,
  cursor: 'pointer',
  flex: 'none',
};
