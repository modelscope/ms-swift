import type { ReactNode } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import { Tooltip } from 'antd';
import {
  ArrowLeftOutlined,
  ArrowRightOutlined,
  BookOutlined,
  GithubOutlined,
  LayoutOutlined,
} from '@ant-design/icons';
import { moduleOf } from '@/theme/modules';
import { chrome } from '@/theme/theme';

/**
 * 窗口标题栏。参考 Codex：交通灯 + 侧栏开合 + 前进/后退 都是无框细图标，
 * 中间只放当前位置的轻量说明，右侧贴边放全局链接。
 * 刻意不用面包屑，也不做居中的大搜索条——搜索入口在侧栏。
 */

function TrafficLights() {
  const dots = ['#FF5F57', '#FEBC2E', '#28C840'];
  return (
    <div style={{ display: 'flex', gap: 8, marginInlineEnd: 8 }}>
      {dots.map((c) => (
        <span key={c} style={{ width: 11, height: 11, borderRadius: '50%', background: c }} />
      ))}
    </div>
  );
}

function IconBtn({
  icon,
  title,
  onClick,
  href,
}: {
  icon: ReactNode;
  title: string;
  onClick?: () => void;
  href?: string;
}) {
  const inner = (
    <span
      className="chrome-icon-btn"
      onClick={onClick}
      style={{
        width: 26,
        height: 26,
        borderRadius: 7,
        display: 'inline-flex',
        alignItems: 'center',
        justifyContent: 'center',
        color: chrome.textFaint,
        cursor: 'pointer',
        fontSize: 13,
      }}
    >
      {icon}
    </span>
  );
  return (
    <Tooltip title={title} mouseEnterDelay={0.4}>
      {href ? (
        <a href={href} target="_blank" rel="noreferrer">
          {inner}
        </a>
      ) : (
        inner
      )}
    </Tooltip>
  );
}

export function Titlebar({
  collapsed,
  onToggleSidebar,
}: {
  collapsed: boolean;
  onToggleSidebar: () => void;
}) {
  const navigate = useNavigate();
  const { pathname } = useLocation();
  const current = moduleOf(pathname);
  const rest = pathname.slice(current.path.length).split('/').filter(Boolean);

  const tail: Record<string, string> = {
    new: '新建',
    log: '日志',
    metrics: '指标',
    result: '结果',
    edit: '编辑',
  };
  const sub = rest.map((s) => tail[s] ?? s).join(' · ');

  return (
    <div
      style={{
        height: 40,
        flex: 'none',
        display: 'flex',
        alignItems: 'center',
        gap: 2,
        padding: '0 10px',
        background: chrome.titlebar,
        borderBottom: `1px solid ${chrome.border}`,
        userSelect: 'none',
      }}
    >
      <TrafficLights />
      <IconBtn
        icon={<LayoutOutlined />}
        title={collapsed ? '展开侧栏' : '收起侧栏'}
        onClick={onToggleSidebar}
      />
      <IconBtn icon={<ArrowLeftOutlined />} title="后退" onClick={() => navigate(-1)} />
      <IconBtn icon={<ArrowRightOutlined />} title="前进" onClick={() => navigate(1)} />

      <div
        style={{
          flex: 1,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          gap: 8,
          minWidth: 0,
        }}
      >
        <span style={{ fontSize: 12.5, color: chrome.textDim, whiteSpace: 'nowrap' }}>
          {current.label}
        </span>
        {sub && (
          <span style={{ fontSize: 12.5, color: chrome.textFaint, whiteSpace: 'nowrap' }}>
            {sub}
          </span>
        )}
      </div>

      <IconBtn icon={<BookOutlined />} title="文档" href="https://swift.readthedocs.io" />
      <IconBtn icon={<GithubOutlined />} title="GitHub" href="https://github.com/modelscope/ms-swift" />
    </div>
  );
}
