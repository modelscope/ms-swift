import { useMemo } from 'react';
import type { ReactNode } from 'react';
import { Link, useLocation, useNavigate } from 'react-router-dom';
import { Dropdown, Tooltip } from 'antd';
import {
  ApiOutlined,
  ApartmentOutlined,
  BellOutlined,
  DownOutlined,
  ExportOutlined,
  MessageOutlined,
  PlusOutlined,
  QuestionCircleOutlined,
  RocketOutlined,
  SearchOutlined,
  ThunderboltOutlined,
} from '@ant-design/icons';
import type { ModuleKey } from '@/theme/modules';
import { MODULES, MODULE_ORDER, moduleOf, taskDetailPath } from '@/theme/modules';
import { chrome } from '@/theme/theme';
import { features } from '@/config/features';
import { tasksByType } from '@/mock/data';

const ICONS: Record<ModuleKey, ReactNode> = {
  chat: <MessageOutlined />,
  train: <ThunderboltOutlined />,
  eval: <ApartmentOutlined />,
  export: <ExportOutlined />,
  deploy: <RocketOutlined />,
  workflow: <ApiOutlined />,
};

/** 无框幽灵图标按钮。刻意脱离导航行之外，不排成工具栏 */
function GhostIcon({
  icon,
  title,
  size = 24,
  onClick,
}: {
  icon: ReactNode;
  title: string;
  size?: number;
  onClick?: () => void;
}) {
  return (
    <Tooltip title={title} mouseEnterDelay={0.4}>
      <span
        className="chrome-icon-btn"
        onClick={(e) => {
          e.preventDefault();
          e.stopPropagation();
          onClick?.();
        }}
        style={{
          width: size,
          height: size,
          borderRadius: 7,
          flex: 'none',
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
    </Tooltip>
  );
}

/** 分组小标签，用来打断长列表，而不是一长条等距菜单 */
function SectionLabel({ children }: { children: ReactNode }) {
  return (
    <div
      style={{
        fontSize: 12,
        color: chrome.textFaint,
        padding: '0 11px',
        margin: '18px 0 6px',
        userSelect: 'none',
      }}
    >
      {children}
    </div>
  );
}

/**
 * 左侧导航。信息设计参考 Codex：小字号、紧凑行高、细图标，
 * 选中态只是一块低对比底色——导航里不出现品牌色，让内容区去承担强调。
 * 操作按钮走「非线性」：贴边的无框幽灵按钮，而非并排的按钮组。
 */
export function Sidebar({ collapsed }: { collapsed: boolean }) {
  const { pathname } = useLocation();
  const navigate = useNavigate();
  const current = moduleOf(pathname);

  const items = useMemo(
    () => MODULE_ORDER.filter((k) => features.modules.includes(k)).map((k) => MODULES[k]),
    [],
  );

  /** 「最近」分组：跨模块取最新几条，给侧栏一点活气 */
  const recent = useMemo(
    () =>
      Object.values(tasksByType)
        .flat()
        .sort((a, b) => b.createdAt.localeCompare(a.createdAt))
        .slice(0, 5),
    [],
  );

  const rowBase: React.CSSProperties = {
    display: 'flex',
    alignItems: 'center',
    gap: 10,
    height: 32,
    borderRadius: 8,
    fontSize: 13,
    textDecoration: 'none',
    padding: collapsed ? 0 : '0 11px',
    justifyContent: collapsed ? 'center' : 'flex-start',
  };

  return (
    <div
      style={{
        width: collapsed ? 60 : 240,
        flex: 'none',
        height: '100%',
        display: 'flex',
        flexDirection: 'column',
        background: chrome.sidebar,
        borderInlineEnd: `1px solid ${chrome.border}`,
        transition: 'width 0.18s ease',
        overflow: 'hidden',
      }}
    >
      {/* 品牌区：纯文字 + 小箭头，右侧贴边的无框图标 */}
      <div
        style={{
          height: 46,
          flex: 'none',
          display: 'flex',
          alignItems: 'center',
          gap: 6,
          padding: collapsed ? '0 16px' : '0 12px 0 14px',
        }}
      >
        {collapsed ? (
          <span style={{ fontWeight: 700, fontSize: 14, color: chrome.text }}>S</span>
        ) : (
          <>
            <Dropdown
              trigger={['click']}
              menu={{
                onClick: ({ key }) => {
                  if (key === 'settings') navigate('/settings');
                },
                items: [
                  { key: 'about', label: '关于 SWIFT WebUI' },
                  { key: 'settings', label: '偏好设置' },
                  { type: 'divider' },
                  { key: 'docs', label: '文档' },
                ],
              }}
            >
              <span
                style={{
                  display: 'inline-flex',
                  alignItems: 'center',
                  gap: 5,
                  cursor: 'pointer',
                  fontWeight: 600,
                  fontSize: 15,
                  color: chrome.text,
                  letterSpacing: 0.2,
                }}
              >
                SWIFT
                <DownOutlined style={{ fontSize: 9, color: chrome.textFaint }} />
              </span>
            </Dropdown>
            <div style={{ marginInlineStart: 'auto', display: 'flex', gap: 2 }}>
              <GhostIcon icon={<SearchOutlined />} title="搜索  ⌘K" size={26} />
              <GhostIcon icon={<BellOutlined />} title="通知" size={26} />
            </div>
          </>
        )}
      </div>

      <div style={{ flex: 1, overflowY: 'auto', padding: collapsed ? '2px 8px' : '2px 8px' }}>
        {/* 模块导航 */}
        {items.map((m) => {
          const active = current.key === m.key;
          const row = (
            <Link
              key={m.key}
              to={m.path}
              className="nav-item"
              style={{
                ...rowBase,
                color: active ? chrome.text : chrome.textDim,
                background: active ? chrome.hover : 'transparent',
                fontWeight: active ? 500 : 400,
                marginBottom: 1,
                /** 对话行右侧留出脱离行外的 + 按钮位置 */
                paddingInlineEnd: !collapsed && m.key === 'chat' ? 4 : undefined,
              }}
            >
              <span style={{ fontSize: 14, display: 'inline-flex', color: 'inherit', opacity: 0.9 }}>
                {ICONS[m.key]}
              </span>
              {!collapsed && m.label}
              {!collapsed && m.key === 'chat' && (
                <span style={{ marginInlineStart: 'auto' }}>
                  <GhostIcon
                    icon={<PlusOutlined />}
                    title="新建对话"
                    onClick={() => navigate('/chat')}
                  />
                </span>
              )}
            </Link>
          );
          return collapsed ? (
            <Tooltip key={m.key} title={m.label} placement="right">
              {row}
            </Tooltip>
          ) : (
            row
          );
        })}

        {/* 最近：跨模块的近期任务 */}
        {!collapsed && (
          <>
            <SectionLabel>最近</SectionLabel>
            {recent.map((t) => (
              <Link
                key={t.id}
                to={taskDetailPath(t)}
                className="nav-item"
                style={{
                  ...rowBase,
                  height: 30,
                  color: pathname.includes(t.id) ? chrome.text : chrome.textDim,
                  background: pathname.includes(t.id) ? chrome.hover : 'transparent',
                }}
              >
                <span
                  style={{
                    overflow: 'hidden',
                    whiteSpace: 'nowrap',
                    textOverflow: 'ellipsis',
                  }}
                >
                  {t.label}
                </span>
              </Link>
            ))}
          </>
        )}
      </div>

      {/* 底部：空间切换 + 用户，右侧贴边帮助 */}
      <div
        style={{
          flex: 'none',
          borderTop: `1px solid ${chrome.border}`,
          padding: collapsed ? '8px' : '8px',
        }}
      >
        {features.multiTenant && !collapsed && (
          <Dropdown
            trigger={['click']}
            menu={{
              items: [
                { key: 'default', label: 'default（默认空间）' },
                { key: 'team-a', label: 'team-a' },
                { key: 'team-b', label: 'team-b' },
              ],
            }}
          >
            <div className="nav-item" style={{ ...rowBase, color: chrome.textDim, cursor: 'pointer' }}>
              <span style={{ color: chrome.textFaint }}>空间</span>
              <span style={{ color: chrome.text }}>default</span>
              <DownOutlined style={{ marginInlineStart: 'auto', fontSize: 9, color: chrome.textFaint }} />
            </div>
          </Dropdown>
        )}
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: 8,
            height: 34,
            padding: collapsed ? 0 : '0 11px',
            justifyContent: collapsed ? 'center' : 'flex-start',
          }}
        >
          <div
            style={{
              width: 22,
              height: 22,
              flex: 'none',
              borderRadius: '50%',
              background: 'rgba(255,255,255,0.12)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              color: chrome.text,
              fontSize: 11,
              fontWeight: 600,
            }}
          >
            U
          </div>
          {!collapsed && (
            <>
              <span
                style={{
                  fontSize: 13,
                  color: chrome.textDim,
                  overflow: 'hidden',
                  whiteSpace: 'nowrap',
                  textOverflow: 'ellipsis',
                }}
              >
                user@swift
              </span>
              <span style={{ marginInlineStart: 'auto' }}>
                <GhostIcon icon={<QuestionCircleOutlined />} title="帮助" size={26} />
              </span>
            </>
          )}
        </div>
      </div>
    </div>
  );
}
