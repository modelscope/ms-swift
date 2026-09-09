import type { ReactNode } from 'react';
import { Link, useNavigate, useParams } from 'react-router-dom';
import { Dropdown, Empty, Tooltip, Typography, message } from 'antd';
import {
  CaretRightOutlined,
  CheckOutlined,
  DeleteOutlined,
  LeftOutlined,
  MoreOutlined,
  PauseOutlined,
  ReloadOutlined,
  StopOutlined,
  WarningOutlined,
} from '@ant-design/icons';
import { PageHeader } from './PageHeader';
import { StatusTag } from './StatusTag';
import type { ModuleMeta } from '@/theme/modules';
import { brand, neutral, wideColumnStyle } from '@/theme/theme';
import { features } from '@/config/features';
import type { TaskItem } from '@/mock/types';

export interface DetailTab {
  key: string;
  label: string;
  content: ReactNode;
}

/** 一枚元信息：标签在上、值在下。取代 antd Descriptions 那种「标签：值」的表格观感 */
export function MetaCell({ label, value }: { label: string; value: ReactNode }) {
  return (
    <div style={{ minWidth: 0 }}>
      <div style={{ fontSize: 12, color: neutral.textTertiary, marginBottom: 3 }}>{label}</div>
      <div
        style={{
          fontSize: 13.5,
          color: neutral.text,
          overflow: 'hidden',
          textOverflow: 'ellipsis',
          whiteSpace: 'nowrap',
        }}
      >
        {value}
      </div>
    </div>
  );
}

/** 元信息条：一排无边框的键值对，横向铺开 */
export function MetaStrip({ children }: { children: ReactNode }) {
  return (
    <div
      style={{
        display: 'flex',
        gap: 34,
        flexWrap: 'wrap',
        padding: '14px 0 18px',
        borderTop: `1px solid ${neutral.borderLight}`,
      }}
    >
      {children}
    </div>
  );
}

/** 提示条。自绘软底，不用 antd Alert 那种带框带图标底的样式 */
export function Notice({
  tone,
  title,
  children,
}: {
  tone: 'warn' | 'error';
  title: string;
  children?: ReactNode;
}) {
  const c =
    tone === 'warn'
      ? { fg: '#8A6320', bg: '#FDF8EC' }
      : { fg: '#B03A3A', bg: '#FCF1F1' };
  return (
    <div
      style={{
        background: c.bg,
        borderRadius: 12,
        padding: '13px 16px',
        marginBottom: 18,
      }}
    >
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: 7,
          fontSize: 13.5,
          fontWeight: 500,
          color: c.fg,
        }}
      >
        <WarningOutlined /> {title}
      </div>
      {children && (
        <div style={{ fontSize: 13, color: neutral.textSecondary, marginTop: 5, lineHeight: '20px' }}>
          {children}
        </div>
      )}
    </div>
  );
}

/** 圆角胶囊按钮，浅色内容区专用。分主次两级，不用 antd 的默认描边按钮 */
export function PillButton({
  icon,
  children,
  tone = 'ghost',
  onClick,
}: {
  icon?: ReactNode;
  children?: ReactNode;
  tone?: 'primary' | 'ghost' | 'danger';
  onClick?: () => void;
}) {
  const skin =
    tone === 'primary'
      ? { color: '#fff', background: brand.primary }
      : tone === 'danger'
        ? { color: '#B03A3A', background: '#FCF1F1' }
        : { color: neutral.textSecondary, background: neutral.bgSubtle };
  return (
    <span
      className={tone === 'ghost' ? 'ghost-icon' : 'pill-btn'}
      onClick={onClick}
      style={{
        ...skin,
        display: 'inline-flex',
        alignItems: 'center',
        gap: 6,
        height: 32,
        padding: children ? '0 14px' : 0,
        width: children ? undefined : 32,
        justifyContent: 'center',
        borderRadius: 999,
        fontSize: 13,
        fontWeight: 500,
        cursor: 'pointer',
        userSelect: 'none',
      }}
    >
      {icon}
      {children}
    </span>
  );
}

/** 自绘 Tab 条：小胶囊，去掉 antd Tabs 的下划线与 ink bar */
export function TabStrip({
  tabs,
  activeKey,
  onChange,
}: {
  tabs: Array<{ key: string; label: string }>;
  activeKey: string;
  onChange: (k: string) => void;
}) {
  return (
    <div style={{ display: 'flex', gap: 2, marginBottom: 18 }}>
      {tabs.map((t) => {
        const active = t.key === activeKey;
        return (
          <span
            key={t.key}
            className="ghost-icon"
            onClick={() => onChange(t.key)}
            style={{
              height: 30,
              padding: '0 13px',
              borderRadius: 9,
              display: 'inline-flex',
              alignItems: 'center',
              fontSize: 13.5,
              cursor: 'pointer',
              color: active ? neutral.text : neutral.textTertiary,
              background: active ? neutral.bgSubtle : 'transparent',
              fontWeight: active ? 500 : 400,
            }}
          >
            {t.label}
          </span>
        );
      })}
    </div>
  );
}

/**
 * 任务详情外壳：返回、状态、元信息、操作、Tab 切换都在这里，
 * 各模块只负责提供 Tab 内容。切 Tab 走路由，刷新后停在同一个 Tab。
 */
export function TaskDetailShell({
  module,
  tasks,
  tabs,
  activeTab,
  extraMeta,
  actions,
}: {
  module: ModuleMeta;
  tasks: TaskItem[];
  tabs: DetailTab[];
  activeTab: string;
  /** 模块特有的元信息格 */
  extraMeta?: ReactNode;
  /** 模块特有的操作按钮，放在通用操作左侧 */
  actions?: ReactNode;
}) {
  const { id } = useParams();
  const navigate = useNavigate();
  const task = tasks.find((t) => t.id === id);

  if (!task) {
    return (
      <div style={{ padding: '96px 0' }}>
        <Empty description={`找不到任务 ${id}`} image={Empty.PRESENTED_IMAGE_SIMPLE}>
          <PillButton tone="ghost" onClick={() => navigate(module.path)}>
            返回{module.label}
          </PillButton>
        </Empty>
      </div>
    );
  }

  const canPause = features.pauseResume && task.type === 'train' && Boolean(task.serverUrl);
  const current = tabs.find((t) => t.key === activeTab) ?? tabs[0];

  return (
    <>
      <PageHeader
        module={module}
        wide
        size="md"
        title={task.label}
        titleExtra={<StatusTag status={task.status} />}
        back={
          <Link
            to={module.path}
            style={{
              display: 'inline-flex',
              alignItems: 'center',
              gap: 4,
              fontSize: 12.5,
              color: neutral.textTertiary,
              marginBottom: 8,
            }}
          >
            <LeftOutlined style={{ fontSize: 9 }} /> {module.label}
          </Link>
        }
        extra={
          <>
            {actions}
            <Tooltip title="刷新">
              <PillButton icon={<ReloadOutlined />} onClick={() => message.success('已刷新（示意）')} />
            </Tooltip>
            {task.status === 'RUNNING' && canPause && (
              <Tooltip title="暂停后服务端仍保留训练状态，显存不释放">
                <PillButton icon={<PauseOutlined />} onClick={() => message.info('暂停（示意）')}>
                  暂停
                </PillButton>
              </Tooltip>
            )}
            {task.status === 'PAUSED' && (
              <PillButton
                tone="primary"
                icon={<CaretRightOutlined />}
                onClick={() => message.info('继续（示意）')}
              >
                继续
              </PillButton>
            )}
            {task.status === 'RUNNING' && (
              <PillButton tone="danger" icon={<StopOutlined />} onClick={() => message.info('停止（示意）')}>
                停止
              </PillButton>
            )}
            <Dropdown
              menu={{
                items: [
                  { key: 'rename', label: '重命名' },
                  { key: 'clone', label: '以此为模板新建' },
                  { type: 'divider' },
                  { key: 'delete', label: '删除', danger: true, icon: <DeleteOutlined /> },
                ],
                onClick: () => message.info('示意，未接后端'),
              }}
            >
              <PillButton icon={<MoreOutlined />} />
            </Dropdown>
          </>
        }
      />

      <div style={{ ...wideColumnStyle, paddingBottom: 40 }}>
        {task.status === 'PAUSED' && (
          <Notice tone="warn" title="任务已暂停，但服务端仍占用显存">
            本地客户端进程已退出，服务端仍在 GPU 中保留模型与优化器状态。点「继续」可无损续跑；确定不再需要请点「停止」，那会真正释放服务端资源。
          </Notice>
        )}
        {task.status === 'FAILED' && task.error && (
          <Notice tone="error" title="任务失败">
            {task.error}
          </Notice>
        )}

        {/* 元信息条：取代 Descriptions */}
        <MetaStrip>
          <MetaCell label="模型 / 来源" value={task.model} />
          <MetaCell label="运行方式" value={task.runner} />
          <MetaCell
            label="服务端"
            value={
              task.serverUrl ?? <span style={{ color: neutral.textTertiary }}>本地</span>
            }
          />
          <MetaCell label="创建时间" value={task.createdAt} />
          {task.endpoint && (
            <MetaCell
              label="服务地址"
              value={
                <Typography.Text copyable style={{ fontSize: 13.5, color: brand.primary }}>
                  {task.endpoint}
                </Typography.Text>
              }
            />
          )}
          <MetaCell
            label="任务 id"
            value={
              <span
                style={{
                  fontFamily: 'ui-monospace, SFMono-Regular, Menlo, monospace',
                  fontSize: 12.5,
                  color: neutral.textSecondary,
                }}
              >
                {task.id}
              </span>
            }
          />
          {extraMeta}
        </MetaStrip>

        <TabStrip
          tabs={tabs}
          activeKey={current.key}
          onChange={(k) => navigate(`${module.path}/${task.id}/${k}`)}
        />

        {current.content}
      </div>
    </>
  );
}

/** 详情页里的内容块。无边框标题 + 内容，取代到处都是的 antd Card */
export function Panel({
  title,
  extra,
  children,
  padded = true,
}: {
  title?: ReactNode;
  extra?: ReactNode;
  children: ReactNode;
  padded?: boolean;
}) {
  return (
    <div
      style={{
        background: '#fff',
        border: `1px solid ${neutral.borderLight}`,
        borderRadius: 14,
        overflow: 'hidden',
      }}
    >
      {title && (
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            gap: 12,
            padding: '12px 16px 0',
          }}
        >
          <span style={{ fontSize: 13.5, fontWeight: 500, color: neutral.text }}>{title}</span>
          {extra}
        </div>
      )}
      <div style={{ padding: padded ? '12px 16px 16px' : 0 }}>{children}</div>
    </div>
  );
}

/** 一枚大数字指标。取代 antd Card + Statistic 的组合 */
export function StatCell({
  label,
  value,
  suffix,
  hint,
}: {
  label: string;
  value: ReactNode;
  suffix?: ReactNode;
  hint?: string;
}) {
  const body = (
    <div
      style={{
        background: neutral.bgSubtle,
        borderRadius: 13,
        padding: '13px 16px',
        minWidth: 0,
        flex: 1,
      }}
    >
      <div style={{ fontSize: 12, color: neutral.textTertiary, marginBottom: 5 }}>{label}</div>
      <div
        style={{
          fontSize: 23,
          fontWeight: 600,
          color: neutral.text,
          letterSpacing: -0.5,
          lineHeight: '28px',
        }}
      >
        {value}
        {suffix && (
          <span style={{ fontSize: 13, fontWeight: 400, color: neutral.textTertiary, marginInlineStart: 5 }}>
            {suffix}
          </span>
        )}
      </div>
    </div>
  );
  return hint ? <Tooltip title={hint}>{body}</Tooltip> : body;
}

/** 一行「已完成」式的产物条目 */
export function ArtifactRow({
  title,
  desc,
  done,
  extra,
}: {
  title: ReactNode;
  desc?: ReactNode;
  done?: boolean;
  extra?: ReactNode;
}) {
  return (
    <div
      className="task-row"
      style={{
        display: 'flex',
        alignItems: 'center',
        gap: 13,
        padding: '12px 12px',
        margin: '0 -12px',
        borderRadius: 12,
      }}
    >
      <span
        style={{
          flex: 'none',
          width: 22,
          height: 22,
          borderRadius: '50%',
          background: done ? '#EDF7F1' : neutral.bgSubtle,
          color: done ? '#1F6F4A' : neutral.textTertiary,
          display: 'inline-flex',
          alignItems: 'center',
          justifyContent: 'center',
          fontSize: 11,
        }}
      >
        {done ? <CheckOutlined /> : ''}
      </span>
      <div style={{ flex: 1, minWidth: 0 }}>
        <div style={{ fontSize: 14, color: neutral.text }}>{title}</div>
        {desc && (
          <div style={{ fontSize: 13, color: neutral.textTertiary, marginTop: 3 }}>{desc}</div>
        )}
      </div>
      {extra}
    </div>
  );
}
