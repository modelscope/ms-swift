import { useMemo, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Button, Dropdown, Empty, Input, Popconfirm, Tooltip, Typography, message } from 'antd';
import {
  ApartmentOutlined,
  ApiOutlined,
  CaretRightOutlined,
  CloudServerOutlined,
  DeleteOutlined,
  ExportOutlined,
  FileTextOutlined,
  LineChartOutlined,
  MoreOutlined,
  PauseOutlined,
  PlusOutlined,
  ReloadOutlined,
  RocketOutlined,
  SearchOutlined,
  StopOutlined,
  ThunderboltOutlined,
} from '@ant-design/icons';
import type { ReactNode } from 'react';
import { PageHeader } from './PageHeader';
import { StatusTag } from './StatusTag';
import type { ModuleKey, ModuleMeta } from '@/theme/modules';
import { taskDetailPath } from '@/theme/modules';
import { brand, columnStyle, neutral } from '@/theme/theme';
import { features } from '@/config/features';
import type { TaskItem, TaskStatus } from '@/mock/types';

/** 列表页可选列，各模块按需开启，避免为每个模块复制一份列表页 */
export interface TaskListOptions {
  /** train：显示 step 进度 */
  showProgress?: boolean;
  /** deploy：显示 endpoint */
  showEndpoint?: boolean;
  /** train：显示指标入口 */
  showMetrics?: boolean;
  /** 「新建」按钮文案 */
  createText: string;
  /** 「模型 / 来源」的语义标签，各模块不同 */
  sourceLabel: string;
}

const STATUS_FILTERS: Array<{ label: string; value: TaskStatus | 'ALL' }> = [
  { label: '全部', value: 'ALL' },
  { label: '运行中', value: 'RUNNING' },
  { label: '已完成', value: 'DONE' },
  { label: '失败', value: 'FAILED' },
  { label: '已暂停', value: 'PAUSED' },
];

const MODULE_ICON: Partial<Record<ModuleKey, ReactNode>> = {
  train: <ThunderboltOutlined />,
  eval: <ApartmentOutlined />,
  export: <ExportOutlined />,
  deploy: <RocketOutlined />,
  workflow: <ApiOutlined />,
};

/** 次要信息之间的分隔点 */
function Dot() {
  return <span style={{ color: neutral.border }}>·</span>;
}

export function TaskListPage({
  module,
  tasks,
  options,
}: {
  module: ModuleMeta;
  tasks: TaskItem[];
  options: TaskListOptions;
}) {
  const navigate = useNavigate();
  const [keyword, setKeyword] = useState('');
  const [status, setStatus] = useState<TaskStatus | 'ALL'>('ALL');

  const filtered = useMemo(
    () =>
      tasks.filter(
        (t) =>
          (status === 'ALL' || t.status === status) &&
          (keyword === '' ||
            t.label.includes(keyword) ||
            t.id.includes(keyword) ||
            t.model.includes(keyword)),
      ),
    [tasks, keyword, status],
  );

  /** 只有连了 twinkle-server 的训练才能暂停/继续，本地训练杀了状态就没了 */
  const canPause = (t: TaskItem) => features.pauseResume && t.type === 'train' && Boolean(t.serverUrl);

  const stop = (e: React.MouseEvent) => e.stopPropagation();

  return (
    <>
      <PageHeader
        module={module}
        title={module.label}
        desc={module.desc}
        extra={
          <>
            <Tooltip title="刷新">
              <span
                className="ghost-icon"
                onClick={() => message.success('已刷新（示意）')}
                style={{
                  width: 32,
                  height: 32,
                  borderRadius: 9,
                  display: 'inline-flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  color: neutral.textTertiary,
                  cursor: 'pointer',
                }}
              >
                <ReloadOutlined />
              </span>
            </Tooltip>
            <Button
              type="primary"
              shape="round"
              icon={<PlusOutlined />}
              onClick={() => navigate(`${module.path}/new`)}
            >
              {options.createText}
            </Button>
          </>
        }
      />

      <div style={{ ...columnStyle, paddingBottom: 40 }}>
        {/* 整宽药丸搜索，是这一页唯一的输入框 */}
        <Input
          allowClear
          variant="filled"
          size="large"
          prefix={<SearchOutlined style={{ color: neutral.textTertiary, marginInlineEnd: 6 }} />}
          placeholder={`搜索名称、id、${options.sourceLabel}`}
          style={{ borderRadius: 999 }}
          onChange={(e) => setKeyword(e.target.value)}
        />

        {/* 状态筛选：小字文本按钮，选中才有一点底色 */}
        <div style={{ display: 'flex', gap: 2, margin: '18px 0 6px', flexWrap: 'wrap' }}>
          {STATUS_FILTERS.map((f) => {
            const active = status === f.value;
            const count =
              f.value === 'ALL' ? tasks.length : tasks.filter((t) => t.status === f.value).length;
            return (
              <span
                key={f.value}
                className="ghost-icon"
                onClick={() => setStatus(f.value)}
                style={{
                  display: 'inline-flex',
                  alignItems: 'center',
                  gap: 5,
                  height: 28,
                  padding: '0 10px',
                  borderRadius: 8,
                  fontSize: 13,
                  cursor: 'pointer',
                  color: active ? neutral.text : neutral.textTertiary,
                  background: active ? neutral.bgSubtle : 'transparent',
                  fontWeight: active ? 500 : 400,
                }}
              >
                {f.label}
                <span style={{ fontSize: 12, color: neutral.textTertiary }}>{count}</span>
              </span>
            );
          })}
        </div>

        {/* 列表：无边框、无卡片，靠留白分行 */}
        {filtered.length === 0 ? (
          <div style={{ padding: '72px 0' }}>
            <Empty description="还没有任务" image={Empty.PRESENTED_IMAGE_SIMPLE} />
          </div>
        ) : (
          <div style={{ display: 'flex', flexDirection: 'column' }}>
            {filtered.map((t) => (
              <div
                key={t.id}
                className="task-row"
                onClick={() => navigate(taskDetailPath(t))}
                style={{
                  display: 'flex',
                  alignItems: 'flex-start',
                  gap: 14,
                  padding: '14px 12px',
                  margin: '0 -12px',
                  borderRadius: 12,
                  cursor: 'pointer',
                }}
              >
                {/* 裸图标，不套底色方块 */}
                <span
                  style={{
                    flex: 'none',
                    marginTop: 2,
                    fontSize: 17,
                    color: module.accent,
                    lineHeight: '20px',
                  }}
                >
                  {MODULE_ICON[module.key] ?? <ApiOutlined />}
                </span>

                <div style={{ flex: 1, minWidth: 0 }}>
                  {/* 标题行：名字 + 行内灰时间 + 状态 */}
                  <div style={{ display: 'flex', alignItems: 'center', gap: 10, flexWrap: 'wrap' }}>
                    <span
                      style={{
                        fontSize: 15,
                        fontWeight: 500,
                        color: neutral.text,
                        overflow: 'hidden',
                        whiteSpace: 'nowrap',
                        textOverflow: 'ellipsis',
                        maxWidth: '100%',
                      }}
                    >
                      {t.label}
                    </span>
                    <span style={{ fontSize: 13, color: neutral.textTertiary }}>{t.createdAt}</span>
                    <StatusTag status={t.status} />
                  </div>

                  {/* 描述行：一行灰字串起所有元信息，不再堆胶囊 */}
                  <div
                    style={{
                      display: 'flex',
                      alignItems: 'center',
                      gap: 7,
                      flexWrap: 'wrap',
                      marginTop: 5,
                      fontSize: 13.5,
                      color: neutral.textSecondary,
                      lineHeight: '20px',
                    }}
                  >
                    <span>{t.model}</span>
                    <Dot />
                    <span>{t.runner}</span>
                    {t.serverUrl && (
                      <>
                        <Dot />
                        <Tooltip title={`训练循环在本地，算子下沉到 ${t.serverUrl}`}>
                          <span style={{ color: brand.primary, display: 'inline-flex', alignItems: 'center', gap: 4 }}>
                            <CloudServerOutlined /> server
                          </span>
                        </Tooltip>
                      </>
                    )}
                    {options.showProgress && t.totalSteps ? (
                      <>
                        <Dot />
                        <span>
                          {t.step ?? 0} / {t.totalSteps} step ·{' '}
                          {Math.round(((t.step ?? 0) / t.totalSteps) * 100)}%
                        </span>
                      </>
                    ) : null}
                    {options.showEndpoint ? (
                      <>
                        <Dot />
                        {t.endpoint ? (
                          <span onClick={stop}>
                            <Typography.Text
                              copyable={{ text: t.endpoint }}
                              style={{ fontSize: 13.5, color: neutral.textSecondary }}
                            >
                              {t.endpoint}
                            </Typography.Text>
                          </span>
                        ) : (
                          <Tooltip title="进程启动后自选端口并写回 runtime.json，此处等它出现">
                            <span style={{ color: neutral.textTertiary }}>端口分配中…</span>
                          </Tooltip>
                        )}
                      </>
                    ) : null}
                    <Dot />
                    <span
                      style={{
                        color: neutral.textTertiary,
                        fontFamily: 'ui-monospace, SFMono-Regular, Menlo, monospace',
                        fontSize: 12.5,
                      }}
                    >
                      {t.id}
                    </span>
                  </div>
                </div>

                <div
                  className="row-actions"
                  style={{ display: 'flex', alignItems: 'center', gap: 2, flex: 'none' }}
                  onClick={stop}
                >
                  {options.showMetrics && (
                    <Tooltip title="指标曲线">
                      <Button
                        type="text"
                        shape="circle"
                        icon={<LineChartOutlined />}
                        onClick={() => navigate(`${module.path}/${t.id}/metrics`)}
                      />
                    </Tooltip>
                  )}
                  <Tooltip title="日志">
                    <Button
                      type="text"
                      shape="circle"
                      icon={<FileTextOutlined />}
                      onClick={() => navigate(taskDetailPath(t))}
                    />
                  </Tooltip>
                  {t.status === 'RUNNING' && canPause(t) && (
                    <Tooltip title="暂停后服务端仍保留训练状态，显存不释放">
                      <Button
                        type="text"
                        shape="circle"
                        icon={<PauseOutlined />}
                        onClick={() => message.info('暂停（示意，未接后端）')}
                      />
                    </Tooltip>
                  )}
                  {t.status === 'PAUSED' && (
                    <Tooltip title="用同一个 adapter_name 起新客户端，无损续跑">
                      <Button
                        type="text"
                        shape="circle"
                        icon={<CaretRightOutlined />}
                        onClick={() => message.info('继续（示意，未接后端）')}
                      />
                    </Tooltip>
                  )}
                  {t.status === 'RUNNING' && (
                    <Popconfirm
                      title="停止该任务？"
                      description="先发 SIGTERM 让它有机会保存 checkpoint"
                      onConfirm={() => message.info('停止（示意，未接后端）')}
                    >
                      <Button type="text" shape="circle" danger icon={<StopOutlined />} />
                    </Popconfirm>
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
                    <Button type="text" shape="circle" icon={<MoreOutlined />} />
                  </Dropdown>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </>
  );
}
