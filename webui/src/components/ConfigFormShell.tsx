import type { ReactNode } from 'react';
import { useNavigate } from 'react-router-dom';
import { Alert, Button, Card, Space, message } from 'antd';
import { ArrowLeftOutlined, PlayCircleOutlined, SaveOutlined } from '@ant-design/icons';
import { PageHeader } from './PageHeader';
import type { ModuleMeta } from '@/theme/modules';
import { neutral } from '@/theme/theme';

/**
 * 新建任务页的外壳：左边表单分区，右边贴一个「即将执行什么」的预览。
 * 预览区展示的就是最终会落到任务目录里的 run.sh，让用户在提交前看清要跑什么，
 * 也顺便解释了任务与文件目录的对应关系。
 */
export function ConfigFormShell({
  module,
  title,
  desc,
  sections,
  preview,
  warning,
}: {
  module: ModuleMeta;
  title: string;
  desc?: string;
  sections: Array<{ title: string; extra?: ReactNode; content: ReactNode }>;
  preview: string;
  warning?: string;
}) {
  const navigate = useNavigate();

  return (
    <>
      <PageHeader
        module={module}
        title={title}
        desc={desc}
        back={
          <Button
            type="link"
            size="small"
            icon={<ArrowLeftOutlined />}
            style={{ padding: 0, height: 20, marginBottom: 2 }}
            onClick={() => navigate(module.path)}
          >
            返回{module.label}列表
          </Button>
        }
        extra={
          <Space>
            <Button icon={<SaveOutlined />} onClick={() => message.success('已存为草稿（示意）')}>
              存为草稿
            </Button>
            <Button
              type="primary"
              icon={<PlayCircleOutlined />}
              onClick={() => message.success('已提交（示意，未接后端）')}
            >
              提交运行
            </Button>
          </Space>
        }
      />

      <div style={{ display: 'flex', gap: 20, padding: '18px 24px 32px', alignItems: 'flex-start' }}>
        <div style={{ flex: 1, minWidth: 0, display: 'flex', flexDirection: 'column', gap: 16 }}>
          {warning && <Alert type="warning" showIcon message={warning} />}
          {sections.map((s) => (
            <Card key={s.title} size="small" title={s.title} extra={s.extra}>
              {s.content}
            </Card>
          ))}
        </div>

        <Card
          size="small"
          title="将要执行"
          style={{ width: 420, flex: 'none', position: 'sticky', top: 16 }}
        >
          <div
            style={{
              background: neutral.bgCode,
              borderRadius: 6,
              padding: 12,
              fontFamily: 'ui-monospace, SFMono-Regular, Menlo, Consolas, monospace',
              fontSize: 12,
              lineHeight: '19px',
              whiteSpace: 'pre-wrap',
              wordBreak: 'break-all',
              color: neutral.text,
              maxHeight: 420,
              overflow: 'auto',
            }}
          >
            {preview}
          </div>
          <div style={{ marginTop: 10, fontSize: 12, color: neutral.textTertiary, lineHeight: '18px' }}>
            提交后会在任务目录下生成 run.sh 与配置文件，由它负责重定向日志、执行命令、并在结束时写出退出码。
          </div>
        </Card>
      </div>
    </>
  );
}
