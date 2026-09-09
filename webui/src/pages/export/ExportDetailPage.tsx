import { useParams, useNavigate } from 'react-router-dom';
import { Tooltip, Typography } from 'antd';
import { RocketOutlined } from '@ant-design/icons';
import {
  TaskDetailShell,
  ArtifactRow,
  MetaCell,
  Panel,
  PillButton,
} from '@/components/TaskDetailShell';
import { LogViewer } from '@/components/LogViewer';
import { MODULES } from '@/theme/modules';
import { brand, neutral } from '@/theme/theme';
import { exportTasks, logLines } from '@/mock/data';

/**
 * 导出详情。产物 Tab 把导出这件事拆成可见的几步（合并 / 量化 / 写出 / 推送），
 * 因为导出失败时用户最想知道「卡在哪一步」。
 */
export function ExportDetailPage({ tab }: { tab: 'artifact' | 'log' }) {
  const { id } = useParams();
  const navigate = useNavigate();
  const task = exportTasks.find((t) => t.id === id);

  const done = task?.status === 'DONE';
  const outDir = `output/${task?.id ?? 'export'}/merged`;

  /** 导出流水线的几步。RUNNING 时最后一步还没完成 */
  const steps = [
    {
      title: '加载来源 checkpoint',
      desc: task?.model,
      done: true,
    },
    {
      title: '合并 LoRA 权重到底座',
      desc: 'merge_lora=true，把 adapter 融进基础模型',
      done: true,
    },
    {
      title: 'int4 量化',
      desc: 'quant_method=gptq · quant_bits=4 · 校准集 128 条',
      done: true,
    },
    {
      title: '写出模型目录',
      desc: `${outDir} · safetensors 分片 + tokenizer + config`,
      done,
    },
    {
      title: '推送到 Hub',
      desc: done ? '未开启（push_to_hub=false）' : '等待前序步骤完成',
      done: false,
    },
  ];

  const artifactTab = (
    <>
      <Panel title="导出步骤">
        {steps.map((s) => (
          <ArtifactRow key={s.title} title={s.title} desc={s.desc} done={s.done} />
        ))}
      </Panel>

      <div style={{ marginTop: 14 }}>
        <Panel title="输出目录">
          <div
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: 10,
              background: neutral.bgCode,
              borderRadius: 11,
              padding: '11px 14px',
              fontFamily: 'ui-monospace, SFMono-Regular, Menlo, monospace',
              fontSize: 12.5,
              color: neutral.text,
            }}
          >
            <span style={{ flex: 1, minWidth: 0, overflow: 'hidden', textOverflow: 'ellipsis' }}>
              {outDir}
            </span>
            <Typography.Text copyable={{ text: outDir, tooltips: ['复制路径', '已复制'] }} />
          </div>
          <div style={{ fontSize: 12.5, color: neutral.textTertiary, marginTop: 9 }}>
            {done
              ? '目录已可直接用于部署或二次训练；血缘会记录它来自哪个训练任务。'
              : '导出仍在进行，目录内容尚不完整，请勿直接使用。'}
          </div>
        </Panel>
      </div>
    </>
  );

  const tabs = [
    { key: 'artifact', label: '产物', content: artifactTab },
    { key: 'log', label: '日志', content: <LogViewer lines={logLines} /> },
  ];

  return (
    <TaskDetailShell
      module={MODULES.export}
      tasks={exportTasks}
      tabs={tabs}
      activeTab={tab}
      extraMeta={<MetaCell label="输出目录" value={outDir} />}
      actions={
        done ? (
          <Tooltip title="以这个导出产物为模型新建部署">
            <PillButton icon={<RocketOutlined />} onClick={() => navigate('/deploy/new')}>
              <span style={{ color: brand.primary }}>去部署</span>
            </PillButton>
          </Tooltip>
        ) : null
      }
    />
  );
}
