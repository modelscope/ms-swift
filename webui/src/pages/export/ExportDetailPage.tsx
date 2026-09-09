import { useParams, useNavigate } from 'react-router-dom';
import { Rocket } from 'lucide-react';
import {
  TaskDetailShell,
  ArtifactRow,
  MetaCell,
  Panel,
  ActionButton,
} from '@/components/TaskDetailShell';
import { CodeLine } from '@/components/CodeBlock';
import { LogViewer } from '@/components/LogViewer';
import { MODULES } from '@/theme/modules';
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

      <div className="mt-3.5">
        <Panel title="输出目录">
          <CodeLine text={outDir} label="复制路径" />
          <div className="text-muted-foreground mt-2.5 text-[12.5px]">
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
          <ActionButton
            icon={<Rocket />}
            tone="primary"
            tooltip="以这个导出产物为模型新建部署"
            onClick={() => navigate('/deploy/new')}
          >
            去部署
          </ActionButton>
        ) : null
      }
    />
  );
}
