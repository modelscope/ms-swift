import { useNavigate, useParams } from 'react-router-dom';
import { MessageSquare } from 'lucide-react';
import { toast } from 'sonner';
import {
  TaskDetailShell,
  ArtifactRow,
  MetaCell,
  Panel,
  ActionButton,
  StatCell,
} from '@/components/TaskDetailShell';
import { CodeBlock } from '@/components/CodeBlock';
import { LogViewer } from '@/components/LogViewer';
import { MODULES } from '@/theme/modules';
import { deployTasks, logLines } from '@/mock/data';

/**
 * 部署详情。概览 Tab 给服务地址与开箱即用的调用示例（这是部署完最想拿到的东西），
 * 日志 Tab 复用 LogViewer。
 */
export function DeployDetailPage({ tab }: { tab: 'overview' | 'log' }) {
  const { id } = useParams();
  const navigate = useNavigate();
  const task = deployTasks.find((t) => t.id === id);

  const base = task?.endpoint ?? 'http://localhost:8000/v1';
  const modelName = task?.model ?? 'model';

  const curl = `curl ${base}/chat/completions \\
  -H "Content-Type: application/json" \\
  -d '{
    "model": "${modelName}",
    "messages": [{"role": "user", "content": "你好"}]
  }'`;

  const python = `from openai import OpenAI

client = OpenAI(base_url="${base}", api_key="EMPTY")
resp = client.chat.completions.create(
    model="${modelName}",
    messages=[{"role": "user", "content": "你好"}],
)
print(resp.choices[0].message.content)`;

  const running = task?.status === 'RUNNING';

  const overviewTab = (
    <>
      <div className="mb-4 flex flex-wrap gap-3">
        <StatCell label="服务状态" value={running ? '在线' : '已停止'} />
        <StatCell label="累计请求" value={running ? '1,284' : '—'} hint="示意数据，接后端后从进程指标读取" />
        <StatCell label="平均延迟" value={running ? '312' : '—'} suffix="ms" hint="示意数据" />
        <StatCell label="显存占用" value={running ? '17.4' : '—'} suffix="GB" hint="示意数据" />
      </div>

      {/* auto-fit + minmax：宽屏两段示例并排，窗口窄了自动堆成上下，不靠断点 */}
      <div className="grid gap-3.5 [grid-template-columns:repeat(auto-fit,minmax(380px,1fr))]">
        <Panel title="curl">
          <CodeBlock code={curl} />
        </Panel>
        <Panel title="Python（OpenAI 兼容）">
          <CodeBlock code={python} />
        </Panel>
      </div>

      <div className="mt-3.5">
        <Panel title="服务参数">
          <ArtifactRow
            done
            title="OpenAI 兼容接口已就绪"
            desc={`${base} · served_model_name = ${modelName}`}
          />
          <ArtifactRow
            done={running}
            title="推理后端"
            desc="vLLM · max_model_len 8192 · gpu_memory_utilization 0.9 · tp 1"
          />
          <ArtifactRow
            done={running}
            title="端口来源"
            desc="进程启动后自选空闲端口并写回 runtime.json，界面据此显示真实地址"
          />
        </Panel>
      </div>
    </>
  );

  const tabs = [
    { key: 'overview', label: '概览', content: overviewTab },
    { key: 'log', label: '日志', content: <LogViewer lines={logLines} /> },
  ];

  return (
    <TaskDetailShell
      module={MODULES.deploy}
      tasks={deployTasks}
      tabs={tabs}
      activeTab={tab}
      extraMeta={<MetaCell label="接口协议" value="OpenAI 兼容" />}
      actions={
        running ? (
          <ActionButton
            icon={<MessageSquare />}
            tone="primary"
            tooltip="带着这个 endpoint 去对话页试用"
            onClick={() => {
              toast.info('已切到对话页（示意：会自动选中该服务）');
              navigate('/chat');
            }}
          >
            试用
          </ActionButton>
        ) : null
      }
    />
  );
}
