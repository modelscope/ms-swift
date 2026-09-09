import { useParams } from 'react-router-dom';
import { useNavigate } from 'react-router-dom';
import { Tooltip, Typography, message } from 'antd';
import { MessageOutlined } from '@ant-design/icons';
import {
  TaskDetailShell,
  ArtifactRow,
  MetaCell,
  Panel,
  PillButton,
  StatCell,
} from '@/components/TaskDetailShell';
import { LogViewer } from '@/components/LogViewer';
import { MODULES } from '@/theme/modules';
import { brand, neutral } from '@/theme/theme';
import { deployTasks, logLines } from '@/mock/data';

/** 可复制的代码块。等宽字体 + 软底，右上角悬浮复制 */
function CodeBlock({ code }: { code: string }) {
  return (
    <div style={{ position: 'relative' }}>
      <pre
        style={{
          margin: 0,
          padding: '12px 14px',
          background: neutral.bgCode,
          borderRadius: 11,
          fontSize: 12.5,
          lineHeight: '20px',
          color: neutral.text,
          fontFamily: 'ui-monospace, SFMono-Regular, Menlo, monospace',
          overflowX: 'auto',
          whiteSpace: 'pre',
        }}
      >
        {code}
      </pre>
      <span style={{ position: 'absolute', top: 7, right: 9 }}>
        <Typography.Text copyable={{ text: code, tooltips: ['复制', '已复制'] }} />
      </span>
    </div>
  );
}

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
      <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap', marginBottom: 16 }}>
        <StatCell label="服务状态" value={running ? '在线' : '已停止'} />
        <StatCell label="累计请求" value={running ? '1,284' : '—'} hint="示意数据，接后端后从进程指标读取" />
        <StatCell label="平均延迟" value={running ? '312' : '—'} suffix="ms" hint="示意数据" />
        <StatCell label="显存占用" value={running ? '17.4' : '—'} suffix="GB" hint="示意数据" />
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(380px, 1fr))', gap: 14 }}>
        <Panel title="curl">
          <CodeBlock code={curl} />
        </Panel>
        <Panel title="Python（OpenAI 兼容）">
          <CodeBlock code={python} />
        </Panel>
      </div>

      <div style={{ marginTop: 14 }}>
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
          <Tooltip title="带着这个 endpoint 去对话页试用">
            <PillButton
              icon={<MessageOutlined />}
              onClick={() => {
                message.info('已切到对话页（示意：会自动选中该服务）');
                navigate('/chat');
              }}
            >
              <span style={{ color: brand.primary }}>试用</span>
            </PillButton>
          </Tooltip>
        ) : null
      }
    />
  );
}
