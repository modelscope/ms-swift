import { useParams } from 'react-router-dom';
import { TaskDetailShell, MetaCell, Panel, StatCell } from '@/components/TaskDetailShell';
import { LogViewer } from '@/components/LogViewer';
import { LineChart } from '@/components/charts/LineChart';
import { MODULES } from '@/theme/modules';
import { logo, neutral } from '@/theme/theme';
import { logLines, metricPoints, trainTasks } from '@/mock/data';

/**
 * 训练详情。两个 Tab：指标 / 日志，用 URL 里的 :tab 决定当前停在哪个。
 * 曲线数据来自 metrics.jsonl，日志来自 output.log，都是设计里定好的产物。
 */
export function TrainDetailPage({ tab }: { tab: 'metrics' | 'log' }) {
  const { id } = useParams();
  const task = trainTasks.find((t) => t.id === id);
  const last = metricPoints.at(-1);

  const pct = task?.totalSteps ? Math.round(((task.step ?? 0) / task.totalSteps) * 100) : 0;

  const metricsTab = (
    <>
      {/* 四枚指标：软底方块，不套 Card */}
      <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap', marginBottom: 14 }}>
        <StatCell label="当前 step" value={task?.step ?? 0} suffix={`/ ${task?.totalSteps ?? 0}`} />
        <StatCell label="最新 loss" value={last?.loss.toFixed(4) ?? '—'} />
        <StatCell label="学习率" value={last?.lr.toExponential(2) ?? '—'} />
        <StatCell label="grad norm" value={last?.gradNorm.toFixed(3) ?? '—'} />
      </div>

      {/* 进度条：细一条，不用 antd Progress 的默认样式 */}
      {task?.totalSteps ? (
        <div style={{ marginBottom: 20 }}>
          <div
            style={{
              display: 'flex',
              justifyContent: 'space-between',
              fontSize: 12.5,
              color: neutral.textTertiary,
              marginBottom: 6,
            }}
          >
            <span>训练进度</span>
            <span>{pct}%</span>
          </div>
          <div style={{ height: 4, borderRadius: 999, background: neutral.borderLight, overflow: 'hidden' }}>
            <div
              style={{
                width: `${pct}%`,
                height: '100%',
                borderRadius: 999,
                background: logo.indigo,
                transition: 'width 0.3s ease',
              }}
            />
          </div>
        </div>
      ) : null}

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(360px, 1fr))', gap: 14 }}>
        <Panel title="loss">
          <LineChart
            points={metricPoints.map((p) => ({ x: p.step, y: p.loss }))}
            color={logo.indigo}
            yLabel="loss"
          />
        </Panel>
        <Panel title="learning rate">
          <LineChart
            points={metricPoints.map((p) => ({ x: p.step, y: p.lr }))}
            color={logo.blue}
            yLabel="lr"
          />
        </Panel>
      </div>

      <div style={{ marginTop: 14, fontSize: 12.5, color: neutral.textTertiary }}>
        曲线按 metrics.jsonl 增量刷新；若任务经历过暂停/继续，横轴会在续跑处出现一次接续。
      </div>
    </>
  );

  const tabs = [
    { key: 'metrics', label: '指标', content: metricsTab },
    { key: 'log', label: '日志', content: <LogViewer lines={logLines} /> },
  ];

  return (
    <TaskDetailShell
      module={MODULES.train}
      tasks={trainTasks}
      tabs={tabs}
      activeTab={tab}
      extraMeta={
        task?.totalSteps ? (
          <MetaCell label="进度" value={`${task.step ?? 0} / ${task.totalSteps} step · ${pct}%`} />
        ) : null
      }
    />
  );
}
