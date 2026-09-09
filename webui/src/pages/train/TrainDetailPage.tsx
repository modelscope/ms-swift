import { useParams } from 'react-router-dom';
import { TaskDetailShell, MetaCell, Panel, StatCell } from '@/components/TaskDetailShell';
import { LogViewer } from '@/components/LogViewer';
import { LineChart } from '@/components/charts/LineChart';
import { MODULES } from '@/theme/modules';
import { logo } from '@/theme/theme';
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
      <div className="mb-3.5 flex flex-wrap gap-3">
        <StatCell label="当前 step" value={task?.step ?? 0} suffix={`/ ${task?.totalSteps ?? 0}`} />
        <StatCell label="最新 loss" value={last?.loss.toFixed(4) ?? '—'} />
        <StatCell label="学习率" value={last?.lr.toExponential(2) ?? '—'} />
        <StatCell label="grad norm" value={last?.gradNorm.toFixed(3) ?? '—'} />
      </div>

      {/* 进度条：细一条，不用现成的进度条组件——它只需要是一条线 */}
      {task?.totalSteps ? (
        <div className="mb-5">
          <div className="text-muted-foreground mb-1.5 flex justify-between text-[12.5px]">
            <span>训练进度</span>
            <span>{pct}%</span>
          </div>
          <div className="bg-border/60 h-1 overflow-hidden rounded-full">
            {/* 用 logo 的靛色而不是 --primary：跟下面那条 loss 曲线是同一件事 */}
            <div
              className="h-full rounded-full transition-[width] duration-300"
              style={{ width: `${pct}%`, background: logo.indigo }}
            />
          </div>
        </div>
      ) : null}

      <div className="grid grid-cols-[repeat(auto-fit,minmax(360px,1fr))] gap-3.5">
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

      <div className="text-muted-foreground mt-3.5 text-[12.5px]">
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
