import { TaskDetailShell, Panel } from '@/components/TaskDetailShell';
import { LogViewer } from '@/components/LogViewer';
import { RadarChart } from '@/components/charts/RadarChart';
import { MODULES } from '@/theme/modules';
import { logo } from '@/theme/theme';
import { evalDatasets, evalResults, evalTasks, logLines } from '@/mock/data';

/**
 * 评测详情。结果 Tab 用雷达图并排对比多个模型（对应「多结果对比」需求），
 * 右侧给一张分数表——自绘网格，只用一行小灰字当表头，没有底色和竖线。
 */
export function EvalDetailPage({ tab }: { tab: 'result' | 'log' }) {
  const palette = [logo.indigo, logo.blue, logo.plum];
  const axes = evalDatasets.filter((d) => d in evalResults[0].scores);

  /** 每行找出最高分，标出来——表格只是罗列，标出胜者才有信息量 */
  const bestOf = (d: string) => Math.max(...evalResults.map((r) => r.scores[d] ?? -Infinity));

  /* 表头和每一行必须用同一份列宽，否则分数会跟模型名错位 */
  const cols = { gridTemplateColumns: `120px repeat(${evalResults.length}, 1fr)` };

  const scoreTable = (
    <div className="text-[13px]">
      <div className="text-muted-foreground grid gap-2 pb-2 text-xs" style={cols}>
        <span>数据集</span>
        {evalResults.map((r, i) => (
          <span key={r.name} className="inline-flex min-w-0 items-center gap-1.5">
            <Swatch color={palette[i % palette.length]} />
            <span className="truncate">{r.name}</span>
          </span>
        ))}
      </div>

      {axes.map((d) => {
        const best = bestOf(d);
        return (
          <div key={d} className="border-border/60 grid items-center gap-2 border-t py-2.5" style={cols}>
            <span className="text-foreground/80">{d}</span>
            {evalResults.map((r) => {
              const v = r.scores[d];
              /* tabular-nums：等宽数字，几列分数才能按小数点对齐着比 */
              return (
                <span
                  key={r.name}
                  className={
                    v === best ? 'text-foreground font-semibold tabular-nums' : 'text-foreground/80 tabular-nums'
                  }
                >
                  {v ?? '—'}
                </span>
              );
            })}
          </div>
        );
      })}
    </div>
  );

  const resultTab = (
    <div className="grid grid-cols-[repeat(auto-fit,minmax(380px,1fr))] items-start gap-3.5">
      <Panel title="能力雷达对比">
        <RadarChart
          axes={axes}
          series={evalResults.map((r, i) => ({
            name: r.name,
            values: axes.map((d) => r.scores[d]),
            color: palette[i % palette.length],
          }))}
        />
        <div className="mt-2 flex flex-wrap justify-center gap-4">
          {evalResults.map((r, i) => (
            <span key={r.name} className="text-foreground/80 inline-flex items-center gap-1.5 text-[12.5px]">
              <Swatch color={palette[i % palette.length]} />
              {r.name}
            </span>
          ))}
        </div>
      </Panel>

      <Panel
        title="分数明细"
        extra={<span className="text-muted-foreground text-xs">加粗为该项最高</span>}
      >
        {scoreTable}
      </Panel>
    </div>
  );

  const tabs = [
    { key: 'result', label: '结果', content: resultTab },
    { key: 'log', label: '日志', content: <LogViewer lines={logLines} /> },
  ];

  return <TaskDetailShell module={MODULES.eval} tasks={evalTasks} tabs={tabs} activeTab={tab} />;
}

/**
 * 系列色块。表头和图例下面各要一个，本来是两份尺寸略不同的 inline style
 * （7px 和 9px），统一成一个尺寸——那点差别没人看得出来，两份代码却要各改一次。
 */
function Swatch({ color }: { color: string }) {
  return <span className="size-2 flex-none rounded-[2px]" style={{ background: color }} />;
}
