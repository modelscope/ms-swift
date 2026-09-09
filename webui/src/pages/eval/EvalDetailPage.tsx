import { TaskDetailShell, Panel } from '@/components/TaskDetailShell';
import { LogViewer } from '@/components/LogViewer';
import { RadarChart } from '@/components/charts/RadarChart';
import { MODULES } from '@/theme/modules';
import { logo, neutral } from '@/theme/theme';
import { evalDatasets, evalResults, evalTasks, logLines } from '@/mock/data';

/**
 * 评测详情。结果 Tab 用雷达图并排对比多个模型（对应「多结果对比」需求），
 * 右侧给一张分数表——自绘网格，不用 antd Table 的表头与竖线。
 */
export function EvalDetailPage({ tab }: { tab: 'result' | 'log' }) {
  const palette = [logo.indigo, logo.blue, logo.plum];
  const axes = evalDatasets.filter((d) => d in evalResults[0].scores);

  /** 每行找出最高分，标出来——表格只是罗列，标出胜者才有信息量 */
  const bestOf = (d: string) => Math.max(...evalResults.map((r) => r.scores[d] ?? -Infinity));

  const scoreTable = (
    <div style={{ fontSize: 13 }}>
      {/* 表头：只有一行小灰字，没有底色和竖线 */}
      <div
        style={{
          display: 'grid',
          gridTemplateColumns: `120px repeat(${evalResults.length}, 1fr)`,
          gap: 8,
          padding: '0 0 8px',
          fontSize: 12,
          color: neutral.textTertiary,
        }}
      >
        <span>数据集</span>
        {evalResults.map((r, i) => (
          <span key={r.name} style={{ display: 'inline-flex', alignItems: 'center', gap: 5 }}>
            <span
              style={{
                width: 7,
                height: 7,
                borderRadius: 2,
                background: palette[i % palette.length],
                flex: 'none',
              }}
            />
            <span style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
              {r.name}
            </span>
          </span>
        ))}
      </div>

      {axes.map((d) => {
        const best = bestOf(d);
        return (
          <div
            key={d}
            style={{
              display: 'grid',
              gridTemplateColumns: `120px repeat(${evalResults.length}, 1fr)`,
              gap: 8,
              padding: '9px 0',
              borderTop: `1px solid ${neutral.borderLight}`,
              alignItems: 'center',
            }}
          >
            <span style={{ color: neutral.textSecondary }}>{d}</span>
            {evalResults.map((r) => {
              const v = r.scores[d];
              const win = v === best;
              return (
                <span
                  key={r.name}
                  style={{
                    fontVariantNumeric: 'tabular-nums',
                    color: win ? neutral.text : neutral.textSecondary,
                    fontWeight: win ? 600 : 400,
                  }}
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
    <div
      style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(auto-fit, minmax(380px, 1fr))',
        gap: 14,
        alignItems: 'start',
      }}
    >
      <Panel title="能力雷达对比">
        <RadarChart
          axes={axes}
          series={evalResults.map((r, i) => ({
            name: r.name,
            values: axes.map((d) => r.scores[d]),
            color: palette[i % palette.length],
          }))}
        />
        <div
          style={{
            display: 'flex',
            gap: 16,
            justifyContent: 'center',
            flexWrap: 'wrap',
            marginTop: 8,
          }}
        >
          {evalResults.map((r, i) => (
            <span
              key={r.name}
              style={{ display: 'inline-flex', alignItems: 'center', gap: 6, fontSize: 12.5, color: neutral.textSecondary }}
            >
              <span
                style={{
                  width: 9,
                  height: 9,
                  borderRadius: 2,
                  background: palette[i % palette.length],
                  display: 'inline-block',
                }}
              />
              {r.name}
            </span>
          ))}
        </div>
      </Panel>

      <Panel title="分数明细" extra={<span style={{ fontSize: 12, color: neutral.textTertiary }}>加粗为该项最高</span>}>
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
