import { useEffect, useMemo, useRef, useState } from 'react';
import { Link } from 'react-router-dom';
import { Dropdown, Modal, Tooltip, message } from 'antd';
import {
  ApiOutlined,
  CodeOutlined,
  DownOutlined,
  FileTextOutlined,
  LeftOutlined,
  PauseOutlined,
  PlayCircleFilled,
  RobotOutlined,
  WarningOutlined,
} from '@ant-design/icons';
import { NodeCanvas } from './NodeCanvas';
import { NodePalette } from './NodePalette';
import { NodeInspector } from './NodeInspector';
import type { InspectorTab } from './NodeInspector';
import { FlowAssistant } from './FlowAssistant';
import type { RunState } from './FlowAssistant';
import type { AiAction } from './aiAssist';
import { NODE_TYPES } from './nodeTypes';
import type { GraphEdge, GraphFrame, GraphNode } from './nodeTypes';
import { DEFAULT_PRESET, PRESETS, clonePreset } from './presets';
import { contentHeight } from '@/layout/metrics';
import { useSettings } from '@/settings/SettingsContext';
import { MODULES } from '@/theme/modules';
import { chrome, brand, neutral } from '@/theme/theme';

/** 图 → YAML。图形是唯一事实来源，YAML 与代码都由它生成 */
function graphToYaml(
  nodes: GraphNode[],
  edges: GraphEdge[],
  frames: GraphFrame[],
  name: string,
): string {
  const nameOf = new Map<string, string>();
  const seen: Record<string, number> = {};
  nodes.forEach((n) => {
    seen[n.type] = (seen[n.type] ?? 0) + 1;
    nameOf.set(n.id, seen[n.type] > 1 ? `${n.type}_${seen[n.type]}` : n.type);
  });

  const lines = [
    '# workflow.yaml —— 图形编辑器的序列化产物（定义，可反复运行）',
    `name: ${name}`,
  ];
  /* 循环框在图上只是视觉分区，落到 YAML 里也只能是注释 */
  frames.forEach((f) => {
    lines.push(`# ${f.label}：${f.note ?? ''}`);
  });
  lines.push('steps:');
  nodes.forEach((n) => {
    const def = NODE_TYPES[n.type];
    const needs = edges
      .filter((e) => e.to === n.id)
      .map((e) => nameOf.get(e.from))
      .filter(Boolean);
    lines.push(`  - id: ${nameOf.get(n.id)}`);
    lines.push(`    run: ${n.type}`);
    if (needs.length) lines.push(`    needs: [${needs.join(', ')}]`);
    if (def.lockedPorts) lines.push('    locked: true  # 连线不允许改，只能改参数');
    (n.params ?? def.params).forEach((p) => lines.push(`    ${p.label}: ${p.value}`));
    /* 代码节点：把片段原样带上，否则它就只剩一个类型名 */
    const code = n.code ?? def.code;
    if (code) {
      lines.push('    code: |');
      code.split('\n').forEach((l) => lines.push(`      ${l}`));
    }
  });
  return lines.join('\n');
}

function graphToCode(nodes: GraphNode[], edges: GraphEdge[], name: string): string {
  const nameOf = new Map<string, string>();
  const seen: Record<string, number> = {};
  nodes.forEach((n) => {
    seen[n.type] = (seen[n.type] ?? 0) + 1;
    nameOf.set(n.id, seen[n.type] > 1 ? `${n.type}_${seen[n.type]}` : n.type);
  });

  const out = [
    '# 由 workflow.yaml 生成（generated）',
    '# 手改本文件后，图形与代码将不再一致，UI 会给出提示',
    'from swift.workflow import Workflow',
    '',
  ];
  /* 代码节点的函数体先定义在前面，后面的装配语句才能引用 */
  nodes.forEach((n) => {
    const code = n.code ?? NODE_TYPES[n.type].code;
    if (!code) return;
    out.push(`# 代码节点 ${nameOf.get(n.id)}`, code, '');
  });
  out.push(`wf = Workflow("${name}")`);
  nodes.forEach((n) => {
    const needs = edges
      .filter((e) => e.to === n.id)
      .map((e) => nameOf.get(e.from))
      .filter(Boolean);
    const args = needs.length ? `needs=[${needs.join(', ')}]` : '';
    out.push(`${nameOf.get(n.id)} = wf.${n.type}(${args})`);
  });
  out.push('', 'if __name__ == "__main__":', '    wf.run()');
  return out.join('\n');
}

/**
 * 编排编辑器。图形 / YAML / 代码 三视图，图形是事实来源。
 *
 * 图形改动重新生成 YAML 与代码；手改代码后指纹不匹配则提示不一致，
 * 此时再动图形会弹「fork / 覆盖」二选一。不做代码 → 图形的反向解析。
 */
export function WorkflowEditorPage() {
  const mod = MODULES.workflow;
  const { settings } = useSettings();
  const [view, setView] = useState<'graph' | 'yaml' | 'code'>('graph');
  const [preset, setPreset] = useState(DEFAULT_PRESET);
  const [graph, setGraph] = useState(() => clonePreset(DEFAULT_PRESET));
  const [selected, setSelected] = useState<string | null>(null);
  const [codeDirty, setCodeDirty] = useState(false);
  const [confirmOpen, setConfirmOpen] = useState(false);
  /** 右侧检查面板：开合 + 当前 tab。tab 由节点上的入口决定 */
  const [inspectorOpen, setInspectorOpen] = useState(false);
  const [inspectorTab, setInspectorTab] = useState<InspectorTab>('params');
  const [assistantOpen, setAssistantOpen] = useState(false);
  const [runState, setRunState] = useState<RunState>('idle');
  /** 变一下就让画布重新 fit，切示例时用 */
  const [fitSignal, setFitSignal] = useState(0);
  const seq = useRef(100);
  /** 画布往里写当前视口中心，点组件面板新增的节点就落在眼前 */
  const viewCenter = useRef({ x: 380, y: 260 });
  /** 模拟运行的定时器。暂停就是把还没到点的那些取消掉 */
  const timers = useRef<number[]>([]);
  const clearTimers = () => {
    timers.current.forEach((t) => clearTimeout(t));
    timers.current = [];
  };
  useEffect(() => clearTimers, []);

  const { nodes, edges, frames } = graph;
  const selectedNode = nodes.find((n) => n.id === selected) ?? null;
  const setNodes = (fn: (ns: GraphNode[]) => GraphNode[]) =>
    setGraph((g) => ({ ...g, nodes: fn(g.nodes) }));
  const setEdges = (fn: (es: GraphEdge[]) => GraphEdge[]) =>
    setGraph((g) => ({ ...g, edges: fn(g.edges) }));

  const yaml = useMemo(
    () => graphToYaml(nodes, edges, frames, preset.label),
    [nodes, edges, frames, preset],
  );
  const code = useMemo(() => graphToCode(nodes, edges, preset.label), [nodes, edges, preset]);

  /** 切示例：整图换掉并重新 fit 视角 */
  const loadPreset = (key: string) => {
    const p = PRESETS.find((x) => x.key === key);
    if (!p) return;
    clearTimers();
    setPreset(p);
    setGraph(clonePreset(p));
    setSelected(null);
    setInspectorOpen(false);
    setRunState('idle');
    setCodeDirty(false);
    setFitSignal((s) => s + 1);
    message.success(`已载入 ${p.label} 示例：${p.desc}`);
  };

  /** 任何图形改动都要过这里：代码被手改过时先问 fork 还是覆盖 */
  const guard = (fn: () => void) => {
    if (codeDirty) {
      setConfirmOpen(true);
      return;
    }
    fn();
  };

  const addNode = (typeKey: string, x: number, y: number) =>
    guard(() => {
      const id = `n${seq.current++}`;
      setNodes((ns) => [...ns, { id, type: typeKey, x, y, status: 'idle' }]);
      setSelected(id);
    });

  const duplicate = (id: string) =>
    guard(() => {
      const src = nodes.find((n) => n.id === id);
      if (!src) return;
      const nid = `n${seq.current++}`;
      setNodes((ns) => [...ns, { ...src, id: nid, x: src.x + 34, y: src.y + 34, status: 'idle' }]);
      setSelected(nid);
      message.success('已复制节点');
    });

  const remove = (id: string) =>
    guard(() => {
      setNodes((ns) => ns.filter((n) => n.id !== id));
      setEdges((es) => es.filter((e) => e.from !== id && e.to !== id));
      setSelected(null);
    });

  const connect = (from: string, fromPort: string, to: string, toPort: string) =>
    guard(() => {
      setEdges((es) => {
        /** 一个输入口只接一条线，重连时替换旧的 */
        const kept = es.filter((e) => !(e.to === to && e.toPort === toPort));
        if (es.some((e) => e.from === from && e.fromPort === fromPort && e.to === to && e.toPort === toPort)) {
          return es;
        }
        return [...kept, { id: `e${seq.current++}`, from, fromPort, to, toPort }];
      });
    });

  /** 选中节点：同时把右侧面板开到参数页 */
  const selectNode = (id: string | null) => {
    setSelected(id);
    if (id) {
      setInspectorTab('params');
      setInspectorOpen(true);
    } else {
      setInspectorOpen(false);
    }
  };

  const openInspector = (id: string, tab: InspectorTab) => {
    setSelected(id);
    setInspectorTab(tab);
    setInspectorOpen(true);
  };

  /** 单节点运行：只跑这一个，不动上下游 */
  const runNode = (id: string) => {
    const n = nodes.find((x) => x.id === id);
    if (!n) return;
    setNodes((ns) => ns.map((x) => (x.id === id ? { ...x, status: 'running' } : x)));
    message.info(`只运行「${NODE_TYPES[n.type].label}」节点（示意，未接后端）`);
    timers.current.push(
      window.setTimeout(() => {
        setNodes((ns) => ns.map((x) => (x.id === id ? { ...x, status: 'done' } : x)));
        /* 设置里开了的话，跑完直接把输出摊开——少一次点击 */
        if (settings.autoOpenOutput) openInspector(id, 'output');
      }, 1600),
    );
  };

  /**
   * 把一批节点排进时间线。不是真拓扑排序，按节点顺序依次点亮——
   * 这里要的只是「它在往前跑」这个观感。
   */
  const schedule = (queue: GraphNode[]) => {
    if (queue.length === 0) {
      setRunState('idle');
      return;
    }
    queue.forEach((n, i) => {
      timers.current.push(
        window.setTimeout(() => {
          setNodes((ns) => ns.map((x) => (x.id === n.id ? { ...x, status: 'running' } : x)));
          timers.current.push(
            window.setTimeout(() => {
              setNodes((ns) => ns.map((x) => (x.id === n.id ? { ...x, status: 'done' } : x)));
              if (i === queue.length - 1) setRunState('idle');
            }, 900),
          );
        }, i * 700),
      );
    });
  };

  /**
   * 启停控制。AI 的控制卡片和工具条上的按钮走的是同一条路。
   *
   * 暂停把当前在跑的节点退回未运行，而不是停在一半：
   * 真实实现里暂停是直接杀 client 进程，那半步的梯度本来就丢了。
   */
  const control = (op: 'start' | 'pause' | 'resume' | 'stop') => {
    if (op === 'start') {
      clearTimers();
      setNodes((ns) => ns.map((x) => ({ ...x, status: 'idle' })));
      setRunState('running');
      message.success('已提交整图运行（示意）');
      schedule(nodes);
      return;
    }
    if (op === 'pause') {
      clearTimers();
      setNodes((ns) => ns.map((x) => (x.status === 'running' ? { ...x, status: 'idle' } : x)));
      setRunState('paused');
      message.info('已暂停。已完成的节点结果留着，正在跑的那一步丢弃');
      return;
    }
    if (op === 'resume') {
      setRunState('running');
      message.success('已继续，从没跑完的节点接上');
      schedule(nodes.filter((n) => n.status !== 'done'));
      return;
    }
    clearTimers();
    setNodes((ns) => ns.map((x) => ({ ...x, status: 'idle' })));
    setRunState('idle');
    message.warning('已停止这次运行');
  };

  /** 按下标改参数。节点本来没有 params 时，先从类型默认值实体化一份 */
  const setParamAt = (nodeId: string, index: number, value: string) =>
    guard(() =>
      setNodes((ns) =>
        ns.map((n) =>
          n.id === nodeId
            ? {
                ...n,
                params: (n.params ?? NODE_TYPES[n.type].params).map((p, i) =>
                  i === index ? { ...p, value } : p,
                ),
              }
            : n,
        ),
      ),
    );

  /** AI 提议被点了应用。注意这是唯一一条 AI 能改到图上的路径 */
  const applyAction = (a: AiAction) => {
    if (a.kind === 'control') {
      control(a.op);
      return;
    }
    if (a.kind === 'addNode') {
      addNode(a.typeKey, viewCenter.current.x - 104, viewCenter.current.y - 60);
      message.success(`已加上「${NODE_TYPES[a.typeKey]?.label ?? a.typeKey}」，还没接线`);
      return;
    }
    const target = nodes.find((n) => n.id === a.nodeId);
    if (!target) {
      message.warning('这个节点已经不在图上了');
      return;
    }
    const idx = (target.params ?? NODE_TYPES[target.type].params).findIndex(
      (p) => p.label === a.label,
    );
    if (idx < 0) {
      message.warning(`「${NODE_TYPES[target.type].label}」上没有 ${a.label} 这个参数`);
      return;
    }
    setParamAt(a.nodeId, idx, a.to);
    message.success(`${a.label} 已改为 ${a.to}`);
  };

  const views = [
    { key: 'graph', label: '图形', icon: <ApiOutlined /> },
    { key: 'yaml', label: 'YAML', icon: <FileTextOutlined /> },
    { key: 'code', label: '代码', icon: <CodeOutlined /> },
  ] as const;

  return (
    <div style={{ height: contentHeight(), display: 'flex', flexDirection: 'column' }}>
      {/* 画布应用用细工具条，不用大标题页头 */}
      <div
        style={{
          flex: 'none',
          height: 46,
          display: 'flex',
          alignItems: 'center',
          gap: 10,
          padding: '0 12px',
          borderBottom: `1px solid ${neutral.borderLight}`,
          background: '#fff',
        }}
      >
        <Link
          to={mod.path}
          style={{
            display: 'inline-flex',
            alignItems: 'center',
            gap: 4,
            fontSize: 12.5,
            color: neutral.textTertiary,
          }}
        >
          <LeftOutlined style={{ fontSize: 9 }} /> {mod.label}
        </Link>
        <span style={{ fontSize: 14, fontWeight: 500, color: neutral.text }}>{preset.label} 编排</span>

        {/* 示例切换：拆成组件后「怎么接」本身就是知识，先给几张能跑的 */ }
        <Dropdown
          trigger={['click']}
          menu={{
            selectedKeys: [preset.key],
            onClick: ({ key }) => loadPreset(key),
            items: PRESETS.map((p) => ({
              key: p.key,
              label: (
                <div style={{ padding: '2px 0' }}>
                  <div style={{ fontSize: 13 }}>{p.label}</div>
                  <div style={{ fontSize: 11.5, color: neutral.textTertiary }}>{p.desc}</div>
                </div>
              ),
            })),
          }}
        >
          <span
            className="ghost-icon"
            style={{
              display: 'inline-flex',
              alignItems: 'center',
              gap: 5,
              height: 28,
              padding: '0 10px',
              borderRadius: 8,
              fontSize: 12.5,
              color: neutral.textTertiary,
              cursor: 'pointer',
            }}
          >
            换示例 <DownOutlined style={{ fontSize: 9 }} />
          </span>
        </Dropdown>

        {/* 视图切换：自绘小胶囊 */}
        <div style={{ display: 'flex', gap: 2, marginInlineStart: 8 }}>
          {views.map((v) => {
            const active = view === v.key;
            return (
              <span
                key={v.key}
                className="ghost-icon"
                onClick={() => setView(v.key)}
                style={{
                  display: 'inline-flex',
                  alignItems: 'center',
                  gap: 5,
                  height: 28,
                  padding: '0 11px',
                  borderRadius: 8,
                  fontSize: 12.5,
                  cursor: 'pointer',
                  color: active ? neutral.text : neutral.textTertiary,
                  background: active ? neutral.bgSubtle : 'transparent',
                  fontWeight: active ? 500 : 400,
                }}
              >
                {v.icon}
                {v.label}
              </span>
            );
          })}
        </div>

        {codeDirty && (
          <Tooltip title="代码被手改过，指纹与最近一次生成结果不匹配。继续用图形编辑会弹 fork / 覆盖二选一">
            <span
              style={{
                display: 'inline-flex',
                alignItems: 'center',
                gap: 5,
                fontSize: 12,
                color: '#8A6320',
                background: '#FDF8EC',
                padding: '4px 10px',
                borderRadius: 999,
                cursor: 'default',
              }}
            >
              <WarningOutlined /> 图形与代码不一致
            </span>
          </Tooltip>
        )}

        <div style={{ marginInlineStart: 'auto', display: 'flex', alignItems: 'center', gap: 8 }}>
          <span style={{ fontSize: 12, color: neutral.textTertiary }}>
            {nodes.length} 节点 · {edges.length} 连线
          </span>
          {settings.aiEnabled && (
            <Tooltip title="问整张图：启停、调参、排查">
              <span
                className="ghost-icon"
                onClick={() => setAssistantOpen(true)}
                style={{
                  display: 'inline-flex',
                  alignItems: 'center',
                  gap: 5,
                  height: 30,
                  padding: '0 12px',
                  borderRadius: 999,
                  fontSize: 12.5,
                  cursor: 'pointer',
                  background: assistantOpen ? brand.soft : neutral.bgSubtle,
                  color: assistantOpen ? brand.primaryActive : neutral.textSecondary,
                }}
              >
                <RobotOutlined /> 流程助手
              </span>
            </Tooltip>
          )}
          {runState === 'running' ? (
            <span className="pill-btn" onClick={() => control('pause')} style={runAllStyle}>
              <PauseOutlined /> 暂停
            </span>
          ) : (
            <span
              className="pill-btn"
              onClick={() => control(runState === 'paused' ? 'resume' : 'start')}
              style={runAllStyle}
            >
              <PlayCircleFilled /> {runState === 'paused' ? '继续' : '运行全部'}
            </span>
          )}
        </div>
      </div>

      {view === 'graph' ? (
        <div style={{ flex: 1, display: 'flex', minHeight: 0 }}>
          <NodePalette
            onAdd={(k) =>
              addNode(k, viewCenter.current.x - 104, viewCenter.current.y - 60)
            }
          />
          <NodeCanvas
            nodes={nodes}
            edges={edges}
            frames={frames}
            selected={selected}
            fitSignal={fitSignal}
            viewCenterRef={viewCenter}
            aiEnabled={settings.aiEnabled}
            onSelect={selectNode}
            onMoveNode={(id, x, y) =>
              setNodes((ns) => ns.map((n) => (n.id === id ? { ...n, x, y } : n)))
            }
            onAddNode={addNode}
            onConnect={connect}
            onRunNode={runNode}
            onOpenOutput={(id) => openInspector(id, 'output')}
            onAskAi={(id) => openInspector(id, 'ai')}
            onDuplicate={duplicate}
            onDelete={remove}
          />
          {inspectorOpen && selectedNode && (
            <NodeInspector
              node={selectedNode}
              nodes={nodes}
              edges={edges}
              presetLabel={preset.label}
              running={runState === 'running'}
              aiEnabled={settings.aiEnabled}
              confirmBeforeApply={settings.aiConfirmBeforeApply}
              defaultTab={inspectorTab}
              onClose={() => setInspectorOpen(false)}
              onRun={runNode}
              onParamChange={setParamAt}
              onAction={applyAction}
            />
          )}
        </div>
      ) : (
        <div style={{ flex: 1, overflow: 'auto', padding: 20, background: '#fff' }}>
          {view === 'code' && (
            <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 10 }}>
              <span
                className="ghost-icon"
                onClick={() => {
                  setCodeDirty(true);
                  message.info('已模拟手改代码，回到图形编辑试试');
                }}
                style={{
                  height: 30,
                  padding: '0 12px',
                  borderRadius: 999,
                  display: 'inline-flex',
                  alignItems: 'center',
                  fontSize: 12.5,
                  cursor: 'pointer',
                  background: neutral.bgSubtle,
                  color: neutral.textSecondary,
                }}
              >
                模拟手改代码
              </span>
              <span style={{ fontSize: 12, color: neutral.textTertiary }}>
                手改后再动图形，会触发一致性确认
              </span>
            </div>
          )}
          <pre
            style={{
              margin: 0,
              background: neutral.bgCode,
              borderRadius: 12,
              padding: 16,
              fontFamily: 'ui-monospace, SFMono-Regular, Menlo, monospace',
              fontSize: 12.5,
              lineHeight: '20px',
              color: neutral.text,
              overflow: 'auto',
            }}
          >
            {view === 'yaml' ? yaml : code}
          </pre>
          <div style={{ fontSize: 12, color: neutral.textTertiary, marginTop: 10 }}>
            这份内容由当前图形实时生成——去图形里加个节点或连根线再回来看。
          </div>
        </div>
      )}

      <FlowAssistant
        open={assistantOpen}
        onClose={() => setAssistantOpen(false)}
        nodes={nodes}
        edges={edges}
        presetLabel={preset.label}
        runState={runState}
        confirmBeforeApply={settings.aiConfirmBeforeApply}
        onControl={control}
        onAction={applyAction}
      />

      <Modal
        open={confirmOpen}
        title="图形与代码不一致"
        okText="fork 成新任务"
        cancelText="覆盖手改"
        onOk={() => {
          setConfirmOpen(false);
          setCodeDirty(false);
          message.success('已 fork 成新任务，保留手改版本（记录 forked_from）');
        }}
        onCancel={() => {
          setConfirmOpen(false);
          setCodeDirty(false);
          message.warning('已用图形生成的代码覆盖手改内容');
        }}
      >
        代码被手动改过。继续用图形编辑，你希望如何处理手改内容？
        <ul style={{ marginTop: 10, color: neutral.textSecondary }}>
          <li>fork 成新任务：复制出新任务（新 id，记 forked_from），手改版本原样保留</li>
          <li>覆盖手改：丢弃手改，以图形重新生成的代码为准</li>
        </ul>
      </Modal>
    </div>
  );
}

const runAllStyle: React.CSSProperties = {
  display: 'inline-flex',
  alignItems: 'center',
  gap: 6,
  height: 32,
  padding: '0 14px',
  borderRadius: 999,
  fontSize: 13,
  fontWeight: 500,
  cursor: 'pointer',
  background: chrome.titlebar,
  color: '#fff',
};
