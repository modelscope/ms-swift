import { useEffect, useMemo, useRef, useState } from 'react';
import { Link } from 'react-router-dom';
import { toast } from 'sonner';
import { Bot, ChevronDown, ChevronLeft, Code, FileText, Pause, Play, TriangleAlert, Workflow } from 'lucide-react';
import { NodeCanvas } from './NodeCanvas';
import { NodePalette } from './NodePalette';
import { NodeInspector } from './NodeInspector';
import type { InspectorTab } from './NodeInspector';
import { FlowAssistant } from './FlowAssistant';
import type { RunState } from './FlowAssistant';
import type { AiAction } from './aiAssist';
import { NODE_TYPES, NODE_W } from './nodeTypes';
import type { GraphEdge, GraphFrame, GraphNode } from './nodeTypes';
import { DEFAULT_PRESET, PRESETS, clonePreset } from './presets';
import { CodeBlock } from '@/components/CodeBlock';
import { Hint } from '@/components/Hint';
import { SegmentedControl } from '@/components/FormField';
import { Button } from '@/components/ui/button';
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from '@/components/ui/alert-dialog';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuRadioGroup,
  DropdownMenuRadioItem,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu';
import { contentHeight } from '@/layout/metrics';
import { useSettings } from '@/settings/SettingsContext';
import { MODULES } from '@/theme/modules';

/**
 * 给每个节点起一个能在 YAML 和代码里被引用的名字。
 * 同类型出现多次才加序号——只有一个 loss 时就叫 loss，不叫 loss_1。
 */
function stepNames(nodes: GraphNode[]): Map<string, string> {
  const seen: Record<string, number> = {};
  return new Map(
    nodes.map((n) => {
      seen[n.type] = (seen[n.type] ?? 0) + 1;
      return [n.id, seen[n.type] > 1 ? `${n.type}_${seen[n.type]}` : n.type];
    }),
  );
}

/** 某个节点的上游名字。YAML 的 needs: 和代码里的 needs= 用的是同一份 */
function upstream(edges: GraphEdge[], nameOf: Map<string, string>, id: string): string[] {
  return edges
    .filter((e) => e.to === id)
    .map((e) => nameOf.get(e.from))
    .filter((n): n is string => !!n);
}

/** 图 → YAML。图形是唯一事实来源，YAML 与代码都由它生成 */
function graphToYaml(
  nodes: GraphNode[],
  edges: GraphEdge[],
  frames: GraphFrame[],
  name: string,
): string {
  const nameOf = stepNames(nodes);
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
    const needs = upstream(edges, nameOf, n.id);
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
  const nameOf = stepNames(nodes);
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
    const needs = upstream(edges, nameOf, n.id);
    const args = needs.length ? `needs=[${needs.join(', ')}]` : '';
    out.push(`${nameOf.get(n.id)} = wf.${n.type}(${args})`);
  });
  out.push('', 'if __name__ == "__main__":', '    wf.run()');
  return out.join('\n');
}

type ViewKey = 'graph' | 'yaml' | 'code';

const VIEWS = [
  { value: 'graph', label: <><Workflow className="size-3.5" />图形</> },
  { value: 'yaml', label: <><FileText className="size-3.5" />YAML</> },
  { value: 'code', label: <><Code className="size-3.5" />代码</> },
];

/**
 * 编排编辑器。图形 / YAML / 代码 三视图，图形是事实来源。
 *
 * 图形改动重新生成 YAML 与代码；手改代码后指纹不匹配则提示不一致，
 * 此时再动图形会弹「fork / 覆盖」二选一。不做代码 → 图形的反向解析。
 */
export function WorkflowEditorPage() {
  const mod = MODULES.workflow;
  const { settings } = useSettings();
  const [view, setView] = useState<ViewKey>('graph');
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
    toast.success(`已载入 ${p.label} 示例`, { description: p.desc });
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

  /**
   * 加节点到当前视野中心。组件面板和 AI 的「加个节点」都走这里。
   * 减掉半个节点的宽高是因为落点是左上角——不减的话新节点会偏在视野的右下角。
   */
  const addAtViewCenter = (typeKey: string) =>
    addNode(typeKey, viewCenter.current.x - NODE_W / 2, viewCenter.current.y - 60);

  const duplicate = (id: string) =>
    guard(() => {
      const src = nodes.find((n) => n.id === id);
      if (!src) return;
      const nid = `n${seq.current++}`;
      setNodes((ns) => [...ns, { ...src, id: nid, x: src.x + 34, y: src.y + 34, status: 'idle' }]);
      setSelected(nid);
      toast.success('已复制节点');
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

  /**
   * 选中节点：把右侧面板开出来。
   *
   * 只有真的换了节点才回到参数页。无条件重置会把 openInspector 刚指定的那一页
   * 顶掉——点节点上的「问 AI」本来要直接开到 AI 页，被顶一下就只能看到参数页。
   *
   * 没选中节点时面板自然就不渲染了（靠 selectedNode 为空卡住），
   * 不需要再额外写一次 inspectorOpen。
   */
  const selectNode = (id: string | null) => {
    setSelected(id);
    if (!id) return;
    if (id !== selected) setInspectorTab('params');
    setInspectorOpen(true);
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
    toast.info(`只运行「${NODE_TYPES[n.type].label}」节点（示意，未接后端）`);
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
      toast.success('已提交整图运行（示意）');
      schedule(nodes);
      return;
    }
    if (op === 'pause') {
      clearTimers();
      setNodes((ns) => ns.map((x) => (x.status === 'running' ? { ...x, status: 'idle' } : x)));
      setRunState('paused');
      toast.info('已暂停。已完成的节点结果留着，正在跑的那一步丢弃');
      return;
    }
    if (op === 'resume') {
      setRunState('running');
      toast.success('已继续，从没跑完的节点接上');
      schedule(nodes.filter((n) => n.status !== 'done'));
      return;
    }
    clearTimers();
    setNodes((ns) => ns.map((x) => ({ ...x, status: 'idle' })));
    setRunState('idle');
    toast.warning('已停止这次运行');
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
      addAtViewCenter(a.typeKey);
      toast.success(`已加上「${NODE_TYPES[a.typeKey]?.label ?? a.typeKey}」，还没接线`);
      return;
    }
    const target = nodes.find((n) => n.id === a.nodeId);
    if (!target) {
      toast.warning('这个节点已经不在图上了');
      return;
    }
    const idx = (target.params ?? NODE_TYPES[target.type].params).findIndex(
      (p) => p.label === a.label,
    );
    if (idx < 0) {
      toast.warning(`「${NODE_TYPES[target.type].label}」上没有 ${a.label} 这个参数`);
      return;
    }
    setParamAt(a.nodeId, idx, a.to);
    toast.success(`${a.label} 已改为 ${a.to}`);
  };

  return (
    <div className="flex flex-col" style={{ height: contentHeight() }}>
      {/* 画布应用用细工具条，不用大标题页头 */}
      <div className="bg-background border-border/60 flex h-[46px] flex-none items-center gap-2.5 border-b px-3">
        <Link
          to={mod.path}
          className="text-muted-foreground hover:text-foreground inline-flex items-center gap-1 text-xs transition-colors"
        >
          <ChevronLeft className="size-3" />
          {mod.label}
        </Link>
        <span className="text-foreground text-sm font-medium">{preset.label} 编排</span>

        {/* 示例切换：拆成组件后「怎么接」本身就是知识，先给几张能跑的 */}
        <DropdownMenu>
          <DropdownMenuTrigger asChild>
            <Button variant="ghost" size="sm" className="text-muted-foreground gap-1.5">
              换示例
              <ChevronDown className="size-3" />
            </Button>
          </DropdownMenuTrigger>
          {/* 用单选组而不是普通菜单项：这是「当前在看哪张图」，得有个选中标记 */}
          <DropdownMenuContent align="start" className="w-64">
            <DropdownMenuRadioGroup value={preset.key} onValueChange={loadPreset}>
              {PRESETS.map((p) => (
                <DropdownMenuRadioItem key={p.key} value={p.key} className="items-start py-1.5">
                  <span className="flex flex-col gap-0.5">
                    <span className="text-[13px]">{p.label}</span>
                    <span className="text-muted-foreground text-[11.5px]">{p.desc}</span>
                  </span>
                </DropdownMenuRadioItem>
              ))}
            </DropdownMenuRadioGroup>
          </DropdownMenuContent>
        </DropdownMenu>

        <SegmentedControl value={view} onChange={(v) => setView(v as ViewKey)} options={VIEWS} />

        {codeDirty && (
          <Hint title="代码被手改过，指纹与最近一次生成结果不匹配。继续用图形编辑会弹 fork / 覆盖二选一">
            {/*
              警示色直接用 amber 而不是主题色：这是「有件事没对齐」的通用信号。
              底色用半透明的同色，这样深浅两套皮肤下都不用各配一个值。
            */}
            <span className="inline-flex cursor-default items-center gap-1.5 rounded-full border border-amber-500/30 bg-amber-500/10 px-2.5 py-1 text-xs text-amber-700 dark:text-amber-400">
              <TriangleAlert className="size-3" /> 图形与代码不一致
            </span>
          </Hint>
        )}

        <div className="ms-auto flex items-center gap-2">
          <span className="text-muted-foreground text-xs">
            {nodes.length} 节点 · {edges.length} 连线
          </span>
          {settings.aiEnabled && (
            <Hint title="问整张图：启停、调参、排查">
              <Button
                variant={assistantOpen ? 'secondary' : 'ghost'}
                size="sm"
                className="rounded-full"
                onClick={() => setAssistantOpen(true)}
              >
                <Bot /> 流程助手
              </Button>
            </Hint>
          )}
          {runState === 'running' ? (
            <Button size="sm" className="rounded-full" onClick={() => control('pause')}>
              <Pause /> 暂停
            </Button>
          ) : (
            <Button
              size="sm"
              className="rounded-full"
              onClick={() => control(runState === 'paused' ? 'resume' : 'start')}
            >
              <Play /> {runState === 'paused' ? '继续' : '运行全部'}
            </Button>
          )}
        </div>
      </div>

      {view === 'graph' ? (
        <div className="flex min-h-0 flex-1">
          <NodePalette onAdd={addAtViewCenter} />
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
            onDeleteEdges={(ids) =>
              guard(() => setEdges((es) => es.filter((e) => !ids.includes(e.id))))
            }
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
              tab={inspectorTab}
              onClose={() => setInspectorOpen(false)}
              onRun={runNode}
              onParamChange={setParamAt}
              onAction={applyAction}
              onTabChange={setInspectorTab}
            />
          )}
        </div>
      ) : (
        <div className="bg-background flex-1 overflow-auto p-5">
          {view === 'code' && (
            <div className="mb-2.5 flex items-center gap-2.5">
              <Button
                variant="secondary"
                size="sm"
                className="rounded-full"
                onClick={() => {
                  setCodeDirty(true);
                  toast.info('已模拟手改代码，回到图形编辑试试');
                }}
              >
                模拟手改代码
              </Button>
              <span className="text-muted-foreground text-xs">
                手改后再动图形，会触发一致性确认
              </span>
            </div>
          )}
          {/* 换成 CodeBlock 之后顺带有了复制——这两份内容本来就是要拿去用的 */}
          <CodeBlock code={view === 'yaml' ? yaml : code} label={view === 'yaml' ? 'YAML' : '代码'} />
          <div className="text-muted-foreground mt-2.5 text-xs">
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

      <AlertDialog open={confirmOpen} onOpenChange={setConfirmOpen}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>图形与代码不一致</AlertDialogTitle>
            <AlertDialogDescription>
              代码被手动改过。继续用图形编辑，你希望如何处理手改内容？
            </AlertDialogDescription>
          </AlertDialogHeader>
          <ul className="text-muted-foreground list-disc space-y-1 ps-5 text-[13px]">
            <li>fork 成新任务：复制出新任务（新 id，记 forked_from），手改版本原样保留</li>
            <li>覆盖手改：丢弃手改，以图形重新生成的代码为准</li>
          </ul>
          <AlertDialogFooter>
            {/*
              「先不动」得自己占一个按钮：原来「覆盖手改」挂在弹窗的取消位上，
              于是按 Esc 或点遮罩关掉弹窗，手改的代码就被静默丢掉了。
              会丢东西的动作不能是关窗口的副作用。
            */}
            <AlertDialogCancel>先不动</AlertDialogCancel>
            <AlertDialogAction
              variant="destructive"
              onClick={() => {
                setCodeDirty(false);
                toast.warning('已用图形生成的代码覆盖手改内容');
              }}
            >
              覆盖手改
            </AlertDialogAction>
            <AlertDialogAction
              onClick={() => {
                setCodeDirty(false);
                toast.success('已 fork 成新任务，保留手改版本（记录 forked_from）');
              }}
            >
              fork 成新任务
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </div>
  );
}
