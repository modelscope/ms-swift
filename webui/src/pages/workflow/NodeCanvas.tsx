import { createContext, memo, useCallback, useContext, useEffect, useMemo, useRef } from 'react';
import type { CSSProperties } from 'react';
import {
  Background,
  BackgroundVariant,
  ControlButton,
  Controls,
  Handle,
  Panel,
  Position,
  ReactFlow,
  ReactFlowProvider,
  useConnection,
  useReactFlow,
} from '@xyflow/react';
import type {
  Connection,
  ConnectionState,
  Edge,
  FitViewOptions,
  Node,
  NodeProps,
  OnNodesChange,
} from '@xyflow/react';
import '@xyflow/react/dist/style.css';
import {
  Bot,
  Check,
  Copy,
  FileText,
  Loader2,
  Lock,
  Maximize2,
  Minus,
  Play,
  Plus,
  TriangleAlert,
  X,
} from 'lucide-react';
import { toast } from 'sonner';
import { cn } from 'cn';
import { Hint } from '@/components/Hint';
import { NODE_TYPES, NODE_W, PORT_TYPES, findPort } from './nodeTypes';
import type { GraphEdge, GraphFrame, GraphNode, PortDef } from './nodeTypes';

/**
 * 节点画布。深底 + 点阵网格，节点是厚边框的卡片，连线是粗贝塞尔曲线。
 *
 * 平移、缩放、拖节点、拉连线、框选、适应视图这些全部交给 React Flow，
 * 这个文件只负责两件事：把编排的「作者格式」翻译成 React Flow 的结构，
 * 以及画节点长什么样。
 *
 * 翻译只发生在这里一处 —— 示例图、YAML/代码生成、AI 上下文、输出模拟
 * 读到的都还是 GraphNode / GraphEdge，它们不认识 React Flow。
 */

/** 节点里塞的就是原始的作者格式，画节点时直接拿出来用 */
type SwiftNodeData = { node: GraphNode };
type FrameNodeData = { frame: GraphFrame };

/**
 * 单节点上的操作走 context 而不是塞进 node.data。
 *
 * 塞进 data 的话，父组件每次重渲染这几个回调的引用都变，
 * 于是每个节点的 data 都成了新对象，整张图跟着重建。
 */
interface NodeActions {
  onRunNode: (id: string) => void;
  onOpenOutput?: (id: string) => void;
  onAskAi?: (id: string) => void;
  onDuplicate: (id: string) => void;
  onDelete: (id: string) => void;
}
const ActionsCtx = createContext<NodeActions | null>(null);

/** 切示例后的复位视角：留出边距但不缩太小，从数据流起点开始看 */
const FIT_ON_LOAD: FitViewOptions = { padding: 0.1, minZoom: 0.72, maxZoom: 1 };
/** 手点「适应」时才真缩到装下全图——GRPO 那张图展开有两千多像素宽 */
const FIT_ALL: FitViewOptions = { padding: 0.08, minZoom: 0.3, maxZoom: 1 };

export interface NodeCanvasProps {
  nodes: GraphNode[];
  edges: GraphEdge[];
  /** 循环容器。画在最底层，只标出“这一块每步重跑”，不参与交互 */
  frames?: GraphFrame[];
  selected: string | null;
  onSelect: (id: string | null) => void;
  onMoveNode: (id: string, x: number, y: number) => void;
  onAddNode: (typeKey: string, x: number, y: number) => void;
  onConnect: (from: string, fromPort: string, to: string, toPort: string) => void;
  onRunNode: (id: string) => void;
  /** 打开这个节点的输出（跳到检查面板的输出 tab） */
  onOpenOutput?: (id: string) => void;
  /** 就这个节点问 AI */
  onAskAi?: (id: string) => void;
  /** 来自偏好设置。关掉时节点上不出现问 AI 入口 */
  aiEnabled?: boolean;
  onDuplicate: (id: string) => void;
  onDelete: (id: string) => void;
  /** 删连线。选中一根线按 Del 就走这里 */
  onDeleteEdges?: (ids: string[]) => void;
  /** 值变化时自动缩放到刚好装下整张图，切换示例后用它复位视角 */
  fitSignal?: number;
  /** 画布把当前视口中心（图坐标）写进来，点组件面板时把节点加在看得见的地方 */
  viewCenterRef?: React.MutableRefObject<{ x: number; y: number }>;
}

/** React Flow 的 hook 都要在 Provider 里面用，所以画布本体包一层 */
export function NodeCanvas(props: NodeCanvasProps) {
  return (
    <ReactFlowProvider>
      <Canvas {...props} />
    </ReactFlowProvider>
  );
}

function Canvas({
  nodes,
  edges,
  frames = [],
  selected,
  onSelect,
  onMoveNode,
  onAddNode,
  onConnect,
  onRunNode,
  onOpenOutput,
  onAskAi,
  aiEnabled = true,
  onDuplicate,
  onDelete,
  onDeleteEdges,
  fitSignal = 0,
  viewCenterRef,
}: NodeCanvasProps) {
  const wrapRef = useRef<HTMLDivElement>(null);
  const { fitView, screenToFlowPosition, zoomIn, zoomOut } = useReactFlow();

  const byId = useMemo(() => new Map(nodes.map((n) => [n.id, n])), [nodes]);

  const flowNodes = useMemo<Node[]>(
    () => [
      /*
       * 循环框做成压在最底下的节点（zIndex -1 会落到连线层下面），
       * 这样它跟着平移缩放走，不用自己算坐标。它只是一块背景标注，
       * 说明「框住的这几个节点每步重跑」，不是真容器：
       * 拖节点不会跟随，拖出框外也不会被移出循环。
       */
      ...frames.map((f) => ({
        id: `frame:${f.id}`,
        type: 'frame',
        position: { x: f.x, y: f.y },
        width: f.w,
        height: f.h,
        data: { frame: f },
        zIndex: -1,
        selectable: false,
        draggable: false,
        deletable: false,
        focusable: false,
        /* 事件全部放过去，点框里的空白仍然是在拖画布 */
        style: { pointerEvents: 'none' as const },
      })),
      ...nodes.map((n) => ({
        id: n.id,
        type: 'swift',
        position: { x: n.x, y: n.y },
        data: { node: n },
        selected: n.id === selected,
        /* 只能拖标题栏移动节点，body 里有可点的东西 */
        dragHandle: '.swift-node-header',
        style: { width: NODE_W },
      })),
    ],
    [nodes, frames, selected],
  );

  const flowEdges = useMemo<Edge[]>(
    () =>
      edges.map((ed) => {
        const src = byId.get(ed.from);
        const p = src ? findPort(src.type, ed.fromPort, 'out') : undefined;
        /* 连线颜色取端口数据类型，不取节点色——一眼看出这根线在传什么 */
        const color = p ? PORT_TYPES[p.type].color : '#6B7280';
        return {
          id: ed.id,
          source: ed.from,
          sourceHandle: ed.fromPort,
          target: ed.to,
          targetHandle: ed.toPort,
          /* 上游正在跑，线上的虚线往前流。animated 是 RF 自带的 */
          animated: src?.status === 'running',
          style: {
            /*
             * 设 --xy-edge-stroke 而不是直接写 stroke：直接写是内联样式，
             * 会压掉 RF 那条「选中时换色」的规则，选中就看不出来了。
             */
            '--xy-edge-stroke': color,
            strokeWidth: 3.5,
            /* 深底点阵上连线要有厚度，原来靠垫一条更粗的暗线，一个投影就够 */
            filter: 'drop-shadow(0 0 2px rgba(0,0,0,0.85))',
          } as CSSProperties,
        };
      }),
    [edges, byId],
  );

  /**
   * 连线合法性。三道校验的原因文案是有信息量的，所以不用 isValidConnection——
   * 那个只会静默不让连，用户只看到线弹回去，不知道为什么。
   * 这里放它连上、在 onConnect 里拦下来并把原因说出来。
   */
  const check = useCallback(
    (from: string, fromPort: string, to: string, toPort: string) => {
      if (from === to) return '不能连到自己身上';
      const a = byId.get(from);
      const b = byId.get(to);
      if (!a || !b) return '节点不存在';
      const pa = findPort(a.type, fromPort, 'out');
      const pb = findPort(b.type, toPort, 'in');
      if (!pa || !pb) return '端口不存在';
      if (pa.type !== pb.type) {
        return `类型不匹配：${PORT_TYPES[pa.type].label} 接不到 ${PORT_TYPES[pb.type].label} 上`;
      }
      /*
       * 锁定节点拒收任何手拉的连线。轨迹过滤就是这种：它的两路数据必须
       * 同进同出、且优势必须在它上游算好。这两条被改了训练会错但不报错——
       * 类型全匹配，上面那两道检查都拦不住。
       */
      if (NODE_TYPES[b.type]?.lockedPorts) {
        return `${NODE_TYPES[b.type].label} 的连线已锁定：两路数据必须同进同出，接错不报错`;
      }
      if (NODE_TYPES[a.type]?.lockedPorts) {
        return `${NODE_TYPES[a.type].label} 的输出已锁定，不能单独引出一路`;
      }
      return null;
    },
    [byId],
  );

  const handleConnect = useCallback(
    (c: Connection) => {
      if (!c.sourceHandle || !c.targetHandle) return;
      const err = check(c.source, c.sourceHandle, c.target, c.targetHandle);
      if (err) toast.warning(err);
      else onConnect(c.source, c.sourceHandle, c.target, c.targetHandle);
    },
    [check, onConnect],
  );

  /*
   * 尺寸变化由 RF 自己在内部记着；选中态是受控的（flowNodes 里按 selected 写死），
   * 所以这里只需要把拖动后的坐标写回去。
   */
  const handleNodesChange = useCallback<OnNodesChange>(
    (changes) => {
      for (const c of changes) {
        if (c.type === 'position' && c.position) {
          onMoveNode(c.id, Math.round(c.position.x), Math.round(c.position.y));
        }
      }
    },
    [onMoveNode],
  );

  /** 把视口中心同步给外面，点组件面板新增节点时要用 */
  const syncCenter = useCallback(() => {
    const r = wrapRef.current?.getBoundingClientRect();
    if (!viewCenterRef || !r) return;
    const p = screenToFlowPosition({ x: r.left + r.width / 2, y: r.top + r.height / 2 });
    viewCenterRef.current = { x: Math.round(p.x), y: Math.round(p.y) };
  }, [screenToFlowPosition, viewCenterRef]);

  const firstFit = useRef(true);
  useEffect(() => {
    /* 首次复位交给 <ReactFlow fitView>：这会儿节点还没量出尺寸，这里算不准 */
    if (firstFit.current) {
      firstFit.current = false;
      return;
    }
    fitView(FIT_ON_LOAD);
    syncCenter();
  }, [fitSignal, fitView, syncCenter]);

  /* Del 删除由 RF 的 deleteKeyCode 管，⌘D 复制它不管，自己留一个 */
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (!selected || !(e.metaKey || e.ctrlKey) || e.key.toLowerCase() !== 'd') return;
      e.preventDefault();
      onDuplicate(selected);
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [selected, onDuplicate]);

  const actions = useMemo<NodeActions>(
    () => ({
      onRunNode,
      onOpenOutput,
      onAskAi: aiEnabled ? onAskAi : undefined,
      onDuplicate,
      onDelete,
    }),
    [onRunNode, onOpenOutput, onAskAi, aiEnabled, onDuplicate, onDelete],
  );

  /** 当前图里出现过的数据类型，给图例用 */
  const legendTypes = useMemo(() => {
    const seen = new Set<string>();
    for (const ed of edges) {
      const src = byId.get(ed.from);
      const p = src ? findPort(src.type, ed.fromPort, 'out') : undefined;
      if (p) seen.add(p.type);
    }
    return [...seen] as (keyof typeof PORT_TYPES)[];
  }, [edges, byId]);

  return (
    <div ref={wrapRef} className="bg-titlebar relative min-w-0 flex-1">
      <ActionsCtx.Provider value={actions}>
        <ReactFlow
          nodes={flowNodes}
          edges={flowEdges}
          nodeTypes={NODE_VIEWS}
          onNodesChange={handleNodesChange}
          onConnect={handleConnect}
          onNodesDelete={(deleted) => deleted.forEach((n) => onDelete(n.id))}
          onEdgesDelete={(deleted) => onDeleteEdges?.(deleted.map((e) => e.id))}
          /*
           * 用「用户点了什么」驱动选中，而不是 onSelectionChange。
           *
           * onSelectionChange 反映的是 RF store 的状态，而 selected 又被我们写回
           * flowNodes——于是我们自己的每一次同步都会把这个回调再触发一遍，
           * 形成一个环。RF 在同步 prop 的间隙会瞬时清空选中，回调就带着 null 进来，
           * 右侧面板被这个回弹反复开关（看起来就是一直闪），
           * 顺便还会把 openInspector 刚指定的 tab 顶回参数页。
           *
           * onNodeClick / onPaneClick 只在真实点击时触发，不会被我们自己的同步引发。
           * 这里不加 id !== selected 的判断：重复点同一个节点要能把关掉的面板重新叫出来。
           */
          onNodeClick={(_, n) => {
            if (n.type === 'swift') onSelect(n.id);
          }}
          onPaneClick={() => onSelect(null)}
          onDragOver={(e) => {
            if (!e.dataTransfer.types.includes('application/swift-node')) return;
            e.preventDefault();
            e.dataTransfer.dropEffect = 'copy';
          }}
          onDrop={(e) => {
            const key = e.dataTransfer.getData('application/swift-node');
            if (!key) return;
            e.preventDefault();
            const p = screenToFlowPosition({ x: e.clientX, y: e.clientY });
            /* 让指针落在标题栏中间，看起来就是「拖到哪放到哪」 */
            onAddNode(key, Math.round(p.x - NODE_W / 2), Math.round(p.y - 15));
          }}
          onInit={syncCenter}
          onMoveEnd={syncCenter}
          fitView
          fitViewOptions={FIT_ON_LOAD}
          deleteKeyCode={['Delete', 'Backspace']}
          minZoom={0.3}
          maxZoom={1.6}
          /* 画布跟侧栏一样永远是深色壳，跟 app 的明暗主题无关 */
          colorMode="dark"
          /* 拉线时不给它上色：颜色每帧都变会把整张图重渲染一遍，兼容的输入口已经在亮了 */
          connectionLineStyle={{ strokeWidth: 3.5, stroke: 'rgba(255,255,255,0.8)', strokeDasharray: '7 6' }}
          attributionPosition="top-right"
        >
          {/* 点阵网格：两层不同疏密的点，跟原来的双层 radial-gradient 一样 */}
          <Background id="fine" variant={BackgroundVariant.Dots} gap={28} size={1.2} color="rgba(255,255,255,0.14)" />
          <Background id="coarse" variant={BackgroundVariant.Dots} gap={140} size={2.4} color="rgba(255,255,255,0.10)" />

          {/* 图例：只列当前图里真用到的数据类型，十五种颜色堆在那里反而没人看 */}
          {legendTypes.length > 0 && (
            <Panel position="bottom-left">
              <div className="border-sidebar-border flex max-w-85 flex-wrap gap-x-3 gap-y-1 rounded-[10px] border bg-black/55 px-2.5 py-1.5 backdrop-blur-sm">
                {legendTypes.map((t) => (
                  <span
                    key={t}
                    className="text-sidebar-foreground/60 inline-flex items-center gap-1.5 text-[11px]"
                  >
                    <span
                      className="h-[3px] w-3.5 rounded-sm"
                      style={{ background: PORT_TYPES[t].color }}
                    />
                    {PORT_TYPES[t].label}
                  </span>
                ))}
              </div>
            </Panel>
          )}

          {/* 三个按钮自己写而不是用 Controls 自带的：自带的提示文案是英文的 */}
          <Controls showZoom={false} showFitView={false} showInteractive={false}>
            <ControlButton onClick={() => zoomOut()} title="缩小">
              <Minus />
            </ControlButton>
            <ControlButton onClick={() => zoomIn()} title="放大">
              <Plus />
            </ControlButton>
            <ControlButton onClick={() => fitView(FIT_ALL)} title="缩放到看得见全图">
              <Maximize2 />
            </ControlButton>
          </Controls>
        </ReactFlow>
      </ActionsCtx.Provider>
    </div>
  );
}

/**
 * 正在拉的线是什么类型、从哪个节点出发，用来给兼容的输入口亮一下。
 *
 * 返回一个字符串而不是对象：指针每动一下这个选择器都会重算，
 * 返回对象的话引用每次都变，图上所有节点都会跟着重渲染。
 */
function linkSelector(c: ConnectionState) {
  if (!c.inProgress || !c.fromHandle?.id) return '';
  const g = (c.fromNode.data as SwiftNodeData).node;
  const p = findPort(g.type, c.fromHandle.id, 'out');
  return p ? `${p.type}|${g.id}` : '';
}

const HEADER_BTN =
  'inline-flex size-[18px] cursor-pointer items-center justify-center rounded text-white/85 transition-colors hover:bg-black/25 hover:text-white [&>svg]:size-3';

/** 单个节点。厚边框 + 饱和色标题栏 + 端口圆点 */
const SwiftNodeView = memo(function SwiftNodeView({ id, data, selected }: NodeProps) {
  /* nodeTypes 那边只认 NodeProps 这一种签名，泛型在这个边界上丢了，转回来 */
  const { node } = data as SwiftNodeData;
  const def = NODE_TYPES[node.type];
  const act = useContext(ActionsCtx);
  const link = useConnection(linkSelector);
  const [linkType, linkFrom] = link.split('|');

  const params = node.params ?? def.params;
  const code = node.code ?? def.code;
  const running = node.status === 'running';
  const hasOutput = node.status === 'done' || node.status === 'failed';
  /** 连线锁定：输出口拖不出来，标题栏挂个锁把原因说在提示里 */
  const locked = !!def.lockedPorts;

  return (
    <div
      className={cn(
        'bg-sidebar w-full overflow-hidden rounded-xl border-2 select-none',
        selected ? 'shadow-2xl' : 'border-white/12 shadow-lg',
      )}
      style={
        selected
          ? { borderColor: def.color, boxShadow: `0 0 0 4px ${def.color}33, 0 14px 30px rgba(0,0,0,0.5)` }
          : undefined
      }
    >
      {/* 标题栏：拖这里移动节点 */}
      <div
        className="swift-node-header flex h-[30px] cursor-grab items-center gap-[7px] px-2"
        style={{ background: def.color }}
      >
        <span className="inline-flex text-white/95 [&>svg]:size-3.5">{def.icon}</span>
        <span className="min-w-0 flex-1 truncate text-[12.5px] font-semibold tracking-wide text-white">
          {def.label}
        </span>

        <StatusDot status={node.status} />

        {locked && (
          <Hint title="这个节点的连线不能改：两路数据必须同进同出，接错不报错">
            <span className="shrink-0 text-white/90 [&>svg]:size-3">
              <Lock />
            </span>
          </Hint>
        )}

        {/*
          单节点操作。
          nodrag 让 RF 别把这里当拖动把手；停掉冒泡是为了别顺手把节点选中——
          「看输出」本来要把面板开到输出页，被选中一挤就跳回参数页了。

          复制/删除藏到鼠标移上来才出现：标题栏只有 208px 宽，
          五个图标一起排完，节点名字就只剩一个字了。
        */}
        <div
          className="nodrag group/act flex shrink-0 gap-px"
          onClick={(e) => e.stopPropagation()}
          onMouseDown={(e) => e.stopPropagation()}
        >
          <Hint title={def.runnable ? '只运行这个节点' : '这是声明式节点，跑一次只是把它准备好'}>
            <span className={HEADER_BTN} onClick={() => act?.onRunNode(id)}>
              {running ? <Loader2 className="animate-spin" /> : <Play />}
            </span>
          </Hint>
          {act?.onOpenOutput && (
            <Hint title={hasOutput ? '看这次跑出了什么' : '还没跑过，跑完才有输出'}>
              <span
                className={cn(HEADER_BTN, !hasOutput && 'opacity-45')}
                onClick={() => act.onOpenOutput?.(id)}
              >
                <FileText />
              </span>
            </Hint>
          )}
          {act?.onAskAi && (
            <Hint title="就这个节点问 AI">
              <span className={HEADER_BTN} onClick={() => act.onAskAi?.(id)}>
                <Bot />
              </span>
            </Hint>
          )}
          <Hint title="复制节点  ⌘D">
            <span
              className={cn(HEADER_BTN, !selected && 'opacity-0 group-hover/act:opacity-100')}
              onClick={() => act?.onDuplicate(id)}
            >
              <Copy />
            </span>
          </Hint>
          <Hint title="删除节点  Del">
            <span
              className={cn(HEADER_BTN, !selected && 'opacity-0 group-hover/act:opacity-100')}
              onClick={() => act?.onDelete(id)}
            >
              <X />
            </span>
          </Hint>
        </div>
      </div>

      {/*
        端口区。左右两列各排一路，第 i 个输入和第 i 个输出自然对齐，
        位置全靠布局撑出来——原来这里是按下标手算每个圆点的绝对坐标。
      */}
      <div className="grid min-h-5.5 grid-cols-2 gap-x-2 pt-1.5">
        <div className="flex flex-col gap-[7px]">
          {def.inputs.map((p) => (
            <PortRow
              key={p.key}
              port={p}
              dir="in"
              highlight={linkType === p.type && linkFrom !== id}
            />
          ))}
        </div>
        <div className="flex flex-col gap-[7px]">
          {def.outputs.map((p) => (
            <PortRow key={p.key} port={p} dir="out" connectable={!locked} />
          ))}
        </div>
      </div>

      {/* 代码节点显代码，其余显参数摘要 */}
      {code ? (
        <pre className="bg-titlebar mx-2 mt-1.5 mb-2.5 overflow-hidden rounded-lg border border-white/8 px-2 py-1.5 font-mono text-[11px] leading-[14px] text-[#C3C7D1]">
          {code}
        </pre>
      ) : (
        <div className="px-2.5 pt-0.5 pb-2.5">
          {params.map((p, i) => (
            <div
              key={`${p.label}-${i}`}
              className="flex justify-between gap-2 text-[11.5px] leading-[19px]"
            >
              <span className="text-sidebar-foreground/45 shrink-0">{p.label}</span>
              <span className="text-sidebar-foreground truncate font-mono">{p.value}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
});

function PortRow({
  port,
  dir,
  highlight,
  connectable = true,
}: {
  port: PortDef;
  dir: 'in' | 'out';
  /** 正在拉的线类型对得上，圆点放大并发光 */
  highlight?: boolean;
  connectable?: boolean;
}) {
  const color = PORT_TYPES[port.type].color;
  const size = highlight ? 14 : 12;
  return (
    <div className={cn('relative flex h-3.5 items-center gap-1.5', dir === 'in' ? 'ps-2.5' : 'justify-end pe-2.5')}>
      <Handle
        id={port.key}
        type={dir === 'in' ? 'target' : 'source'}
        position={dir === 'in' ? Position.Left : Position.Right}
        isConnectableStart={connectable}
        title={
          dir === 'out'
            ? connectable
              ? '从这里拖到下一个节点的输入口'
              : '这路输出已锁定，不能单独引出'
            : undefined
        }
        style={{
          ...(dir === 'in' ? { left: -7 } : { right: -7 }),
          top: '50%',
          transform: 'translateY(-50%)',
          width: size,
          height: size,
          minWidth: 0,
          minHeight: 0,
          background: color,
          /* 描边用画布底色，圆点看起来是嵌在节点边上的 */
          border: '2.5px solid var(--titlebar)',
          borderRadius: '50%',
          cursor: dir === 'out' && !connectable ? 'not-allowed' : undefined,
          boxShadow: highlight ? `0 0 0 3px ${color}66, 0 0 10px ${color}` : `0 0 0 1px ${color}`,
          transition: 'width 0.1s ease, height 0.1s ease, box-shadow 0.1s ease',
        }}
      />
      <span className="text-sidebar-foreground/60 text-[11px]">{port.label}</span>
    </div>
  );
}

function StatusDot({ status }: { status: GraphNode['status'] }) {
  if (status === 'idle') return null;
  const icon = {
    running: <Loader2 className="animate-spin" />,
    done: <Check />,
    failed: <TriangleAlert />,
  }[status];
  return (
    <span
      className={cn(
        'inline-flex size-4 shrink-0 items-center justify-center rounded-full text-white [&>svg]:size-2.5',
        status === 'failed' ? 'bg-[#B03A3A]' : 'bg-white/20',
      )}
    >
      {icon}
    </span>
  );
}

function FrameNodeView({ data }: NodeProps) {
  const { frame } = data as FrameNodeData;
  const c = frame.color ?? '#6366F1';
  return (
    <div
      className="size-full rounded-2xl border-2 border-dashed"
      style={{ borderColor: `${c}88`, background: `${c}12` }}
    >
      <div className="absolute -top-3 left-3 flex max-w-[calc(100%-24px)] items-center gap-2">
        <span
          className="shrink-0 rounded-[7px] px-2.5 py-0.5 text-[11.5px] font-semibold tracking-wide text-white"
          style={{ background: c }}
        >
          {frame.label}
        </span>
        {frame.note && (
          <span
            className="bg-titlebar text-sidebar-foreground/60 truncate rounded-[7px] border px-2 py-0.5 text-[11px]"
            style={{ borderColor: `${c}55` }}
          >
            {frame.note}
          </span>
        )}
      </div>
    </div>
  );
}

/** 必须是模块级常量：每次渲染换一个新对象，RF 会把所有节点重新挂载一遍 */
const NODE_VIEWS = { swift: SwiftNodeView, frame: FrameNodeView };
