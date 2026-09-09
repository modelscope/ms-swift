import { useEffect, useRef, useState } from 'react';
import { Tooltip, message } from 'antd';
import {
  CheckOutlined,
  CopyOutlined,
  CloseOutlined,
  ExclamationOutlined,
  FileTextOutlined,
  LoadingOutlined,
  LockOutlined,
  PlayCircleFilled,
  RobotOutlined,
} from '@ant-design/icons';
import {
  HEADER_H,
  NODE_TYPES,
  NODE_W,
  PORT_TYPES,
  findPort,
  nodeHeight,
  portY,
} from './nodeTypes';
import type { GraphEdge, GraphFrame, GraphNode, PortTypeKey } from './nodeTypes';
import { chrome } from '@/theme/theme';

/** 交互状态。拖节点 / 平移画布 / 拉连线三种互斥，用一个联合类型管住 */
type Interaction =
  | { kind: 'none' }
  | { kind: 'node'; id: string; dx: number; dy: number }
  | { kind: 'pan'; clientX: number; clientY: number; ox: number; oy: number }
  | { kind: 'link'; from: string; fromPort: string; x: number; y: number };

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
  /** 值变化时自动缩放到刚好装下整张图，切换示例后用它复位视角 */
  fitSignal?: number;
  /** 画布把当前视口中心（图坐标）写进来，点组件面板时把节点加在看得见的地方 */
  viewCenterRef?: React.MutableRefObject<{ x: number; y: number }>;
}

/**
 * 节点画布。深底 + 点阵网格，节点是厚边框的卡片，连线是粗贝塞尔曲线。
 *
 * 坐标系：外层容器是视口，内层 layer 上挂 translate(offset) scale(scale)。
 * 所有鼠标位置都先过 toCanvas() 换算回图坐标，缩放后拖拽才不会漂。
 */
export function NodeCanvas({
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
  fitSignal = 0,
  viewCenterRef,
}: NodeCanvasProps) {
  const wrapRef = useRef<HTMLDivElement>(null);
  const [offset, setOffset] = useState({ x: 40, y: 24 });
  const [scale, setScale] = useState(1);
  const [act, setAct] = useState<Interaction>({ kind: 'none' });

  /** 屏幕坐标 → 图坐标 */
  const toCanvas = (clientX: number, clientY: number) => {
    const r = wrapRef.current?.getBoundingClientRect();
    if (!r) return { x: 0, y: 0 };
    return {
      x: (clientX - r.left - offset.x) / scale,
      y: (clientY - r.top - offset.y) / scale,
    };
  };

  /**
   * 缩放到装下整张图。minScale 是下限：
   * GRPO 那张图展开有两千多像素宽，真按完全装下算会缩到 50%，字小得没法看。
   * 所以初始视角给个较高的下限、靠左对齐（从数据流起点开始看，往右拖），
   * 只有手点「适应」时才真缩到全图。
   */
  const fitView = (minScale = 0.4) => {
    const r = wrapRef.current?.getBoundingClientRect();
    if (!r || nodes.length === 0) return;
    /* 循环框比里面的节点大一圈，不算进去的话框边会被裁掉 */
    const xs = [...nodes.map((n) => n.x), ...frames.map((f) => f.x)];
    const ys = [...nodes.map((n) => n.y), ...frames.map((f) => f.y)];
    const xe = [
      ...nodes.map((n) => n.x + NODE_W),
      ...frames.map((f) => f.x + f.w),
    ];
    const ye = [
      ...nodes.map((n) => n.y + nodeHeight(NODE_TYPES[n.type])),
      ...frames.map((f) => f.y + f.h),
    ];
    const x1 = Math.min(...xs);
    const y1 = Math.min(...ys);
    const x2 = Math.max(...xe);
    const y2 = Math.max(...ye);
    const pad = 34;
    const raw = Math.min(
      1,
      (r.width - pad * 2) / Math.max(1, x2 - x1),
      (r.height - pad * 2) / Math.max(1, y2 - y1),
    );
    const sc = Math.max(minScale, +raw.toFixed(2));
    setScale(sc);
    /* 装不下时 Math.max(0, ...) 会归零，自然变成靠左上对齐 */
    setOffset({
      x: pad - x1 * sc + Math.max(0, (r.width - pad * 2 - (x2 - x1) * sc) / 2),
      y: pad - y1 * sc + Math.max(0, (r.height - pad * 2 - (y2 - y1) * sc) / 2),
    });
  };

  useEffect(() => {
    fitView(0.72);
  }, [fitSignal]);

  /* 把视口中心同步给外面，点组件面板新增节点时要用 */
  useEffect(() => {
    if (!viewCenterRef) return;
    const r = wrapRef.current?.getBoundingClientRect();
    if (!r) return;
    viewCenterRef.current = {
      x: Math.round((r.width / 2 - offset.x) / scale),
      y: Math.round((r.height / 2 - offset.y) / scale),
    };
  }, [offset, scale, viewCenterRef]);

  /**
   * 连线合法性：端口类型必须一致。
   * 挡在这里而不是等运行时报错——优势接不到损失口上，当场就该拒绝。
   */
  const checkConnect = (to: string, toPort: string, from: string, fromPort: string) => {
    if (from === to) return '不能连到自己身上';
    const a = nodes.find((n) => n.id === from);
    const b = nodes.find((n) => n.id === to);
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
  };

  /* 拖拽期间把 move/up 挂到 window 上，指针移出节点也不会断 */
  useEffect(() => {
    if (act.kind === 'none') return;

    const onMove = (e: PointerEvent) => {
      if (act.kind === 'node') {
        const p = toCanvas(e.clientX, e.clientY);
        onMoveNode(act.id, Math.round(p.x - act.dx), Math.round(p.y - act.dy));
      } else if (act.kind === 'pan') {
        setOffset({
          x: act.ox + (e.clientX - act.clientX),
          y: act.oy + (e.clientY - act.clientY),
        });
      } else if (act.kind === 'link') {
        const p = toCanvas(e.clientX, e.clientY);
        setAct({ ...act, x: p.x, y: p.y });
      }
    };

    const onUp = (e: PointerEvent) => {
      if (act.kind === 'link') {
        /** 落点是否是一个输入口：用 DOM 上的 data 属性反查，省掉一套命中测试 */
        const el = document.elementFromPoint(e.clientX, e.clientY) as HTMLElement | null;
        const port = el?.closest('[data-port-in]') as HTMLElement | null;
        if (port) {
          const to = port.dataset.nodeId;
          const toPort = port.dataset.portKey;
          if (to && toPort) {
            const err = checkConnect(to, toPort, act.from, act.fromPort);
            if (err) message.warning(err);
            else onConnect(act.from, act.fromPort, to, toPort);
          }
        }
      }
      setAct({ kind: 'none' });
    };

    window.addEventListener('pointermove', onMove);
    window.addEventListener('pointerup', onUp);
    return () => {
      window.removeEventListener('pointermove', onMove);
      window.removeEventListener('pointerup', onUp);
    };
  }, [act, offset, scale, onMoveNode, onConnect]);

  /* 选中节点后的键盘操作：复制 / 删除 */
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (!selected) return;
      const tag = (e.target as HTMLElement)?.tagName;
      if (tag === 'INPUT' || tag === 'TEXTAREA') return;
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'd') {
        e.preventDefault();
        onDuplicate(selected);
      } else if (e.key === 'Delete' || e.key === 'Backspace') {
        e.preventDefault();
        onDelete(selected);
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [selected, onDuplicate, onDelete]);

  /** 端口圆心在图坐标里的位置 */
  const portPos = (nodeId: string, portKey: string, dir: 'in' | 'out') => {
    const n = nodes.find((x) => x.id === nodeId);
    if (!n) return null;
    const def = NODE_TYPES[n.type];
    const list = dir === 'in' ? def.inputs : def.outputs;
    const i = list.findIndex((p) => p.key === portKey);
    if (i < 0) return null;
    return { x: n.x + (dir === 'out' ? NODE_W : 0), y: n.y + portY(i) };
  };

  /** 粗贝塞尔：横向控制点随距离伸缩，短连线不打结 */
  const path = (x1: number, y1: number, x2: number, y2: number) => {
    const d = Math.max(46, Math.abs(x2 - x1) * 0.5);
    return `M ${x1} ${y1} C ${x1 + d} ${y1}, ${x2 - d} ${y2}, ${x2} ${y2}`;
  };

  /** 正在拉的线是什么类型，用来给兼容的输入口亮一下 */
  const linkingType: PortTypeKey | null = (() => {
    if (act.kind !== 'link') return null;
    const src = nodes.find((n) => n.id === act.from);
    const sp = src ? findPort(src.type, act.fromPort, 'out') : undefined;
    return sp ? sp.type : null;
  })();

  /** 当前图里出现过的数据类型，给图例用 */
  const legendTypes: PortTypeKey[] = Array.from(
    new Set(
      edges
        .map((ed) => {
          const src = nodes.find((n) => n.id === ed.from);
          return src ? findPort(src.type, ed.fromPort, 'out')?.type : undefined;
        })
        .filter((t): t is PortTypeKey => !!t),
    ),
  );

  return (
    <div
      ref={wrapRef}
      onPointerDown={(e) => {
        /** 空白处按下：取消选中并开始平移 */
        if (e.target === e.currentTarget || (e.target as HTMLElement).dataset.canvasBg) {
          onSelect(null);
          setAct({ kind: 'pan', clientX: e.clientX, clientY: e.clientY, ox: offset.x, oy: offset.y });
        }
      }}
      onDragOver={(e) => {
        if (e.dataTransfer.types.includes('application/swift-node')) {
          e.preventDefault();
          e.dataTransfer.dropEffect = 'copy';
        }
      }}
      onDrop={(e) => {
        const key = e.dataTransfer.getData('application/swift-node');
        if (!key) return;
        e.preventDefault();
        const p = toCanvas(e.clientX, e.clientY);
        onAddNode(key, Math.round(p.x - NODE_W / 2), Math.round(p.y - HEADER_H / 2));
      }}
      style={{
        position: 'relative',
        flex: 1,
        minWidth: 0,
        overflow: 'hidden',
        background: '#191A23',
        /* 点阵网格：两层 radial-gradient，跟着平移一起动 */
        backgroundImage:
          'radial-gradient(rgba(255,255,255,0.10) 1.2px, transparent 1.2px), radial-gradient(rgba(255,255,255,0.045) 1px, transparent 1px)',
        backgroundSize: `${28 * scale}px ${28 * scale}px, ${140 * scale}px ${140 * scale}px`,
        backgroundPosition: `${offset.x}px ${offset.y}px, ${offset.x}px ${offset.y}px`,
        cursor: act.kind === 'pan' ? 'grabbing' : 'default',
        touchAction: 'none',
      }}
      data-canvas-bg="1"
    >
      <div
        data-canvas-bg="1"
        style={{
          position: 'absolute',
          inset: 0,
          transform: `translate(${offset.x}px, ${offset.y}px) scale(${scale})`,
          transformOrigin: '0 0',
        }}
      >
        {/*
          循环框层。压在连线和节点下面，pointerEvents 关掉——它只是一块
          背景标注，说明「框住的这几个节点每步重跑」，不是真容器：
          拖节点不会跟随，拖出框外也不会被移出循环。
        */}
        {frames.map((f) => {
          const c = f.color ?? '#6366F1';
          return (
            <div
              key={f.id}
              style={{
                position: 'absolute',
                left: f.x,
                top: f.y,
                width: f.w,
                height: f.h,
                borderRadius: 16,
                border: `2px dashed ${c}88`,
                background: `${c}12`,
                pointerEvents: 'none',
              }}
            >
              <div
                style={{
                  position: 'absolute',
                  left: 12,
                  top: -12,
                  display: 'flex',
                  alignItems: 'center',
                  gap: 8,
                  maxWidth: f.w - 24,
                }}
              >
                <span
                  style={{
                    background: c,
                    color: '#fff',
                    fontSize: 11.5,
                    fontWeight: 600,
                    letterSpacing: 0.3,
                    padding: '2px 9px',
                    borderRadius: 7,
                    flex: 'none',
                  }}
                >
                  {f.label}
                </span>
                {f.note && (
                  <span
                    style={{
                      background: '#191A23',
                      color: chrome.textDim,
                      fontSize: 11,
                      padding: '2px 8px',
                      borderRadius: 7,
                      border: `1px solid ${c}55`,
                      overflow: 'hidden',
                      whiteSpace: 'nowrap',
                      textOverflow: 'ellipsis',
                    }}
                  >
                    {f.note}
                  </span>
                )}
              </div>
            </div>
          );
        })}

        {/* 连线层。放在节点下面，pointerEvents 关掉不挡拖拽 */}
        <svg
          style={{
            position: 'absolute',
            left: 0,
            top: 0,
            width: 6000,
            height: 4000,
            overflow: 'visible',
            pointerEvents: 'none',
          }}
        >
          {edges.map((ed) => {
            const a = portPos(ed.from, ed.fromPort, 'out');
            const b = portPos(ed.to, ed.toPort, 'in');
            if (!a || !b) return null;
            const src = nodes.find((n) => n.id === ed.from);
            const sp = src ? findPort(src.type, ed.fromPort, 'out') : undefined;
            /* 连线颜色取端口数据类型，不取节点色——一眼看出这根线在传什么 */
            const color = sp ? PORT_TYPES[sp.type].color : '#6B7280';
            const flowing = src?.status === 'running';
            return (
              <g key={ed.id}>
                {/* 底下垫一条更粗的暗线，让连线在网格上有厚度 */}
                <path
                  d={path(a.x, a.y, b.x, b.y)}
                  fill="none"
                  stroke="#0E0F16"
                  strokeWidth={7}
                  strokeLinecap="round"
                  opacity={0.55}
                />
                <path
                  d={path(a.x, a.y, b.x, b.y)}
                  fill="none"
                  stroke={color}
                  strokeWidth={3.5}
                  strokeLinecap="round"
                  className={flowing ? 'edge-flow' : undefined}
                  strokeDasharray={flowing ? '9 7' : undefined}
                />
              </g>
            );
          })}

          {/* 正在拉的那根线 */}
          {act.kind === 'link' &&
            (() => {
              const a = portPos(act.from, act.fromPort, 'out');
              if (!a) return null;
              const src = nodes.find((n) => n.id === act.from);
              const sp = src ? findPort(src.type, act.fromPort, 'out') : undefined;
              return (
                <path
                  d={path(a.x, a.y, act.x, act.y)}
                  fill="none"
                  stroke={sp ? PORT_TYPES[sp.type].color : '#fff'}
                  strokeWidth={3.5}
                  strokeLinecap="round"
                  strokeDasharray="7 6"
                  opacity={0.85}
                />
              );
            })()}
        </svg>

        {/* 节点层 */}
        {nodes.map((n) => (
          <NodeBox
            key={n.id}
            node={n}
            selected={selected === n.id}
            linkType={linkingType}
            linkFrom={act.kind === 'link' ? act.from : null}
            onHeaderDown={(e) => {
              const p = toCanvas(e.clientX, e.clientY);
              onSelect(n.id);
              setAct({ kind: 'node', id: n.id, dx: p.x - n.x, dy: p.y - n.y });
            }}
            onStartLink={(portKey, e) => {
              const p = toCanvas(e.clientX, e.clientY);
              setAct({ kind: 'link', from: n.id, fromPort: portKey, x: p.x, y: p.y });
            }}
            onSelect={() => onSelect(n.id)}
            onRun={() => onRunNode(n.id)}
            onOpenOutput={onOpenOutput && (() => onOpenOutput(n.id))}
            onAskAi={aiEnabled && onAskAi ? () => onAskAi(n.id) : undefined}
            onDuplicate={() => onDuplicate(n.id)}
            onDelete={() => onDelete(n.id)}
          />
        ))}
      </div>

      {/* 图例：只列当前图里真用到的数据类型，否则十一种颜色堆在那里反而没人看 */ }
      {legendTypes.length > 0 && (
        <div
          style={{
            position: 'absolute',
            left: 12,
            bottom: 12,
            display: 'flex',
            flexWrap: 'wrap',
            gap: '4px 12px',
            maxWidth: 340,
            background: 'rgba(20,21,29,0.86)',
            border: `1px solid ${chrome.border}`,
            borderRadius: 10,
            padding: '7px 10px',
          }}
        >
          {legendTypes.map((t) => (
            <span
              key={t}
              style={{
                display: 'inline-flex',
                alignItems: 'center',
                gap: 5,
                fontSize: 11,
                color: chrome.textDim,
              }}
            >
              <span
                style={{
                  width: 14,
                  height: 3,
                  borderRadius: 2,
                  background: PORT_TYPES[t].color,
                }}
              />
              {PORT_TYPES[t].label}
            </span>
          ))}
        </div>
      )}

      {/* 缩放控件，右下角浮层 */}
      <div
        style={{
          position: 'absolute',
          right: 12,
          bottom: 12,
          display: 'flex',
          alignItems: 'center',
          gap: 2,
          background: 'rgba(20,21,29,0.86)',
          border: `1px solid ${chrome.border}`,
          borderRadius: 10,
          padding: 3,
        }}
      >
        <ZoomBtn label="缩小" onClick={() => setScale((s) => Math.max(0.4, +(s - 0.1).toFixed(2)))}>
          −
        </ZoomBtn>
        <span
          onClick={() => {
            setScale(1);
            setOffset({ x: 40, y: 24 });
          }}
          className="chrome-icon-btn"
          style={{
            fontSize: 11.5,
            color: chrome.textDim,
            padding: '0 8px',
            height: 24,
            display: 'inline-flex',
            alignItems: 'center',
            borderRadius: 7,
            cursor: 'pointer',
            fontVariantNumeric: 'tabular-nums',
          }}
        >
          {Math.round(scale * 100)}%
        </span>
        <ZoomBtn label="放大" onClick={() => setScale((s) => Math.min(1.6, +(s + 0.1).toFixed(2)))}>
          +
        </ZoomBtn>
        <Tooltip title="缩放到看得见全图">
          <span
            className="chrome-icon-btn"
            onClick={() => fitView(0.4)}
            style={{
              fontSize: 11.5,
              color: chrome.textDim,
              padding: '0 8px',
              height: 24,
              display: 'inline-flex',
              alignItems: 'center',
              borderRadius: 7,
              cursor: 'pointer',
            }}
          >
            适应
          </span>
        </Tooltip>
      </div>
    </div>
  );
}

function ZoomBtn({
  children,
  label,
  onClick,
}: {
  children: React.ReactNode;
  label: string;
  onClick: () => void;
}) {
  return (
    <Tooltip title={label}>
      <span
        className="chrome-icon-btn"
        onClick={onClick}
        style={{
          width: 24,
          height: 24,
          borderRadius: 7,
          display: 'inline-flex',
          alignItems: 'center',
          justifyContent: 'center',
          color: chrome.textDim,
          cursor: 'pointer',
          fontSize: 14,
        }}
      >
        {children}
      </span>
    </Tooltip>
  );
}

/** 单个节点。厚边框 + 饱和色标题栏 + 端口圆点 */
function NodeBox({
  node,
  selected,
  linkType,
  linkFrom,
  onHeaderDown,
  onStartLink,
  onSelect,
  onRun,
  onOpenOutput,
  onAskAi,
  onDuplicate,
  onDelete,
}: {
  node: GraphNode;
  selected: boolean;
  /** 正在拉线的类型；匹配的输入口会高亮 */
  linkType: PortTypeKey | null;
  linkFrom: string | null;
  onHeaderDown: (e: React.PointerEvent) => void;
  onStartLink: (portKey: string, e: React.PointerEvent) => void;
  onSelect: () => void;
  onRun: () => void;
  onOpenOutput?: () => void;
  onAskAi?: () => void;
  onDuplicate: () => void;
  onDelete: () => void;
}) {
  const def = NODE_TYPES[node.type];
  const params = node.params ?? def.params;
  const code = node.code ?? def.code;
  const h = nodeHeight(def);
  const running = node.status === 'running';
  /** 连线锁定：输出口拖不动，标题栏挂个锁把原因说在 tooltip 里 */
  const locked = !!def.lockedPorts;
  /**
   * 标题栏只有 208px 宽。运行/看输出/问 AI 是常用的，一直在；
   * 复制/删除每天用不到一次，藏到鼠标移上去才出现——
   * 不然五个图标排完，节点名字就只剩一个字了。
   */
  const [hover, setHover] = useState(false);
  const hasOutput = node.status === 'done' || node.status === 'failed';

  return (
    <div
      onPointerDown={onSelect}
      onPointerEnter={() => setHover(true)}
      onPointerLeave={() => setHover(false)}
      style={{
        position: 'absolute',
        left: node.x,
        top: node.y,
        width: NODE_W,
        minHeight: h,
        borderRadius: 12,
        background: '#232532',
        /* 厚边框：选中时用节点主色，平时是一圈亮边 */
        border: `2px solid ${selected ? def.color : 'rgba(255,255,255,0.13)'}`,
        boxShadow: selected
          ? `0 0 0 4px ${def.color}33, 0 14px 30px rgba(0,0,0,0.5)`
          : '0 8px 20px rgba(0,0,0,0.42)',
        overflow: 'hidden',
        userSelect: 'none',
      }}
    >
      {/* 标题栏：拖这里移动节点 */}
      <div
        onPointerDown={onHeaderDown}
        style={{
          height: HEADER_H,
          background: def.color,
          display: 'flex',
          alignItems: 'center',
          gap: 7,
          padding: '0 8px',
          cursor: 'grab',
        }}
      >
        <span style={{ color: '#fff', fontSize: 13, display: 'inline-flex', opacity: 0.95 }}>
          {def.icon}
        </span>
        <span
          style={{
            flex: 1,
            minWidth: 0,
            color: '#fff',
            fontSize: 12.5,
            fontWeight: 600,
            letterSpacing: 0.2,
            overflow: 'hidden',
            whiteSpace: 'nowrap',
            textOverflow: 'ellipsis',
          }}
        >
          {def.label}
        </span>

        <StatusDot status={node.status} />

        {locked && (
          <Tooltip title="这个节点的连线不能改：两路数据必须同进同出，接错不报错">
            <span style={{ color: '#fff', fontSize: 11, opacity: 0.9, flex: 'none' }}>
              <LockOutlined />
            </span>
          </Tooltip>
        )}

        {/* 单节点操作。按在按钮上不触发拖拽 */}
        <div style={{ display: 'flex', gap: 1 }} onPointerDown={(e) => e.stopPropagation()}>
          <Tooltip title={def.runnable ? '只运行这个节点' : '这是声明式节点，跑一次只是把它准备好'}>
            <span className="node-btn" onClick={onRun}>
              {running ? <LoadingOutlined /> : <PlayCircleFilled />}
            </span>
          </Tooltip>
          {onOpenOutput && (
            <Tooltip title={hasOutput ? '看这次跑出了什么' : '还没跑过，跑完才有输出'}>
              <span
                className="node-btn"
                onClick={onOpenOutput}
                style={{ opacity: hasOutput ? 1 : 0.42 }}
              >
                <FileTextOutlined />
              </span>
            </Tooltip>
          )}
          {onAskAi && (
            <Tooltip title="就这个节点问 AI">
              <span className="node-btn" onClick={onAskAi}>
                <RobotOutlined />
              </span>
            </Tooltip>
          )}
          {(hover || selected) && (
            <>
              <Tooltip title="复制节点  ⌘D">
                <span className="node-btn" onClick={onDuplicate}>
                  <CopyOutlined />
                </span>
              </Tooltip>
              <Tooltip title="删除节点  Del">
                <span className="node-btn" onClick={onDelete}>
                  <CloseOutlined />
                </span>
              </Tooltip>
            </>
          )}
        </div>
      </div>

      {/* 端口区 */}
      <div style={{ position: 'relative', paddingTop: 6, minHeight: 22 }}>
        {def.inputs.map((p, i) => (
          <div
            key={p.key}
            data-port-in="1"
            data-node-id={node.id}
            data-port-key={p.key}
            style={{
              position: 'absolute',
              left: -2,
              top: portY(i) - HEADER_H - 7,
              display: 'flex',
              alignItems: 'center',
              gap: 6,
              height: 14,
            }}
          >
            <Port
              color={PORT_TYPES[p.type].color}
              highlight={linkType === p.type && linkFrom !== node.id}
            />
            <span style={{ fontSize: 11, color: chrome.textDim }}>{p.label}</span>
          </div>
        ))}
        {def.outputs.map((p, i) => (
          <div
            key={p.key}
            onPointerDown={(e) => {
              e.stopPropagation();
              if (locked) return;
              onStartLink(p.key, e);
            }}
            title={locked ? '这路输出已锁定，不能单独引出' : '从这里拖到下一个节点的输入口'}
            style={{
              position: 'absolute',
              right: -2,
              top: portY(i) - HEADER_H - 7,
              display: 'flex',
              alignItems: 'center',
              gap: 6,
              height: 14,
              cursor: locked ? 'not-allowed' : 'crosshair',
            }}
          >
            <span style={{ fontSize: 11, color: chrome.textDim }}>{p.label}</span>
            <Port color={PORT_TYPES[p.type].color} />
          </div>
        ))}
        <div
          style={{
            height: Math.max(def.inputs.length, def.outputs.length, 1) * 21,
          }}
        />
      </div>

      {/* 代码节点显代码，其余显参数摘要 */}
      {code ? (
        <div
          style={{
            margin: '0 8px 9px',
            padding: '6px 8px',
            borderRadius: 8,
            background: '#15161E',
            border: '1px solid rgba(255,255,255,0.08)',
            fontFamily: 'ui-monospace, SFMono-Regular, Menlo, monospace',
            fontSize: 11,
            lineHeight: '14px',
            color: '#C3C7D1',
            whiteSpace: 'pre',
            overflow: 'hidden',
          }}
        >
          {code}
        </div>
      ) : (
        <div style={{ padding: '2px 10px 9px' }}>
          {params.map((p, i) => (
            <div
              key={`${p.label}-${i}`}
              style={{
                display: 'flex',
                justifyContent: 'space-between',
                gap: 8,
                fontSize: 11.5,
                lineHeight: '19px',
              }}
            >
              <span style={{ color: chrome.textFaint, flex: 'none' }}>{p.label}</span>
              <span
                style={{
                  color: chrome.text,
                  fontFamily: 'ui-monospace, SFMono-Regular, Menlo, monospace',
                  overflow: 'hidden',
                  whiteSpace: 'nowrap',
                  textOverflow: 'ellipsis',
                }}
              >
                {p.value}
              </span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function Port({ color, highlight }: { color: string; highlight?: boolean }) {
  return (
    <span
      style={{
        width: highlight ? 14 : 12,
        height: highlight ? 14 : 12,
        borderRadius: '50%',
        background: color,
        border: '2.5px solid #191A23',
        boxShadow: highlight ? `0 0 0 3px ${color}66, 0 0 10px ${color}` : `0 0 0 1px ${color}`,
        flex: 'none',
        transition: 'width 0.1s ease, height 0.1s ease, box-shadow 0.1s ease',
      }}
    />
  );
}

function StatusDot({ status }: { status: GraphNode['status'] }) {
  if (status === 'idle') return null;
  const map = {
    running: { bg: 'rgba(255,255,255,0.22)', icon: <LoadingOutlined /> },
    done: { bg: 'rgba(255,255,255,0.22)', icon: <CheckOutlined /> },
    failed: { bg: '#B03A3A', icon: <ExclamationOutlined /> },
  } as const;
  const s = map[status];
  return (
    <span
      style={{
        width: 16,
        height: 16,
        borderRadius: '50%',
        background: s.bg,
        color: '#fff',
        fontSize: 9,
        display: 'inline-flex',
        alignItems: 'center',
        justifyContent: 'center',
        flex: 'none',
      }}
    >
      {s.icon}
    </span>
  );
}
