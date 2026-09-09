import { useMemo, useState } from 'react';
import { Search } from 'lucide-react';
import { Hint } from '@/components/Hint';
import { Input } from '@/components/ui/input';
import { CATEGORY_ORDER, NODE_TYPE_LIST, PORT_TYPES } from './nodeTypes';
import type { NodeTypeDef } from './nodeTypes';

/**
 * 左侧组件面板。两种添加方式都支持：
 * 拖到画布上（HTML5 DnD，dataTransfer 带 type key），或者直接点一下加到画布中央。
 */
export function NodePalette({ onAdd }: { onAdd: (typeKey: string) => void }) {
  const [kw, setKw] = useState('');

  const grouped = useMemo(() => {
    const hit = NODE_TYPE_LIST.filter(
      (d) => kw === '' || d.label.includes(kw) || d.key.includes(kw.toLowerCase()),
    );
    return CATEGORY_ORDER.map((c) => ({
      category: c,
      items: hit.filter((d) => d.category === c),
    })).filter((g) => g.items.length > 0);
  }, [kw]);

  return (
    <div className="bg-sidebar border-sidebar-border flex w-54 flex-none flex-col overflow-hidden border-e">
      <div className="px-2.5 pt-2.5 pb-2">
        <div className="text-sidebar-foreground/45 mb-2 ps-0.5 text-xs">组件</div>
        {/* 搜索图标压在输入框里，所以给输入框留出左侧内边距 */}
        <div className="relative">
          <Search className="text-sidebar-foreground/45 pointer-events-none absolute start-2 top-1/2 size-3 -translate-y-1/2" />
          <Input
            value={kw}
            onChange={(e) => setKw(e.target.value)}
            placeholder="搜索组件"
            className="text-sidebar-foreground placeholder:text-sidebar-foreground/40 h-8 border-white/10 bg-white/6 ps-7 text-[13px]"
          />
        </div>
      </div>

      <div className="flex-1 overflow-y-auto px-2 pb-3">
        {grouped.map((g) => (
          <div key={g.category} className="mb-3">
            <div className="text-sidebar-foreground/45 px-1 pt-1.5 pb-1 text-[11px] tracking-wide">
              {g.category}
            </div>
            {g.items.map((d) => (
              <PaletteItem key={d.key} def={d} onAdd={onAdd} />
            ))}
          </div>
        ))}
      </div>

      <div className="border-sidebar-border text-sidebar-foreground/45 flex-none border-t px-3 py-2.5 text-[11.5px] leading-[17px]">
        拖到画布，或点一下加到中央
      </div>
    </div>
  );
}

function PaletteItem({ def, onAdd }: { def: NodeTypeDef; onAdd: (k: string) => void }) {
  /** 端口类型比个数有用：知道要接什么才知道往哪拖 */
  const io = (ps: NodeTypeDef['inputs']) =>
    ps.length ? ps.map((p) => PORT_TYPES[p.type].label).join(' + ') : '—';

  return (
    <Hint
      side="right"
      title={
        <div className="text-xs leading-[18px]">
          {def.hint && <div className="mb-0.5">{def.hint}</div>}
          <div className="opacity-70">
            入 {io(def.inputs)} → 出 {io(def.outputs)}
          </div>
        </div>
      }
    >
      {/*
        hover 反馈原来挂在一个叫 palette-item 的类上，而这个类整个项目里
        没有任何定义——面板项其实一直是没有反馈的。这里显式写出来。
      */}
      <div
        draggable
        onDragStart={(e) => {
          e.dataTransfer.setData('application/swift-node', def.key);
          e.dataTransfer.effectAllowed = 'copy';
        }}
        onClick={() => onAdd(def.key)}
        className="text-sidebar-foreground/70 hover:bg-sidebar-accent hover:text-sidebar-foreground flex h-8 cursor-grab items-center gap-2.5 rounded-lg px-2 text-[13px] transition-colors select-none"
      >
        {/* 左侧色条：跟节点标题栏同色，拖之前就知道会得到什么 */}
        <span
          className="h-[17px] w-1 flex-none rounded-sm"
          style={{ background: def.color }}
        />
        <span className="inline-flex flex-none [&>svg]:size-3.5" style={{ color: def.color }}>
          {def.icon}
        </span>
        <span className="truncate">{def.label}</span>
      </div>
    </Hint>
  );
}
