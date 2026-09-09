import { useMemo, useState } from 'react';
import { Input, Tooltip } from 'antd';
import { SearchOutlined } from '@ant-design/icons';
import { CATEGORY_ORDER, NODE_TYPE_LIST, PORT_TYPES } from './nodeTypes';
import type { NodeTypeDef } from './nodeTypes';
import { chrome } from '@/theme/theme';

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
    <div
      style={{
        width: 214,
        flex: 'none',
        display: 'flex',
        flexDirection: 'column',
        background: chrome.sidebar,
        borderInlineEnd: `1px solid ${chrome.border}`,
        overflow: 'hidden',
      }}
    >
      <div style={{ padding: '10px 10px 8px' }}>
        <div
          style={{
            fontSize: 12,
            color: chrome.textFaint,
            marginBottom: 8,
            paddingInlineStart: 2,
          }}
        >
          组件
        </div>
        <Input
          variant="filled"
          size="small"
          placeholder="搜索组件"
          prefix={<SearchOutlined style={{ color: chrome.textFaint, fontSize: 11 }} />}
          onChange={(e) => setKw(e.target.value)}
          style={{
            background: 'rgba(255,255,255,0.06)',
            color: chrome.text,
            borderRadius: 8,
          }}
        />
      </div>

      <div style={{ flex: 1, overflowY: 'auto', padding: '0 8px 12px' }}>
        {grouped.map((g) => (
          <div key={g.category} style={{ marginBottom: 12 }}>
            <div
              style={{
                fontSize: 11,
                color: chrome.textFaint,
                padding: '6px 4px 5px',
                letterSpacing: 0.3,
              }}
            >
              {g.category}
            </div>
            {g.items.map((d) => (
              <PaletteItem key={d.key} def={d} onAdd={onAdd} />
            ))}
          </div>
        ))}
      </div>

      <div
        style={{
          flex: 'none',
          borderTop: `1px solid ${chrome.border}`,
          padding: '9px 12px',
          fontSize: 11.5,
          color: chrome.textFaint,
          lineHeight: '17px',
        }}
      >
        拖到画布，或点一下加到中央
      </div>
    </div>
  );
}

function PaletteItem({ def, onAdd }: { def: NodeTypeDef; onAdd: (k: string) => void }) {
  /** 端口类型比个数有用：知道要接什么才知道往哪拖 */
  const io = (ps: typeof def.inputs) =>
    ps.length ? ps.map((p) => PORT_TYPES[p.type].label).join(' + ') : '—';

  return (
    <Tooltip
      placement="right"
      mouseEnterDelay={0.35}
      title={
        <div style={{ fontSize: 12, lineHeight: '18px' }}>
          {def.hint && <div style={{ marginBottom: 3 }}>{def.hint}</div>}
          <div style={{ opacity: 0.7 }}>
            入 {io(def.inputs)} → 出 {io(def.outputs)}
          </div>
        </div>
      }
    >
      <div
        draggable
        onDragStart={(e) => {
          e.dataTransfer.setData('application/swift-node', def.key);
          e.dataTransfer.effectAllowed = 'copy';
        }}
        onClick={() => onAdd(def.key)}
        className="palette-item"
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: 9,
          height: 32,
          padding: '0 8px',
          borderRadius: 8,
          cursor: 'grab',
          color: chrome.textDim,
          fontSize: 13,
          userSelect: 'none',
        }}
      >
        {/* 左侧色条：跟节点标题栏同色，拖之前就知道会得到什么 */}
        <span
          style={{
            width: 4,
            height: 17,
            borderRadius: 3,
            background: def.color,
            flex: 'none',
          }}
        />
        <span style={{ fontSize: 13, color: def.color, display: 'inline-flex', flex: 'none' }}>
          {def.icon}
        </span>
        <span style={{ overflow: 'hidden', whiteSpace: 'nowrap', textOverflow: 'ellipsis' }}>
          {def.label}
        </span>
      </div>
    </Tooltip>
  );
}
