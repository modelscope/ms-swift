import { useState } from 'react';
import { Avatar, Button, Empty, Input, List, Segmented, Select, Space, Tag, Tooltip } from 'antd';
import {
  ArrowUpOutlined,
  BulbOutlined,
  CodeOutlined,
  GlobalOutlined,
  PaperClipOutlined,
  PlusOutlined,
  RobotOutlined,
  SplitCellsOutlined,
  UserOutlined,
} from '@ant-design/icons';
import { PageHeader } from '@/components/PageHeader';
import { contentHeight } from '@/layout/metrics';
import { MODULES } from '@/theme/modules';
import { brand, logo, logoGradient, neutral } from '@/theme/theme';
import { features } from '@/config/features';
import { availableModels, conversations } from '@/mock/data';

const DEMO_TURNS = [
  { role: 'user' as const, content: '你是谁？介绍一下你自己。' },
  {
    role: 'assistant' as const,
    content: '我是基于 Qwen 微调的助手，由 SWIFT 训练框架产出。可以帮你写代码、答疑、处理文本任务。',
  },
  { role: 'user' as const, content: '用一句话解释什么是 LoRA。' },
  {
    role: 'assistant' as const,
    content: 'LoRA 通过在原权重旁挂一对低秩矩阵、只训练这对小矩阵来微调大模型，从而大幅降低显存与存储开销。',
  },
];

/** 可开关的工具胶囊，选中时染上品牌淡紫 */
function Chip({
  active,
  onClick,
  icon,
  label,
}: {
  active?: boolean;
  onClick?: () => void;
  icon: React.ReactNode;
  label: string;
}) {
  return (
    <Button
      size="small"
      shape="round"
      icon={icon}
      onClick={onClick}
      style={{
        color: active ? brand.primaryActive : neutral.textSecondary,
        background: active ? brand.soft : 'transparent',
        borderColor: active ? brand.softBorder : 'transparent',
        fontWeight: active ? 600 : 400,
      }}
    >
      {label}
    </Button>
  );
}

interface ComposerState {
  value: string;
  onChange: (v: string) => void;
  web: boolean;
  setWeb: (v: boolean) => void;
  code: boolean;
  setCode: (v: boolean) => void;
  thinking: string;
  setThinking: (v: string) => void;
  onSend: () => void;
}

/** 大对话框：hero 用大号（big），会话进行中用普通号贴底 */
function Composer({ big, state }: { big?: boolean; state: ComposerState }) {
  return (
    <div
      style={{
        width: '100%',
        maxWidth: big ? 720 : undefined,
        background: '#fff',
        border: `1px solid ${neutral.border}`,
        borderRadius: 22,
        boxShadow: big ? '0 10px 30px rgba(30,34,68,0.10)' : 'none',
        padding: 14,
      }}
    >
      <Input.TextArea
        variant="borderless"
        value={state.value}
        onChange={(e) => state.onChange(e.target.value)}
        placeholder="问点什么……（示意，未接后端）"
        autoSize={{ minRows: big ? 2 : 1, maxRows: 6 }}
        style={{ fontSize: 15, padding: '2px 6px', resize: 'none' }}
      />
      <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginTop: 8 }}>
        <Button type="text" size="small" shape="circle" icon={<PaperClipOutlined />} />
        <Chip active={state.web} onClick={() => state.setWeb(!state.web)} icon={<GlobalOutlined />} label="联网搜索" />
        <Chip active={state.code} onClick={() => state.setCode(!state.code)} icon={<CodeOutlined />} label="代码" />
        <div style={{ marginInlineStart: 'auto', display: 'flex', alignItems: 'center', gap: 8 }}>
          <Select
            size="small"
            variant="borderless"
            value={state.thinking}
            onChange={state.setThinking}
            suffixIcon={<BulbOutlined />}
            popupMatchSelectWidth={false}
            options={[
              { value: 'off', label: '不思考' },
              { value: 'mid', label: '思考 · 中' },
              { value: 'high', label: '思考 · 高' },
            ]}
          />
          <Button type="primary" shape="circle" icon={<ArrowUpOutlined />} onClick={state.onSend} />
        </div>
      </div>
    </div>
  );
}

/** 单个对话列，多模型对比时并排放多列 */
function ChatColumn({ model, accent }: { model: string; accent: string }) {
  return (
    <div
      style={{
        flex: 1,
        minWidth: 0,
        display: 'flex',
        flexDirection: 'column',
        border: `1px solid ${neutral.borderLight}`,
        borderRadius: 14,
        overflow: 'hidden',
      }}
    >
      <div
        style={{
          padding: '8px 14px',
          borderBottom: `1px solid ${neutral.borderLight}`,
          background: neutral.bgSubtle,
          display: 'flex',
          alignItems: 'center',
          gap: 8,
        }}
      >
        <Tag color={accent} style={{ margin: 0, borderRadius: 8 }}>
          <RobotOutlined /> {model}
        </Tag>
      </div>
      <div style={{ flex: 1, overflow: 'auto', padding: 16, display: 'flex', flexDirection: 'column', gap: 14 }}>
        {DEMO_TURNS.map((m, i) => (
          <div
            key={i}
            style={{ display: 'flex', gap: 10, flexDirection: m.role === 'user' ? 'row-reverse' : 'row' }}
          >
            <Avatar
              size={28}
              style={{ flex: 'none', background: m.role === 'user' ? '#e5e7eb' : accent }}
              icon={m.role === 'user' ? <UserOutlined /> : <RobotOutlined />}
            />
            <div
              style={{
                maxWidth: '78%',
                padding: '9px 13px',
                borderRadius: 12,
                fontSize: 14,
                lineHeight: '22px',
                background: m.role === 'user' ? brand.soft : neutral.bgSubtle,
                color: neutral.text,
              }}
            >
              {m.content}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

/**
 * 对话页。打开即见一个居中的大对话框（hero）；发一条或点开历史会话后，
 * 才切到左列表 + 右对话区。「对比模式」下右侧变多列——对应设计里的多模型并排对比。
 */
export function ChatPage() {
  const mod = MODULES.chat;
  const [started, setStarted] = useState(false);
  const [compare, setCompare] = useState(false);
  const [models, setModels] = useState([availableModels[0], availableModels[4]]);
  const [single, setSingle] = useState(availableModels[0]);

  // composer 共享状态
  const [draft, setDraft] = useState('');
  const [web, setWeb] = useState(true);
  const [code, setCode] = useState(false);
  const [thinking, setThinking] = useState('mid');

  const composer: ComposerState = {
    value: draft,
    onChange: setDraft,
    web,
    setWeb,
    code,
    setCode,
    thinking,
    setThinking,
    onSend: () => setStarted(true),
  };

  // 未开始：整屏居中的大对话框
  if (!started) {
    return (
      <div
        style={{
          height: contentHeight(),
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center',
          gap: 26,
          padding: 24,
        }}
      >
        <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 16 }}>
          <div
            style={{
              width: 60,
              height: 60,
              borderRadius: 18,
              background: logoGradient,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              color: '#fff',
              fontSize: 28,
              boxShadow: '0 8px 20px rgba(62,69,133,0.32)',
            }}
          >
            <RobotOutlined />
          </div>
          <div style={{ fontSize: 26, fontWeight: 600, color: neutral.text }}>有什么可以帮你？</div>
          <Select
            variant="borderless"
            value={single}
            onChange={setSingle}
            popupMatchSelectWidth={false}
            style={{ fontSize: 13 }}
            options={availableModels.map((m) => ({ value: m, label: m }))}
          />
        </div>
        <Composer big state={composer} />
        <div style={{ fontSize: 12, color: neutral.textTertiary }}>
          可对话本地或已部署的远端模型，支持多模型并排对比
        </div>
      </div>
    );
  }

  // 会话进行中：左会话列表 + 右对话区
  return (
    <>
      <PageHeader
        module={mod}
        title={mod.label}
        desc={mod.desc}
        extra={
          features.chatCompare ? (
            <Segmented
              value={compare ? 'compare' : 'single'}
              onChange={(v) => setCompare(v === 'compare')}
              options={[
                { label: '单模型', value: 'single' },
                { label: '对比', value: 'compare', icon: <SplitCellsOutlined /> },
              ]}
            />
          ) : undefined
        }
      />

      {/* 74px 是上面那截页头 */}
      <div style={{ display: 'flex', height: contentHeight(74) }}>
        <div
          style={{
            width: 248,
            flex: 'none',
            borderInlineEnd: `1px solid ${neutral.borderLight}`,
            display: 'flex',
            flexDirection: 'column',
          }}
        >
          <div style={{ padding: 12 }}>
            <Button type="primary" block shape="round" icon={<PlusOutlined />} onClick={() => setStarted(false)}>
              新对话
            </Button>
          </div>
          <List
            style={{ flex: 1, overflow: 'auto' }}
            dataSource={conversations}
            renderItem={(c, i) => (
              <List.Item
                onClick={() => setStarted(true)}
                style={{
                  padding: '10px 14px',
                  cursor: 'pointer',
                  background: i === 0 ? mod.accentSoft : undefined,
                  borderInlineStart: i === 0 ? `2px solid ${mod.accent}` : '2px solid transparent',
                }}
              >
                <List.Item.Meta
                  title={<span style={{ fontSize: 13, color: neutral.text }}>{c.title}</span>}
                  description={<span style={{ fontSize: 12, color: neutral.textTertiary }}>{c.updatedAt}</span>}
                />
              </List.Item>
            )}
          />
        </div>

        <div style={{ flex: 1, minWidth: 0, display: 'flex', flexDirection: 'column', padding: 16 }}>
          <div style={{ marginBottom: 12 }}>
            {compare ? (
              <Space wrap>
                <span style={{ fontSize: 13, color: neutral.textSecondary }}>对比模型：</span>
                <Select
                  mode="multiple"
                  value={models}
                  onChange={setModels}
                  style={{ minWidth: 360 }}
                  maxCount={3}
                  options={availableModels.map((m) => ({ value: m, label: m }))}
                />
                <Tooltip title="部署页里 RUNNING 的服务会自动出现在这里">
                  <Tag>来自部署服务</Tag>
                </Tooltip>
              </Space>
            ) : (
              <Select
                value={single}
                onChange={setSingle}
                style={{ width: 320 }}
                options={availableModels.map((m) => ({ value: m, label: m }))}
              />
            )}
          </div>

          <div style={{ flex: 1, display: 'flex', gap: 14, minHeight: 0 }}>
            {compare ? (
              models.length ? (
                models.map((m, i) => (
                  <ChatColumn key={m} model={m} accent={[logo.indigo, logo.blue, logo.plum][i % 3]} />
                ))
              ) : (
                <Empty description="选择要对比的模型" style={{ margin: 'auto' }} />
              )
            ) : (
              <ChatColumn model={single} accent={mod.accent} />
            )}
          </div>

          <div style={{ marginTop: 14 }}>
            <Composer state={composer} />
          </div>
        </div>
      </div>
    </>
  );
}
