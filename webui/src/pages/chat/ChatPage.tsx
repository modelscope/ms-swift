import { useState, type ReactNode } from 'react';
import { ArrowUp, Bot, Code, Columns2, Globe, Lightbulb, Paperclip, Plus, User } from 'lucide-react';
import { cn } from 'cn';
import { PageHeader } from '@/components/PageHeader';
import { CheckboxGroup, SegmentedControl, SelectInput } from '@/components/FormField';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Textarea } from '@/components/ui/textarea';
import { Tooltip, TooltipContent, TooltipTrigger } from '@/components/ui/tooltip';
import { contentHeight } from '@/layout/metrics';
import { MODULES } from '@/theme/modules';
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

/**
 * 思考档位。灯泡画在每个选项的标签里而不是当作下拉的后缀图标：
 * Radix 的 Select 触发器右边固定是那个箭头，而 SelectValue 显示的就是选中项的标签，
 * 把图标写进标签，收起状态下也一样能看见。
 */
const THINKING_OPTIONS = [
  { value: 'off', label: '不思考' },
  { value: 'mid', label: '思考 · 中' },
  { value: 'high', label: '思考 · 高' },
].map((o) => ({
  value: o.value,
  label: (
    <span className="flex items-center gap-1.5">
      <Lightbulb size={12} />
      {o.label}
    </span>
  ),
}));

/**
 * 对比模式下三列各自的底色。取自 logo 渐变上采样出的靛 / 蓝 / 紫（chart-1..3），
 * 作用是让人一眼看出某句话出自哪个模型——单列时不需要区分，直接走品牌色。
 */
const COLUMN_TONES = ['bg-chart-1', 'bg-chart-2', 'bg-chart-3'];

/** 可开关的工具胶囊，选中时染上品牌淡色 */
function Chip({
  active,
  onClick,
  icon,
  label,
}: {
  active?: boolean;
  onClick?: () => void;
  icon: ReactNode;
  label: string;
}) {
  return (
    <Button
      type="button"
      variant="ghost"
      size="sm"
      onClick={onClick}
      className={cn(
        'h-7 rounded-full px-2.5 text-xs font-normal',
        active && 'bg-primary/10 text-primary hover:bg-primary/15 hover:text-primary font-medium',
      )}
    >
      {icon}
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
      className={cn(
        'bg-card border-border w-full rounded-2xl border p-3.5',
        big && 'max-w-[720px] shadow-lg',
      )}
    >
      {/*
        高度不用 antd 那套 autoSize：shadcn 的 textarea 自带 field-sizing-content，
        跟着内容长，只要给一个 max-h 兜住就等于原来的 maxRows。
      */}
      <Textarea
        value={state.value}
        onChange={(e) => state.onChange(e.target.value)}
        placeholder="问点什么……（示意，未接后端）"
        className={cn(
          'max-h-40 resize-none border-0 bg-transparent px-1.5 py-0.5 shadow-none focus-visible:ring-0 md:text-[15px]',
          big ? 'min-h-12' : 'min-h-7',
        )}
      />
      <div className="mt-2 flex items-center gap-2">
        <Button variant="ghost" size="icon" className="size-7 rounded-full">
          <Paperclip />
        </Button>
        <Chip active={state.web} onClick={() => state.setWeb(!state.web)} icon={<Globe />} label="联网搜索" />
        <Chip active={state.code} onClick={() => state.setCode(!state.code)} icon={<Code />} label="代码" />
        <div className="ms-auto flex items-center gap-2">
          <SelectInput
            value={state.thinking}
            onChange={state.setThinking}
            options={THINKING_OPTIONS}
            className="text-muted-foreground h-7 w-auto gap-1.5 border-0 px-2 text-xs shadow-none"
          />
          <Button size="icon" className="size-8 rounded-full" onClick={state.onSend}>
            <ArrowUp />
          </Button>
        </div>
      </div>
    </div>
  );
}

/** 单个对话列，多模型对比时并排放多列。tone 是一个背景色 class，见 COLUMN_TONES */
function ChatColumn({ model, tone }: { model: string; tone: string }) {
  return (
    <div className="border-border/60 flex min-w-0 flex-1 flex-col overflow-hidden rounded-lg border">
      <div className="border-border/60 bg-muted/50 flex items-center gap-2 border-b px-3.5 py-2">
        <Badge className={cn('gap-1.5 text-white', tone)}>
          <Bot /> {model}
        </Badge>
      </div>
      <div className="flex flex-1 flex-col gap-3.5 overflow-auto p-4">
        {DEMO_TURNS.map((m, i) => {
          const mine = m.role === 'user';
          return (
            <div key={i} className={cn('flex gap-2.5', mine && 'flex-row-reverse')}>
              <span
                className={cn(
                  'flex size-7 flex-none items-center justify-center rounded-full [&>svg]:size-3.5',
                  mine ? 'bg-secondary text-secondary-foreground' : cn('text-white', tone),
                )}
              >
                {mine ? <User /> : <Bot />}
              </span>
              <div
                className={cn(
                  'text-foreground max-w-[78%] rounded-lg px-3.5 py-2.5 text-sm leading-[22px]',
                  mine ? 'bg-primary/10' : 'bg-muted',
                )}
              >
                {m.content}
              </div>
            </div>
          );
        })}
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
  const [activeConv, setActiveConv] = useState(conversations[0].id);

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

  const modelOptions = availableModels.map((m) => ({ value: m, label: m }));

  // 未开始：整屏居中的大对话框
  if (!started) {
    return (
      <div
        className="flex flex-col items-center justify-center gap-6 p-6"
        style={{ height: contentHeight() }}
      >
        <div className="flex flex-col items-center gap-4">
          {/* 渐变直接用 Tailwind 的 from/to 表达，方向和 logo 一致：钢蓝走到紫 */}
          <span className="from-chart-2 to-chart-3 flex size-15 items-center justify-center rounded-[18px] bg-gradient-to-br text-white shadow-lg [&>svg]:size-7">
            <Bot />
          </span>
          <div className="text-foreground text-[26px] font-semibold">有什么可以帮你？</div>
          <SelectInput
            value={single}
            onChange={setSingle}
            options={modelOptions}
            className="text-muted-foreground h-8 w-auto border-0 text-[13px] shadow-none"
          />
        </div>
        <Composer big state={composer} />
        <div className="text-muted-foreground text-xs">
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
            <SegmentedControl
              value={compare ? 'compare' : 'single'}
              onChange={(v) => setCompare(v === 'compare')}
              options={[
                { value: 'single', label: '单模型' },
                {
                  value: 'compare',
                  label: (
                    <span className="flex items-center gap-1.5">
                      <Columns2 size={13} /> 对比
                    </span>
                  ),
                },
              ]}
            />
          ) : undefined
        }
      />

      {/* 74px 是上面那截页头 */}
      <div className="flex" style={{ height: contentHeight(74) }}>
        <div className="border-border/60 flex w-62 flex-none flex-col border-e">
          <div className="p-3">
            <Button className="w-full rounded-full" onClick={() => setStarted(false)}>
              <Plus /> 新对话
            </Button>
          </div>
          <div className="flex-1 overflow-auto">
            {conversations.map((c) => (
              <button
                key={c.id}
                type="button"
                onClick={() => setActiveConv(c.id)}
                className={cn(
                  'flex w-full flex-col gap-0.5 border-s-2 px-3.5 py-2.5 text-left',
                  c.id === activeConv
                    ? 'border-primary bg-primary/8'
                    : 'hover:bg-muted border-transparent',
                )}
              >
                <span className="text-foreground truncate text-[13px]">{c.title}</span>
                <span className="text-muted-foreground text-xs">{c.updatedAt}</span>
              </button>
            ))}
          </div>
        </div>

        <div className="flex min-w-0 flex-1 flex-col p-4">
          <div className="mb-3">
            {compare ? (
              <div className="flex flex-wrap items-center gap-x-4 gap-y-2">
                <span className="text-muted-foreground text-[13px]">对比模型：</span>
                {/*
                  原来是 mode="multiple" 的下拉 + maxCount。摊开成多选框的理由和表单页一样：
                  只有 5 个选项，勾选比「点开、勾、点空白收起」少两步。max=3 是布局上限，
                  再多一列每列就窄到读不下一句完整的回答了。
                */}
                <CheckboxGroup value={models} onChange={setModels} options={modelOptions} max={3} />
                <Tooltip>
                  <TooltipTrigger asChild>
                    <Badge variant="outline" className="cursor-help">
                      来自部署服务
                    </Badge>
                  </TooltipTrigger>
                  <TooltipContent>部署页里 RUNNING 的服务会自动出现在这里</TooltipContent>
                </Tooltip>
              </div>
            ) : (
              <SelectInput value={single} onChange={setSingle} options={modelOptions} className="w-80" />
            )}
          </div>

          <div className="flex min-h-0 flex-1 gap-3.5">
            {compare ? (
              models.length ? (
                models.map((m, i) => (
                  <ChatColumn key={m} model={m} tone={COLUMN_TONES[i % COLUMN_TONES.length]} />
                ))
              ) : (
                <div className="text-muted-foreground m-auto text-[13px]">选择要对比的模型</div>
              )
            ) : (
              <ChatColumn model={single} tone="bg-primary" />
            )}
          </div>

          <div className="mt-3.5">
            <Composer state={composer} />
          </div>
        </div>
      </div>
    </>
  );
}
