import type { ReactNode } from 'react';
import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { ArrowLeft, Bot, CirclePlay, Info, Save, TriangleAlert } from 'lucide-react';
import { toast } from 'sonner';
import { Hint } from './Hint';
import { PageHeader } from './PageHeader';
import { Panel } from './TaskDetailShell';
import { Button } from './ui/button';
import { ConfigAssistant } from './ConfigAssistant';
import type { ConfigField } from './configAssist';
import { useSettings } from '@/settings/SettingsContext';
import type { ModuleMeta } from '@/theme/modules';

/**
 * 新建任务页的外壳：左边表单分区，右边贴一个「即将执行什么」的预览。
 * 预览区展示的就是最终会落到任务目录里的 run.sh，让用户在提交前看清要跑什么，
 * 也顺便解释了任务与文件目录的对应关系。
 *
 * 分区用的是详情页那个 Panel，不另造一种卡片：同一个应用里「一块内容」
 * 只应该有一种长相，否则新建页和详情页看着像两个产品。
 *
 * AI 助手也接在这里，而不是四个新建页各写一遍：四个页面的差异只在「有哪些字段」，
 * 开抽屉、接偏好设置、把提议写回表单这三件事是一模一样的，各写一遍就是四份要同步改的代码。
 */
export function ConfigFormShell({
  module,
  title,
  desc,
  sections,
  preview,
  warning,
  fields,
}: {
  module: ModuleMeta;
  title: string;
  desc?: string;
  sections: Array<{ title: string; extra?: ReactNode; content: ReactNode }>;
  preview: string;
  warning?: string;
  /**
   * 交给 AI 助手的字段。每个字段自带 setter，所以这里不用再接一个 onApply。
   * 必填（可以为空数组）：新增一个新建页时得明确表态它的字段给不给 AI 看，
   * 而不是忘了传就默默少了一个功能。
   */
  fields: ConfigField[];
}) {
  const navigate = useNavigate();
  const { settings } = useSettings();
  const [assistantOpen, setAssistantOpen] = useState(false);

  return (
    <>
      <PageHeader
        module={module}
        title={title}
        desc={desc}
        wide
        back={
          <button
            type="button"
            onClick={() => navigate(module.path)}
            className="text-muted-foreground hover:text-foreground mb-0.5 inline-flex cursor-pointer items-center gap-1 text-[12.5px] transition-colors"
          >
            <ArrowLeft size={12} /> 返回{module.label}列表
          </button>
        }
        extra={
          <>
            {/* 助手摆在两个提交按钮前面：它是提交之前的事，而且得比「提交运行」低一级 */}
            {settings.aiEnabled && fields.length > 0 && (
              <Hint title="问这份配置：字段怎么填、有没有坑">
                <Button
                  variant={assistantOpen ? 'secondary' : 'ghost'}
                  onClick={() => setAssistantOpen(true)}
                >
                  <Bot /> AI 助手
                </Button>
              </Hint>
            )}
            <Button variant="secondary" onClick={() => toast.success('已存为草稿（示意）')}>
              <Save /> 存为草稿
            </Button>
            <Button onClick={() => toast.success('已提交（示意，未接后端）')}>
              <CirclePlay /> 提交运行
            </Button>
          </>
        }
      />

      {/* items-start：右边预览要 sticky，父容器一拉伸就粘不住了 */}
      <div className="flex items-start gap-5 px-6 pt-1 pb-8">
        <div className="flex min-w-0 flex-1 flex-col gap-4">
          {warning && (
            <div className="border-state-paused/25 bg-state-paused/8 text-foreground flex items-start gap-2.5 rounded-md border px-3.5 py-2.5 text-[13px] leading-5">
              <TriangleAlert size={14} className="text-state-paused mt-0.5 flex-none" />
              {warning}
            </div>
          )}
          {sections.map((s) => (
            <Panel key={s.title} title={s.title} extra={s.extra}>
              {s.content}
            </Panel>
          ))}
        </div>

        <div className="sticky top-4 w-[420px] flex-none">
          <Panel title="将要执行">
            {/*
              这里 whitespace-pre-wrap 而不是横向滚动：预览是「读」的，
              命令太长时折行看得见全部内容比拖横条更有用；要复制的是任务目录里
              那份真的 run.sh，不是这段预览。
            */}
            <div className="bg-muted text-foreground max-h-[420px] overflow-auto rounded-md p-3 font-mono text-xs leading-[19px] break-all whitespace-pre-wrap">
              {preview}
            </div>
            <div className="text-muted-foreground mt-2.5 flex items-start gap-2 text-xs leading-[18px]">
              <Info size={13} className="mt-0.5 flex-none" />
              提交后会在任务目录下生成 run.sh 与配置文件，由它负责重定向日志、执行命令、并在结束时写出退出码。
            </div>
          </Panel>
        </div>
      </div>

      <ConfigAssistant
        open={assistantOpen}
        onClose={() => setAssistantOpen(false)}
        module={module.key}
        fields={fields}
        confirmBeforeApply={settings.aiConfirmBeforeApply}
      />
    </>
  );
}
