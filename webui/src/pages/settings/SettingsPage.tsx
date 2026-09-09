import type { ReactNode } from 'react';
import { Lightbulb, Bot, Undo2 } from 'lucide-react';
import { toast } from 'sonner';
import { PageHeader } from '@/components/PageHeader';
import { Column } from '@/components/Column';
import { Button } from '@/components/ui/button';
import { Switch } from '@/components/ui/switch';
import { Slider } from '@/components/ui/slider';
import { ToggleGroup, ToggleGroupItem } from '@/components/ui/toggle-group';
import { TIPS, visibleTips } from '@/config/tips';
import { useSettings } from '@/settings/SettingsContext';

/**
 * 偏好设置。
 *
 * 走的是「一行一件事」的排法，不套表单——这些开关彼此独立，
 * 套一层表单只会多出提交按钮和校验状态两样不需要的东西。改完即生效。
 */
export function SettingsPage() {
  const { settings, update, reset, tipsDismissed } = useSettings();
  const poolSize = visibleTips(settings.tipsBeginnerOnly).length;

  return (
    <div className="pb-10">
      <PageHeader
        title="偏好设置"
        desc="只影响这台机器上的这个浏览器，改完立即生效，不用保存"
        extra={
          <Button
            variant="ghost"
            size="sm"
            onClick={() => {
              reset();
              toast.success('已恢复默认设置');
            }}
          >
            <Undo2 /> 恢复默认
          </Button>
        }
      />

      <Column>
        <Group icon={<Lightbulb size={13} />} title="底部提示" desc="等训练的时候顺手看两眼，都是真事">
          <Row
            label="显示底部提示条"
            hint={
              tipsDismissed && settings.tipsEnabled
                ? '当前已被叉掉，刷新页面后会回来。想永久关闭就把这个开关关掉'
                : `当前文案池 ${poolSize} 条，共 ${TIPS.length} 条`
            }
          >
            <Switch
              checked={settings.tipsEnabled}
              onCheckedChange={(v) => update('tipsEnabled', v)}
            />
          </Row>

          <Row label="只显示通俗的" hint="关掉涉及具体算法和数值细节的条目，避免照着用出问题">
            <Switch
              checked={settings.tipsBeginnerOnly}
              disabled={!settings.tipsEnabled}
              onCheckedChange={(v) => update('tipsBeginnerOnly', v)}
            />
          </Row>

          <Row label="轮换间隔" hint={`每 ${settings.tipsInterval} 秒换一条`}>
            {/*
              Radix 的 Slider 是多游标模型，value 永远是数组，所以这里进出都要拆装一次。
              当前值不显示在游标上而是写进上面那句 hint：拖动时提示气泡会挡住相邻的行，
              而这一页的说明文字本来就在左边，读起来是连贯的。
            */}
            <Slider
              className="w-50"
              min={5}
              max={40}
              step={1}
              disabled={!settings.tipsEnabled}
              value={[settings.tipsInterval]}
              onValueChange={([v]) => update('tipsInterval', v)}
            />
          </Row>
        </Group>

        <Group icon={<Bot size={13} />} title="AI 协助" desc="节点上和整张流程上的问 AI 入口">
          <Row label="启用 AI 入口" hint="关掉后节点标题栏和工具条上的 AI 按钮都不显示">
            <Switch checked={settings.aiEnabled} onCheckedChange={(v) => update('aiEnabled', v)} />
          </Row>

          <Row
            label="AI 改动需要我确认"
            hint="建议保持开启。图形是这份流程的唯一事实来源，AI 直接改会让「谁把这个参数调掉的」变成查不出来的事"
          >
            <Switch
              checked={settings.aiConfirmBeforeApply}
              disabled={!settings.aiEnabled}
              onCheckedChange={(v) => {
                update('aiConfirmBeforeApply', v);
                if (!v) toast.warning('已关闭确认：AI 之后会直接改图，改动仍会记在对话里');
              }}
            />
          </Row>
        </Group>

        <Group title="画布行为" desc="编排画布上的一些默认动作">
          <Row label="节点跑完自动展开输出" hint="只对单节点运行生效，整图运行时不会一个个弹出来">
            <Switch
              checked={settings.autoOpenOutput}
              onCheckedChange={(v) => update('autoOpenOutput', v)}
            />
          </Row>
        </Group>

        <Group title="界面" desc="以下几项还没接上，先占位">
          <Row label="主题" hint="深色 chrome 是固定的，这里只切内容区">
            {/* type="single" 的 ToggleGroup 就是 antd Segmented 的等价物 */}
            <ToggleGroup type="single" variant="outline" size="sm" disabled value="light">
              <ToggleGroupItem value="light">浅色</ToggleGroupItem>
              <ToggleGroupItem value="dark">深色</ToggleGroupItem>
              <ToggleGroupItem value="auto">跟随系统</ToggleGroupItem>
            </ToggleGroup>
          </Row>
        </Group>
      </Column>
    </div>
  );
}

/** 一组设置。标题 + 说明 + 若干行 */
function Group({
  icon,
  title,
  desc,
  children,
}: {
  icon?: ReactNode;
  title: string;
  desc: string;
  children: ReactNode;
}) {
  return (
    <section className="mb-8">
      <div className="mb-0.5 flex items-center gap-[7px]">
        {icon && <span className="text-muted-foreground">{icon}</span>}
        <h3 className="text-foreground text-sm font-semibold">{title}</h3>
      </div>
      <div className="text-muted-foreground mb-2.5 text-[12.5px]">{desc}</div>
      <div className="border-border/60 border-t">{children}</div>
    </section>
  );
}

/**
 * 一行设置。左边名字 + 一句说明，右边控件。
 * 说明这一行是有意留出的：这些开关里有几个（尤其 AI 确认）关掉是有代价的，
 * 代价必须写在旁边，而不是等出了事再解释。
 */
function Row({ label, hint, children }: { label: string; hint?: string; children: ReactNode }) {
  return (
    <div className="border-border/60 flex items-center gap-5 border-b px-0.5 py-3.5">
      <div className="min-w-0 flex-1">
        <div className="text-foreground text-[13.5px]">{label}</div>
        {hint && (
          <div className="text-muted-foreground mt-0.5 text-xs leading-[17px]">{hint}</div>
        )}
      </div>
      <div className="flex-none">{children}</div>
    </div>
  );
}
