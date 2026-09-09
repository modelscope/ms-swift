import { Segmented, Slider, Switch, message } from 'antd';
import { BulbOutlined, RobotOutlined, UndoOutlined } from '@ant-design/icons';
import { PageHeader } from '@/components/PageHeader';
import { TIPS, visibleTips } from '@/config/tips';
import { useSettings } from '@/settings/SettingsContext';
import { columnStyle, neutral } from '@/theme/theme';

/**
 * 偏好设置。
 *
 * 走的是「一行一件事」的排法，不用 antd Form——这些开关彼此独立，
 * 套一层表单只会多出提交按钮和校验状态两样不需要的东西。改完即生效。
 */
export function SettingsPage() {
  const { settings, update, reset, tipsDismissed } = useSettings();
  const poolSize = visibleTips(settings.tipsBeginnerOnly).length;

  return (
    <div style={{ paddingBottom: 40 }}>
      <PageHeader
        title="偏好设置"
        desc="只影响这台机器上的这个浏览器，改完立即生效，不用保存"
        extra={
          <span
            className="ghost-icon"
            onClick={() => {
              reset();
              message.success('已恢复默认设置');
            }}
            style={{
              display: 'inline-flex',
              alignItems: 'center',
              gap: 6,
              height: 30,
              padding: '0 12px',
              borderRadius: 999,
              fontSize: 12.5,
              cursor: 'pointer',
              color: neutral.textSecondary,
            }}
          >
            <UndoOutlined /> 恢复默认
          </span>
        }
      />

      <div style={columnStyle}>
        <Group icon={<BulbOutlined />} title="底部提示" desc="等训练的时候顺手看两眼，都是真事">
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
              onChange={(v) => update('tipsEnabled', v)}
            />
          </Row>

          <Row label="只显示通俗的" hint="关掉涉及具体算法和数值细节的条目，避免照着用出问题">
            <Switch
              checked={settings.tipsBeginnerOnly}
              disabled={!settings.tipsEnabled}
              onChange={(v) => update('tipsBeginnerOnly', v)}
            />
          </Row>

          <Row label="轮换间隔" hint={`每 ${settings.tipsInterval} 秒换一条`}>
            <div style={{ width: 200 }}>
              <Slider
                min={5}
                max={40}
                step={1}
                disabled={!settings.tipsEnabled}
                value={settings.tipsInterval}
                onChange={(v) => update('tipsInterval', v)}
                tooltip={{ formatter: (v) => `${v} 秒` }}
              />
            </div>
          </Row>
        </Group>

        <Group icon={<RobotOutlined />} title="AI 协助" desc="节点上和整张流程上的问 AI 入口">
          <Row label="启用 AI 入口" hint="关掉后节点标题栏和工具条上的 AI 按钮都不显示">
            <Switch checked={settings.aiEnabled} onChange={(v) => update('aiEnabled', v)} />
          </Row>

          <Row
            label="AI 改动需要我确认"
            hint="建议保持开启。图形是这份流程的唯一事实来源，AI 直接改会让「谁把这个参数调掉的」变成查不出来的事"
          >
            <Switch
              checked={settings.aiConfirmBeforeApply}
              disabled={!settings.aiEnabled}
              onChange={(v) => {
                update('aiConfirmBeforeApply', v);
                if (!v) message.warning('已关闭确认：AI 之后会直接改图，改动仍会记在对话里');
              }}
            />
          </Row>
        </Group>

        <Group title="画布行为" desc="编排画布上的一些默认动作">
          <Row label="节点跑完自动展开输出" hint="只对单节点运行生效，整图运行时不会一个个弹出来">
            <Switch
              checked={settings.autoOpenOutput}
              onChange={(v) => update('autoOpenOutput', v)}
            />
          </Row>
        </Group>

        <Group title="界面" desc="以下几项还没接上，先占位">
          <Row label="主题" hint="深色 chrome 是固定的，这里只切内容区">
            <Segmented
              size="small"
              disabled
              value="light"
              options={[
                { label: '浅色', value: 'light' },
                { label: '深色', value: 'dark' },
                { label: '跟随系统', value: 'auto' },
              ]}
            />
          </Row>
        </Group>
      </div>
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
  icon?: React.ReactNode;
  title: string;
  desc: string;
  children: React.ReactNode;
}) {
  return (
    <section style={{ marginBottom: 34 }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 7, marginBottom: 2 }}>
        {icon && <span style={{ color: neutral.textTertiary, fontSize: 13 }}>{icon}</span>}
        <h3 style={{ margin: 0, fontSize: 14, fontWeight: 600, color: neutral.text }}>{title}</h3>
      </div>
      <div style={{ fontSize: 12.5, color: neutral.textTertiary, marginBottom: 10 }}>{desc}</div>
      <div style={{ borderTop: `1px solid ${neutral.borderLight}` }}>{children}</div>
    </section>
  );
}

/**
 * 一行设置。左边名字 + 一句说明，右边控件。
 * 说明这一行是有意留出的：这些开关里有几个（尤其 AI 确认）关掉是有代价的，
 * 代价必须写在旁边，而不是等出了事再解释。
 */
function Row({
  label,
  hint,
  children,
}: {
  label: string;
  hint?: string;
  children: React.ReactNode;
}) {
  return (
    <div
      style={{
        display: 'flex',
        alignItems: 'center',
        gap: 20,
        padding: '13px 2px',
        borderBottom: `1px solid ${neutral.borderLight}`,
      }}
    >
      <div style={{ flex: 1, minWidth: 0 }}>
        <div style={{ fontSize: 13.5, color: neutral.text }}>{label}</div>
        {hint && (
          <div style={{ fontSize: 12, color: neutral.textTertiary, marginTop: 2, lineHeight: '17px' }}>
            {hint}
          </div>
        )}
      </div>
      <div style={{ flex: 'none' }}>{children}</div>
    </div>
  );
}
