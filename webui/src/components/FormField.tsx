import type { ReactNode } from 'react';
import { CircleHelp } from 'lucide-react';
import { cn } from 'cn';
import { Input } from '@/components/ui/input';
import { Checkbox } from '@/components/ui/checkbox';
import { ToggleGroup, ToggleGroupItem } from '@/components/ui/toggle-group';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { Tooltip, TooltipContent, TooltipTrigger } from '@/components/ui/tooltip';

/**
 * 表单控件层。取代 antd 的 Form / Form.Item / InputNumber / Radio.Group / Checkbox.Group。
 *
 * 为什么不用 shadcn 官方的表单方案（react-hook-form + zod）：这个项目的表单实测
 * 一处校验都没有——rules= 0 处、onFinish= 0 处、Form.useForm 0 处，那 19 个 Form.Item
 * 纯粹当「标签 + 控件」的排版壳在用，状态各页面自己 useState 管着。引一套表单库
 * 进来只会多出一层 register/control 的转接，一行校验也换不来。
 *
 * 所以这里只做两件事：统一「标签 + 说明 + 控件」的排法，以及把 Radix 那几个
 * 需要多写五六行的控件收成一行能用的形态。
 */

/** 一个字段：标签在上，控件在下，说明挂在标签后面的问号上 */
export function Field({
  label,
  hint,
  children,
  className,
}: {
  label: ReactNode;
  /** 悬浮说明。写「为什么」和「代价」，不重复标签已经说过的话 */
  hint?: ReactNode;
  children: ReactNode;
  className?: string;
}) {
  return (
    /*
      用 <label> 包住控件，靠隐式关联替代 htmlFor + id：
      不引表单库就没有地方统一发 id，手工发 id 又要每个调用方传一遍。
      隐式关联的效果是点标签文字会激活里面第一个控件，正好是想要的行为。
    */
    <label className={cn('block min-w-0', className)}>
      <span className="mb-1.5 flex items-center gap-1.5">
        <span className="text-foreground text-[13px] font-medium">{label}</span>
        {hint && (
          <Tooltip>
            <TooltipTrigger asChild>
              <span className="text-muted-foreground/70 hover:text-foreground inline-flex cursor-help">
                <CircleHelp size={12.5} />
              </span>
            </TooltipTrigger>
            <TooltipContent className="max-w-[280px]">{hint}</TooltipContent>
          </Tooltip>
        )}
      </span>
      {children}
    </label>
  );
}

/** 几个字段并排。窄了自动折行，不靠断点 */
export function FieldRow({ children }: { children: ReactNode }) {
  return <div className="flex flex-wrap gap-4 [&>*]:min-w-40 [&>*]:flex-1">{children}</div>;
}

/** 字段之间的竖向间距。一个分区里所有字段共用同一个节奏 */
export function FieldStack({ children }: { children: ReactNode }) {
  return <div className="flex flex-col gap-3.5">{children}</div>;
}

/**
 * 数字输入。取代 antd InputNumber。
 *
 * 直接用原生 type=number，不自己造加减按钮：原生的上下箭头、滚轮、方向键、
 * step 对齐全都是白拿的，自己实现一遍只会漏掉其中几样。
 *
 * 只支持受控用法（value + onChange），不支持 defaultValue：value 一旦给了 ''
 * 兜底，React 就认定这是受控组件，defaultValue 会被静默忽略，框里显示成空的。
 * 清空输入框时 onChange 给的是 undefined 而不是 0——「没填」和「填了 0」
 * 对命令行参数是两件事。
 */
export function NumberInput({
  value,
  onChange,
  className,
  ...rest
}: Omit<React.ComponentProps<typeof Input>, 'value' | 'onChange' | 'type' | 'defaultValue'> & {
  value?: number;
  onChange?: (v: number | undefined) => void;
}) {
  return (
    <Input
      type="number"
      /* tabular：数字不等宽的话，改一位数字整个框里的内容都会横向抖一下 */
      className={cn('tabular', className)}
      value={value ?? ''}
      onChange={(e) => {
        /* valueAsNumber 而不是 parseFloat(e.target.value)：清空时它给 NaN，
           能跟「输入了 0」区分开，parseFloat('') 也是 NaN 但还要自己处理 '0x' 之类 */
        const n = e.currentTarget.valueAsNumber;
        onChange?.(Number.isNaN(n) ? undefined : n);
      }}
      {...rest}
    />
  );
}

export interface Option {
  value: string;
  label: ReactNode;
}

/**
 * 下拉选择。取代 antd Select。
 *
 * 没有做 showSearch：Radix 的 Select 自带按键跳转（打头几个字母直接跳到对应项），
 * 现在最长的一组选项是 5 个模型，搜索框反而多一次点击。选项真多到要搜的那天，
 * 换成 Command + Popover 的组合，改动只在这个函数里。
 */
export function SelectInput({
  value,
  onChange,
  options,
  placeholder,
  className,
  disabled,
}: {
  value?: string;
  onChange?: (v: string) => void;
  options: Option[];
  placeholder?: string;
  className?: string;
  disabled?: boolean;
}) {
  return (
    <Select value={value} onValueChange={onChange} disabled={disabled}>
      <SelectTrigger className={cn('w-full', className)}>
        <SelectValue placeholder={placeholder} />
      </SelectTrigger>
      <SelectContent>
        {options.map((o) => (
          <SelectItem key={o.value} value={o.value}>
            {o.label}
          </SelectItem>
        ))}
      </SelectContent>
    </Select>
  );
}

/**
 * 分段选择器。取代 antd 的 Radio.Group optionType="button" 和 Segmented。
 *
 * 那个 `if (v)` 不能省：Radix 的单选 ToggleGroup 允许再点一次取消选中，
 * 这时它回传空串。单选语义下「什么都没选」是非法状态，会让预览里的命令少一个参数。
 */
export function SegmentedControl({
  value,
  onChange,
  options,
  size = 'sm',
  block,
  disabled,
}: {
  value: string;
  onChange?: (v: string) => void;
  options: Option[];
  size?: 'sm' | 'default';
  /** 撑满一行、每段等宽。当它是一块区域的页签而不是一个字段的取值时用 */
  block?: boolean;
  disabled?: boolean;
}) {
  return (
    <ToggleGroup
      type="single"
      variant="outline"
      size={size}
      value={value}
      disabled={disabled}
      className={cn(block && 'w-full')}
      onValueChange={(v) => {
        if (v) onChange?.(v);
      }}
    >
      {options.map((o) => (
        <ToggleGroupItem
          key={o.value}
          value={o.value}
          className={cn('text-[13px]', block && 'flex-1')}
        >
          {o.label}
        </ToggleGroupItem>
      ))}
    </ToggleGroup>
  );
}

/**
 * 多选组。取代 antd Checkbox.Group，也用来替掉「选项不多的多选下拉」。
 *
 * max 用来表达「布局撑不下更多了」这类硬上限（比如对话页最多并排 3 列），
 * 到顶之后没勾的那几项变灰而不是点了没反应——后者会让人以为界面卡住了。
 */
export function CheckboxGroup({
  value,
  onChange,
  options,
  max,
}: {
  value: string[];
  onChange: (v: string[]) => void;
  options: Option[];
  max?: number;
}) {
  const atMax = max !== undefined && value.length >= max;
  return (
    <div className="flex flex-wrap gap-x-6 gap-y-2.5">
      {options.map((o) => {
        const checked = value.includes(o.value);
        const locked = !checked && atMax;
        return (
        <label
          key={o.value}
          className={cn(
            'text-foreground flex items-center gap-2 text-[13px]',
            locked ? 'text-muted-foreground/60 cursor-not-allowed' : 'cursor-pointer',
          )}
        >
          <Checkbox
            checked={checked}
            disabled={locked}
            onCheckedChange={(on) =>
              /* 保持 options 的顺序而不是点击顺序：这个数组会拼进命令行，
                 顺序跳来跳去的话预览区每点一下都在动 */
              onChange(
                on === true
                  ? options.filter((x) => x.value === o.value || value.includes(x.value)).map((x) => x.value)
                  : value.filter((v) => v !== o.value),
              )
            }
          />
          {o.label}
        </label>
        );
      })}
    </div>
  );
}
