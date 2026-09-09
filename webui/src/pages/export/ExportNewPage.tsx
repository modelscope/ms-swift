import { useMemo, useState } from 'react';
import { ConfigFormShell } from '@/components/ConfigFormShell';
import type { ConfigField } from '@/components/configAssist';
import { OFF, ON } from '@/components/configAssist';
import { Field, FieldStack, SelectInput } from '@/components/FormField';
import { Input } from '@/components/ui/input';
import { Switch } from '@/components/ui/switch';
import { MODULES } from '@/theme/modules';
import { checkpoints } from '@/mock/data';

/** 新建导出：LoRA 合并 / 量化 / 推送 Hub 三选项拼在一条命令里 */
export function ExportNewPage() {
  const [ckpt, setCkpt] = useState(`${checkpoints[0].fromTask}/${checkpoints[0].name}`);
  const [mergeLora, setMergeLora] = useState(true);
  const [quant, setQuant] = useState('none');
  const [push, setPush] = useState(false);

  const preview = useMemo(() => {
    const args = [
      'swift export',
      `    --model ${ckpt}`,
      mergeLora ? '    --merge_lora true' : null,
      /* awq / gptq 都是 int4，bnb 是 int8——位宽跟着方法走，不单独给一个选项 */
      quant !== 'none' ? `    --quant_method ${quant} --quant_bits ${quant === 'bnb' ? 8 : 4}` : null,
      push ? '    --push_to_hub true --hub_model_id my-org/my-model' : null,
      '    --output_dir outputs',
    ]
      .filter(Boolean)
      .join(' \\\n');
    return [
      '#!/usr/bin/env bash',
      '# run.sh（导出）',
      'cd "$(dirname "$0")"',
      `${args} >> output.log 2>&1`,
      'echo $? > exit_code',
    ].join('\n');
  }, [ckpt, mergeLora, quant, push]);

  /* 交给 AI 助手的字段。源 ckpt 只读，其余三项之间有顺序约束，正是助手要盯的 */
  const fields: ConfigField[] = [
    { key: 'model', label: '源 checkpoint', value: ckpt },
    { key: 'merge_lora', label: '合并 LoRA', value: mergeLora ? ON : OFF, apply: (v) => setMergeLora(v === ON) },
    { key: 'quant_method', label: '量化方式', value: quant, apply: setQuant },
    { key: 'push_to_hub', label: '推送到 Hub', value: push ? ON : OFF, apply: (v) => setPush(v === ON) },
  ];

  return (
    <ConfigFormShell
      module={MODULES.export}
      title="新建导出"
      desc="LoRA 合并、量化、推送到 Hub"
      preview={preview}
      fields={fields}
      sections={[
        {
          title: '源与操作',
          content: (
            <FieldStack>
              <Field label="源 checkpoint">
                <SelectInput
                  value={ckpt}
                  onChange={setCkpt}
                  options={checkpoints.map((c) => ({
                    value: `${c.fromTask}/${c.name}`,
                    label: `${c.name} · 来自 ${c.fromTask}`,
                  }))}
                />
              </Field>
              <Field label="合并 LoRA 权重">
                <Switch checked={mergeLora} onCheckedChange={setMergeLora} />
              </Field>
              <Field label="量化方式">
                <SelectInput
                  value={quant}
                  onChange={setQuant}
                  options={[
                    { value: 'none', label: '不量化' },
                    { value: 'awq', label: 'AWQ (int4)' },
                    { value: 'gptq', label: 'GPTQ (int4)' },
                    { value: 'bnb', label: 'BitsAndBytes (int8)' },
                  ]}
                />
              </Field>
            </FieldStack>
          ),
        },
        {
          title: '推送到 Hub',
          content: (
            <FieldStack>
              <Field label="推送到 ModelScope Hub">
                <Switch checked={push} onCheckedChange={setPush} />
              </Field>
              {push && (
                <Field label="hub_model_id">
                  <Input placeholder="my-org/my-model" defaultValue="my-org/my-model" />
                </Field>
              )}
            </FieldStack>
          ),
        },
      ]}
    />
  );
}
