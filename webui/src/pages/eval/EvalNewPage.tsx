import { useMemo, useState } from 'react';
import { ConfigFormShell } from '@/components/ConfigFormShell';
import type { ConfigField } from '@/components/configAssist';
import { CheckboxGroup, Field, FieldStack, SelectInput } from '@/components/FormField';
import { MODULES } from '@/theme/modules';
import { checkpoints, evalDatasets } from '@/mock/data';

/** 新建评测。被测对象来自 checkpoint 下拉；评测集多选 */
export function EvalNewPage() {
  const [ckpt, setCkpt] = useState(`${checkpoints[0].fromTask}/${checkpoints[0].name}`);
  const [sets, setSets] = useState<string[]>(['gsm8k', 'ceval']);

  const preview = useMemo(
    () =>
      [
        '#!/usr/bin/env bash',
        '# run.sh（评测）',
        'cd "$(dirname "$0")"',
        'swift eval \\',
        `    --model ${ckpt} \\`,
        `    --eval_dataset ${sets.join(' ')} \\`,
        '    --eval_output_dir outputs >> output.log 2>&1',
        'echo $? > exit_code',
      ].join('\n'),
    [ckpt, sets],
  );

  /* 交给 AI 助手的字段。被测对象只读——要测哪个 ckpt 只有用户知道 */
  const fields: ConfigField[] = [
    { key: 'model', label: '被测 checkpoint', value: ckpt },
    {
      key: 'eval_dataset',
      label: '评测集',
      value: sets.join(' '),
      apply: (v) => setSets(v.split(' ').filter(Boolean)),
    },
  ];

  return (
    <ConfigFormShell
      module={MODULES.eval}
      title="新建评测"
      desc="选择被测 checkpoint 与评测集"
      preview={preview}
      fields={fields}
      warning="评测依赖 evalscope，请确认已安装（否则任务会以 exit_code=127 失败）"
      sections={[
        {
          title: '被测对象',
          content: (
            <FieldStack>
              <Field label="checkpoint" hint="从各训练任务产出的 ckpt 中选择">
                <SelectInput
                  value={ckpt}
                  onChange={setCkpt}
                  options={checkpoints.map((c) => ({
                    value: `${c.fromTask}/${c.name}`,
                    label: `${c.name} · step ${c.step} · 来自 ${c.fromTask}`,
                  }))}
                />
              </Field>
            </FieldStack>
          ),
        },
        {
          title: '评测集',
          content: (
            <CheckboxGroup
              value={sets}
              onChange={setSets}
              options={evalDatasets.map((d) => ({ value: d, label: d }))}
            />
          ),
        },
      ]}
    />
  );
}
