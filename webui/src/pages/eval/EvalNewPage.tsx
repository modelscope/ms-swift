import { useMemo, useState } from 'react';
import { Checkbox, Form, Select } from 'antd';
import { ConfigFormShell } from '@/components/ConfigFormShell';
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

  return (
    <ConfigFormShell
      module={MODULES.eval}
      title="新建评测"
      desc="选择被测 checkpoint 与评测集"
      preview={preview}
      warning="评测依赖 evalscope，请确认已安装（否则任务会以 exit_code=127 失败）"
      sections={[
        {
          title: '被测对象',
          content: (
            <Form layout="vertical">
              <Form.Item label="checkpoint" tooltip="从各训练任务产出的 ckpt 中选择">
                <Select
                  value={ckpt}
                  onChange={setCkpt}
                  options={checkpoints.map((c) => ({
                    value: `${c.fromTask}/${c.name}`,
                    label: `${c.name} · step ${c.step} · 来自 ${c.fromTask}`,
                  }))}
                />
              </Form.Item>
            </Form>
          ),
        },
        {
          title: '评测集',
          content: (
            <Checkbox.Group
              value={sets}
              onChange={(v) => setSets(v as string[])}
              options={evalDatasets.map((d) => ({ label: d, value: d }))}
            />
          ),
        },
      ]}
    />
  );
}
