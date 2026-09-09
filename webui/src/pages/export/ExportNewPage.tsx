import { useMemo, useState } from 'react';
import { Form, Input, Select, Switch } from 'antd';
import { ConfigFormShell } from '@/components/ConfigFormShell';
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
      quant !== 'none' ? `    --quant_method ${quant} --quant_bits ${quant === 'awq' ? 4 : 4}` : null,
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

  return (
    <ConfigFormShell
      module={MODULES.export}
      title="新建导出"
      desc="LoRA 合并、量化、推送到 Hub"
      preview={preview}
      sections={[
        {
          title: '源与操作',
          content: (
            <Form layout="vertical">
              <Form.Item label="源 checkpoint">
                <Select
                  value={ckpt}
                  onChange={setCkpt}
                  options={checkpoints.map((c) => ({
                    value: `${c.fromTask}/${c.name}`,
                    label: `${c.name} · 来自 ${c.fromTask}`,
                  }))}
                />
              </Form.Item>
              <Form.Item label="合并 LoRA 权重">
                <Switch checked={mergeLora} onChange={setMergeLora} />
              </Form.Item>
              <Form.Item label="量化方式">
                <Select
                  value={quant}
                  onChange={setQuant}
                  options={[
                    { value: 'none', label: '不量化' },
                    { value: 'awq', label: 'AWQ (int4)' },
                    { value: 'gptq', label: 'GPTQ (int4)' },
                    { value: 'bnb', label: 'BitsAndBytes (int8)' },
                  ]}
                />
              </Form.Item>
            </Form>
          ),
        },
        {
          title: '推送到 Hub',
          content: (
            <Form layout="vertical">
              <Form.Item label="推送到 ModelScope Hub">
                <Switch checked={push} onChange={setPush} />
              </Form.Item>
              {push && (
                <Form.Item label="hub_model_id">
                  <Input placeholder="my-org/my-model" defaultValue="my-org/my-model" />
                </Form.Item>
              )}
            </Form>
          ),
        },
      ]}
    />
  );
}
