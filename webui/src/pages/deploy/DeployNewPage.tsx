import { useMemo, useState } from 'react';
import { Info } from 'lucide-react';
import { ConfigFormShell } from '@/components/ConfigFormShell';
import { Field, FieldStack, NumberInput, SegmentedControl, SelectInput } from '@/components/FormField';
import { MODULES } from '@/theme/modules';
import { availableModels } from '@/mock/data';

/**
 * 新建部署。部署只走本机 local（endpoint 固定 localhost），
 * 端口由进程自选后写回 runtime.json，这里只填一个「期望端口」作提示。
 */
export function DeployNewPage() {
  const [model, setModel] = useState(availableModels[0]);
  const [engine, setEngine] = useState('vllm');
  const [port, setPort] = useState<number | undefined>(0);

  const preview = useMemo(
    () =>
      [
        '#!/usr/bin/env bash',
        '# run.sh（部署，常驻服务，仅本机 local）',
        'cd "$(dirname "$0")"',
        'swift deploy \\',
        `    --model ${model} \\`,
        `    --infer_backend ${engine} \\`,
        port ? `    --port ${port} \\` : '    --port 0 \\  # 0=自选空闲端口，写回 runtime.json',
        '    >> output.log 2>&1',
        'echo $? > exit_code',
      ].join('\n'),
    [model, engine, port],
  );

  return (
    <ConfigFormShell
      module={MODULES.deploy}
      title="新建部署"
      desc="把模型拉起成常驻推理服务"
      preview={preview}
      sections={[
        {
          title: '服务配置',
          content: (
            <FieldStack>
              <Field label="模型">
                <SelectInput
                  value={model}
                  onChange={setModel}
                  options={availableModels.map((m) => ({ value: m, label: m }))}
                />
              </Field>
              <Field label="推理引擎">
                <SegmentedControl
                  value={engine}
                  onChange={setEngine}
                  options={[
                    { value: 'vllm', label: 'vLLM' },
                    { value: 'lmdeploy', label: 'LMDeploy' },
                    { value: 'pt', label: 'PyTorch' },
                  ]}
                />
              </Field>
              <Field
                label="期望端口"
                hint="留 0 表示由进程自选空闲端口，避免多个部署抢同一端口的竞态"
              >
                <NumberInput
                  className="w-50"
                  value={port}
                  onChange={setPort}
                  min={0}
                  max={65535}
                />
              </Field>
            </FieldStack>
          ),
        },
        {
          title: '说明',
          content: (
            <div className="text-muted-foreground flex items-start gap-2 text-[13px] leading-5">
              <Info size={14} className="mt-0.5 flex-none" />
              部署任务起来后一直处于运行中，没有「完成」态；停止即视为正常结束。服务地址会在进程绑定端口后回填到列表。
            </div>
          ),
        },
      ]}
    />
  );
}
