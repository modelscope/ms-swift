import { useMemo, useState } from 'react';
import { Alert, Form, InputNumber, Radio, Select } from 'antd';
import { ConfigFormShell } from '@/components/ConfigFormShell';
import { MODULES } from '@/theme/modules';
import { availableModels } from '@/mock/data';

/**
 * 新建部署。部署只走本机 local（endpoint 固定 localhost），
 * 端口由进程自选后写回 runtime.json，这里只填一个「期望端口」作提示。
 */
export function DeployNewPage() {
  const [model, setModel] = useState(availableModels[0]);
  const [engine, setEngine] = useState('vllm');
  const [port, setPort] = useState(0);

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
            <Form layout="vertical">
              <Form.Item label="模型">
                <Select value={model} onChange={setModel} options={availableModels.map((m) => ({ value: m, label: m }))} showSearch />
              </Form.Item>
              <Form.Item label="推理引擎">
                <Radio.Group
                  value={engine}
                  onChange={(e) => setEngine(e.target.value)}
                  optionType="button"
                  options={[
                    { value: 'vllm', label: 'vLLM' },
                    { value: 'lmdeploy', label: 'LMDeploy' },
                    { value: 'pt', label: 'PyTorch' },
                  ]}
                />
              </Form.Item>
              <Form.Item
                label="期望端口"
                tooltip="留 0 表示由进程自选空闲端口，避免多个部署抢同一端口的竞态"
              >
                <InputNumber style={{ width: 200 }} value={port} onChange={(v) => setPort(v ?? 0)} min={0} max={65535} />
              </Form.Item>
            </Form>
          ),
        },
        {
          title: '说明',
          content: (
            <Alert
              type="info"
              showIcon
              message="部署任务起来后一直处于运行中，没有「完成」态；停止即视为正常结束。服务地址会在进程绑定端口后回填到列表。"
            />
          ),
        },
      ]}
    />
  );
}
