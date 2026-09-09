import { useMemo, useState } from 'react';
import { Divider, Form, Input, InputNumber, Radio, Select, Switch, Tooltip } from 'antd';
import { ConfigFormShell } from '@/components/ConfigFormShell';
import { MODULES } from '@/theme/modules';
import { neutral } from '@/theme/theme';
import { availableModels, checkpoints, datasets } from '@/mock/data';

/**
 * 新建训练。表单字段最终来自各 Config 类的 json_schema，这里先摆常用的几项。
 * ckpt 不做独立页面，只以下拉形式出现在「从 checkpoint 续训」这一处（对应设计决议）。
 */
export function TrainNewPage() {
  const [model, setModel] = useState(availableModels[0]);
  const [trainType, setTrainType] = useState('lora');
  const [useServer, setUseServer] = useState(false);
  const [dataset, setDataset] = useState(datasets[0]);
  const [resumeCkpt, setResumeCkpt] = useState<string | undefined>();

  const preview = useMemo(() => {
    if (useServer) {
      return [
        '#!/usr/bin/env bash',
        '# run.sh（连 twinkle-server：训练循环在本地，算子下沉到 server）',
        'cd "$(dirname "$0")"',
        'export TWINKLE_SERVER_URL=http://10.0.1.7:8000',
        'export TWINKLE_MODEL_ID=' + model,
        'python train.py >> output.log 2>&1',
        'echo $? > exit_code',
      ].join('\n');
    }
    const args = [
      'swift sft',
      `    --model ${model}`,
      `    --train_type ${trainType}`,
      `    --dataset ${dataset}`,
      resumeCkpt ? `    --resume_from_checkpoint ${resumeCkpt}` : null,
      '    --output_dir outputs',
    ]
      .filter(Boolean)
      .join(' \\\n');
    return [
      '#!/usr/bin/env bash',
      '# run.sh（本地训练）',
      'cd "$(dirname "$0")"',
      `${args} >> output.log 2>&1`,
      'echo $? > exit_code',
    ].join('\n');
  }, [model, trainType, dataset, resumeCkpt, useServer]);

  return (
    <ConfigFormShell
      module={MODULES.train}
      title="新建训练"
      desc="配置基座模型、数据与超参，提交后生成 run.sh 并拉起"
      preview={preview}
      sections={[
        {
          title: '模型与数据',
          content: (
            <Form layout="vertical" size="middle">
              <Form.Item label="基座模型">
                <Select value={model} onChange={setModel} options={availableModels.map((m) => ({ value: m, label: m }))} showSearch />
              </Form.Item>
              <Form.Item label="数据集">
                <Select value={dataset} onChange={setDataset} options={datasets.map((d) => ({ value: d, label: d }))} showSearch mode="multiple" maxTagCount="responsive" />
              </Form.Item>
              <Form.Item
                label="从 checkpoint 续训"
                tooltip="ckpt 不单独建页，只在这里以下拉出现；血缘记录在 lineage.json"
              >
                <Select
                  allowClear
                  placeholder="从头开始（不选）"
                  value={resumeCkpt}
                  onChange={setResumeCkpt}
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
          title: '训练方式',
          content: (
            <Form layout="vertical" size="middle">
              <Form.Item label="微调类型">
                <Radio.Group
                  value={trainType}
                  onChange={(e) => setTrainType(e.target.value)}
                  optionType="button"
                  options={[
                    { value: 'lora', label: 'LoRA' },
                    { value: 'full', label: '全参' },
                    { value: 'longlora', label: 'LongLoRA' },
                  ]}
                />
              </Form.Item>
              <Form.Item
                label={
                  <Tooltip title="连 twinkle-server 的训练可暂停/继续，本地训练不行">
                    <span>连接 twinkle-server（远端算子）</span>
                  </Tooltip>
                }
              >
                <Switch checked={useServer} onChange={setUseServer} />
              </Form.Item>
              {useServer && (
                <Form.Item label="server 地址">
                  <Input placeholder="http://10.0.1.7:8000" defaultValue="http://10.0.1.7:8000" />
                </Form.Item>
              )}
            </Form>
          ),
        },
        {
          title: '超参数',
          content: (
            <Form layout="vertical" size="middle">
              <div style={{ display: 'flex', gap: 16, flexWrap: 'wrap' }}>
                <Form.Item label="学习率" style={{ flex: 1, minWidth: 160 }}>
                  <InputNumber style={{ width: '100%' }} defaultValue={1e-4} step={1e-5} />
                </Form.Item>
                <Form.Item label="epochs" style={{ flex: 1, minWidth: 160 }}>
                  <InputNumber style={{ width: '100%' }} defaultValue={3} min={1} />
                </Form.Item>
                <Form.Item label="batch size" style={{ flex: 1, minWidth: 160 }}>
                  <InputNumber style={{ width: '100%' }} defaultValue={4} min={1} />
                </Form.Item>
                <Form.Item label="max length" style={{ flex: 1, minWidth: 160 }}>
                  <InputNumber style={{ width: '100%' }} defaultValue={2048} step={256} />
                </Form.Item>
              </div>
              <Divider style={{ margin: '4px 0 12px' }} />
              <div style={{ fontSize: 12, color: neutral.textTertiary }}>
                完整参数将由对应 Config 的 json_schema 自动展开，这里仅列常用项。
              </div>
            </Form>
          ),
        },
      ]}
    />
  );
}
