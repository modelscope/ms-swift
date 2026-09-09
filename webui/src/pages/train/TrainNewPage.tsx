import { useMemo, useState } from 'react';
import { ConfigFormShell } from '@/components/ConfigFormShell';
import {
  CheckboxGroup,
  Field,
  FieldRow,
  FieldStack,
  NumberInput,
  SegmentedControl,
  SelectInput,
} from '@/components/FormField';
import { Input } from '@/components/ui/input';
import { Switch } from '@/components/ui/switch';
import { Separator } from '@/components/ui/separator';
import { MODULES } from '@/theme/modules';
import { availableModels, checkpoints, datasets } from '@/mock/data';

/** 「从头开始」这一项的哨兵值。Radix 的 Select 不允许空串做值，所以不能用 '' 表示不选 */
const NO_CKPT = 'none';

/**
 * 新建训练。表单字段最终来自各 Config 类的 json_schema，这里先摆常用的几项。
 * ckpt 不做独立页面，只以下拉形式出现在「从 checkpoint 续训」这一处（对应设计决议）。
 */
export function TrainNewPage() {
  const [model, setModel] = useState(availableModels[0]);
  const [trainType, setTrainType] = useState('lora');
  const [useServer, setUseServer] = useState(false);
  const [selectedDatasets, setSelectedDatasets] = useState<string[]>([datasets[0]]);
  const [resumeCkpt, setResumeCkpt] = useState(NO_CKPT);
  /**
   * 超参也拿状态接起来并拼进预览。
   * 原来这四个框只有 defaultValue，填了不影响任何东西——一个写着「将要执行」
   * 的预览区旁边放四个不进命令的输入框，比不放更容易误导人。
   * 一个值可以为空（用户删完），为空时就不出那个参数，跟命令行的默认值语义一致。
   */
  const [lr, setLr] = useState<number | undefined>(1e-4);
  const [epochs, setEpochs] = useState<number | undefined>(3);
  const [batchSize, setBatchSize] = useState<number | undefined>(4);
  const [maxLength, setMaxLength] = useState<number | undefined>(2048);

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
      selectedDatasets.length ? `    --dataset ${selectedDatasets.join(' ')}` : null,
      resumeCkpt !== NO_CKPT ? `    --resume_from_checkpoint ${resumeCkpt}` : null,
      lr !== undefined ? `    --learning_rate ${lr}` : null,
      epochs !== undefined ? `    --num_train_epochs ${epochs}` : null,
      batchSize !== undefined ? `    --per_device_train_batch_size ${batchSize}` : null,
      maxLength !== undefined ? `    --max_length ${maxLength}` : null,
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
  }, [model, trainType, selectedDatasets, resumeCkpt, useServer, lr, epochs, batchSize, maxLength]);

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
            <FieldStack>
              <Field label="基座模型">
                <SelectInput
                  value={model}
                  onChange={setModel}
                  options={availableModels.map((m) => ({ value: m, label: m }))}
                />
              </Field>
              {/*
                数据集本来是 mode="multiple" 的下拉。选项只有 4 个，摊开成多选框
                比「点开下拉、勾几个、再点空白处收起」少两步操作，也不用再解释
                那些标签能不能删。选项多到摊不开的那天再换回下拉。
              */}
              <Field label="数据集" hint="多选。会按这里的顺序拼进 --dataset">
                <CheckboxGroup
                  value={selectedDatasets}
                  onChange={setSelectedDatasets}
                  options={datasets.map((d) => ({ value: d, label: d }))}
                />
              </Field>
              <Field
                label="从 checkpoint 续训"
                hint="ckpt 不单独建页，只在这里以下拉出现；血缘记录在 lineage.json"
              >
                <SelectInput
                  value={resumeCkpt}
                  onChange={setResumeCkpt}
                  options={[
                    { value: NO_CKPT, label: '从头开始' },
                    ...checkpoints.map((c) => ({
                      value: `${c.fromTask}/${c.name}`,
                      label: `${c.name} · step ${c.step} · 来自 ${c.fromTask}`,
                    })),
                  ]}
                />
              </Field>
            </FieldStack>
          ),
        },
        {
          title: '训练方式',
          content: (
            <FieldStack>
              <Field label="微调类型">
                <SegmentedControl
                  value={trainType}
                  onChange={setTrainType}
                  options={[
                    { value: 'lora', label: 'LoRA' },
                    { value: 'full', label: '全参' },
                    { value: 'longlora', label: 'LongLoRA' },
                  ]}
                />
              </Field>
              <Field
                label="连接 twinkle-server（远端算子）"
                hint="连 twinkle-server 的训练可暂停/继续，本地训练不行"
              >
                <Switch checked={useServer} onCheckedChange={setUseServer} />
              </Field>
              {useServer && (
                <Field label="server 地址">
                  <Input placeholder="http://10.0.1.7:8000" defaultValue="http://10.0.1.7:8000" />
                </Field>
              )}
            </FieldStack>
          ),
        },
        {
          title: '超参数',
          content: (
            <>
              <FieldRow>
                <Field label="学习率">
                  <NumberInput value={lr} onChange={setLr} step={1e-5} min={0} />
                </Field>
                <Field label="epochs">
                  <NumberInput value={epochs} onChange={setEpochs} min={1} />
                </Field>
                <Field label="batch size">
                  <NumberInput value={batchSize} onChange={setBatchSize} min={1} />
                </Field>
                <Field label="max length">
                  <NumberInput value={maxLength} onChange={setMaxLength} step={256} min={1} />
                </Field>
              </FieldRow>
              <Separator className="mt-4 mb-3" />
              <div className="text-muted-foreground text-xs">
                完整参数将由对应 Config 的 json_schema 自动展开，这里仅列常用项。
              </div>
            </>
          ),
        },
      ]}
    />
  );
}
