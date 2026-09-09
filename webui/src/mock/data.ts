import type {
  CheckpointItem,
  Conversation,
  EvalResult,
  MetricPoint,
  TaskItem,
} from './types';

/**
 * 全部是假数据，只为看界面。
 * 接后端时把这个文件换成 api/*.ts（同样的返回类型），页面不用改。
 */

export const trainTasks: TaskItem[] = [
  {
    id: 'train-20260908-142310-a3f9',
    type: 'train',
    label: 'qwen3-8b 自我认知 LoRA',
    status: 'RUNNING',
    createdAt: '2026-09-08 14:23:10',
    runner: 'local',
    model: 'Qwen/Qwen3-8B',
    step: 640,
    totalSteps: 1500,
  },
  {
    id: 'train-20260908-101502-7b21',
    type: 'train',
    label: '远端 4B 全参微调',
    status: 'PAUSED',
    createdAt: '2026-09-08 10:15:02',
    runner: 'local',
    serverUrl: 'http://10.0.1.7:8000',
    model: 'Qwen/Qwen3.5-4B',
    step: 820,
    totalSteps: 2000,
  },
  {
    id: 'train-20260907-231044-c5e8',
    type: 'train',
    label: '多机 DPO 对齐',
    status: 'DONE',
    createdAt: '2026-09-07 23:10:44',
    runner: 'ray',
    model: 'Qwen/Qwen3-8B',
    step: 3000,
    totalSteps: 3000,
  },
  {
    id: 'train-20260907-183355-91da',
    type: 'train',
    label: 'VL 图文混训 试跑',
    status: 'FAILED',
    createdAt: '2026-09-07 18:33:55',
    runner: 'local',
    model: 'Qwen/Qwen2-VL-7B-Instruct',
    step: 12,
    totalSteps: 900,
    error: 'CUDA out of memory (exit_code=1)',
  },
  {
    id: 'train-20260906-090012-2f70',
    type: 'train',
    label: 'embedding 蒸馏',
    status: 'DONE',
    createdAt: '2026-09-06 09:00:12',
    runner: 'local',
    model: 'Qwen/Qwen3-0.6B',
    step: 500,
    totalSteps: 500,
  },
];

export const evalTasks: TaskItem[] = [
  {
    id: 'eval-20260908-150220-b8c1',
    type: 'eval',
    label: 'ckpt-1500 综合评测',
    status: 'RUNNING',
    createdAt: '2026-09-08 15:02:20',
    runner: 'local',
    model: 'train-20260907-231044-c5e8 / checkpoint-3000',
  },
  {
    id: 'eval-20260907-234510-4d33',
    type: 'eval',
    label: 'base 模型基线',
    status: 'DONE',
    createdAt: '2026-09-07 23:45:10',
    runner: 'local',
    model: 'Qwen/Qwen3-8B',
  },
  {
    id: 'eval-20260907-120033-ee02',
    type: 'eval',
    label: 'gsm8k 单项复测',
    status: 'FAILED',
    createdAt: '2026-09-07 12:00:33',
    runner: 'local',
    model: 'train-20260906-090012-2f70 / checkpoint-500',
    error: 'evalscope 未安装（exit_code=127）',
  },
];

export const exportTasks: TaskItem[] = [
  {
    id: 'export-20260908-113000-6a5b',
    type: 'export',
    label: 'LoRA 合并 + int4 量化',
    status: 'DONE',
    createdAt: '2026-09-08 11:30:00',
    runner: 'local',
    model: 'train-20260907-231044-c5e8 / checkpoint-3000',
  },
  {
    id: 'export-20260908-093011-d7f4',
    type: 'export',
    label: '推送到 ModelScope Hub',
    status: 'RUNNING',
    createdAt: '2026-09-08 09:30:11',
    runner: 'local',
    model: 'merged-qwen3-8b-dpo',
  },
];

export const deployTasks: TaskItem[] = [
  {
    id: 'deploy-20260908-160500-f1a2',
    type: 'deploy',
    label: 'dpo 模型线上服务',
    status: 'RUNNING',
    createdAt: '2026-09-08 16:05:00',
    runner: 'local',
    model: 'merged-qwen3-8b-dpo',
    endpoint: 'http://localhost:8021/v1',
  },
  {
    id: 'deploy-20260908-155000-9cd6',
    type: 'deploy',
    label: 'base 对照服务',
    status: 'RUNNING',
    createdAt: '2026-09-08 15:50:00',
    runner: 'local',
    model: 'Qwen/Qwen3-8B',
    endpoint: 'http://localhost:8022/v1',
  },
  {
    id: 'deploy-20260907-201133-3ba9',
    type: 'deploy',
    label: '临时调试服务',
    status: 'DONE',
    createdAt: '2026-09-07 20:11:33',
    runner: 'local',
    model: 'Qwen/Qwen3-0.6B',
  },
];

export const workflowTasks: TaskItem[] = [
  {
    id: 'workflow-20260908-140000-aa10',
    type: 'workflow',
    label: '训练 → 评测 → 合并 → 部署',
    status: 'RUNNING',
    createdAt: '2026-09-08 14:00:00',
    runner: 'local',
    model: 'Qwen/Qwen3-8B',
  },
  {
    id: 'workflow-20260906-100000-bb22',
    type: 'workflow',
    label: '每日回归流程',
    status: 'DONE',
    createdAt: '2026-09-06 10:00:00',
    runner: 'ray',
    model: 'Qwen/Qwen3-8B',
  },
];

export const tasksByType = {
  train: trainTasks,
  eval: evalTasks,
  export: exportTasks,
  deploy: deployTasks,
  workflow: workflowTasks,
};

/** 模拟 metrics.jsonl 逐行读出来的曲线 */
export const metricPoints: MetricPoint[] = Array.from({ length: 64 }, (_, i) => {
  const step = (i + 1) * 10;
  return {
    step,
    loss: 2.4 * Math.exp(-i / 22) + 0.42 + Math.sin(i / 2.3) * 0.045,
    lr: 1e-4 * (1 - i / 80),
    gradNorm: 1.1 + Math.cos(i / 3) * 0.28,
  };
});

/** 模拟 output.log 的 tail */
export const logLines: string[] = [
  '[2026-09-08 14:23:10] run.sh: cd /root/.swift/webui/default/train/train-20260908-142310-a3f9',
  '[2026-09-08 14:23:10] run.sh: launch payload -> python train.py',
  '[INFO] loading model Qwen/Qwen3-8B ...',
  '[INFO] model loaded, dtype=bfloat16, device_map=cuda:0',
  '[INFO] trainable params: 20,971,520 || all params: 8,051,363,008 || ratio: 0.26%',
  '[INFO] dataset: self_cognition (train=1500, val=100)',
  '[INFO] ***** Running training *****',
  '[INFO]   num examples = 1500',
  '[INFO]   num epochs = 3',
  '[INFO]   total optimization steps = 1500',
  '{"step": 610, "loss": 0.5123, "grad_norm": 1.02, "lr": 8.4e-05, "epoch": 1.22}',
  '{"step": 620, "loss": 0.5081, "grad_norm": 0.98, "lr": 8.3e-05, "epoch": 1.24}',
  '{"step": 630, "loss": 0.4996, "grad_norm": 1.11, "lr": 8.2e-05, "epoch": 1.26}',
  '{"step": 640, "loss": 0.4952, "grad_norm": 1.05, "lr": 8.1e-05, "epoch": 1.28}',
  '[INFO] saving checkpoint to outputs/checkpoint-640',
];

/** ckpt 只在下拉里出现，不做独立页面；血缘记在 lineage.json */
export const checkpoints: CheckpointItem[] = [
  {
    name: 'checkpoint-3000',
    fromTask: 'train-20260907-231044-c5e8',
    step: 3000,
    createdAt: '2026-09-08 02:41:00',
  },
  {
    name: 'checkpoint-2000',
    fromTask: 'train-20260907-231044-c5e8',
    step: 2000,
    createdAt: '2026-09-08 01:12:00',
  },
  {
    name: 'checkpoint-640',
    fromTask: 'train-20260908-142310-a3f9',
    step: 640,
    createdAt: '2026-09-08 15:58:00',
  },
  {
    name: 'checkpoint-500',
    fromTask: 'train-20260906-090012-2f70',
    step: 500,
    createdAt: '2026-09-06 09:44:00',
  },
];

export const evalResults: EvalResult[] = [
  {
    name: 'base Qwen3-8B',
    scores: { gsm8k: 62, mmlu: 68, ceval: 71, humaneval: 44, ifeval: 55 },
  },
  {
    name: 'dpo checkpoint-3000',
    scores: { gsm8k: 74, mmlu: 70, ceval: 75, humaneval: 51, ifeval: 68 },
  },
];

export const conversations: Conversation[] = [
  { id: 'c1', title: '模型自我介绍测试', updatedAt: '16:20' },
  { id: 'c2', title: '数学题对比：base vs dpo', updatedAt: '15:04' },
  { id: 'c3', title: '让它写一段 SQL', updatedAt: '昨天' },
  { id: 'c4', title: '长文摘要能力探查', updatedAt: '昨天' },
  { id: 'c5', title: '工具调用联通性验证', updatedAt: '9月6日' },
];

export const availableModels = [
  'Qwen/Qwen3-8B',
  'Qwen/Qwen3.5-4B',
  'Qwen/Qwen3-0.6B',
  'Qwen/Qwen2-VL-7B-Instruct',
  'merged-qwen3-8b-dpo',
];

export const datasets = [
  'swift/self-cognition',
  'AI-ModelScope/alpaca-gpt4-data-zh',
  'modelscope/gsm8k',
  'swift/RLAIF-V-Dataset',
];

export const evalDatasets = ['gsm8k', 'mmlu', 'ceval', 'humaneval', 'ifeval', 'arc'];
