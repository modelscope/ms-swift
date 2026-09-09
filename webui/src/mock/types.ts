/**
 * 领域模型的类型定义。
 * 字段刻意对齐设计文档里 meta.json / runtime.json 的约定，
 * 后面接后端时这些类型可以直接复用，页面不用动。
 */

/** 任务状态。由后端从 exit_code + 查活推导，前端只负责显示，不自己算 */
export type TaskStatus = 'RUNNING' | 'DONE' | 'FAILED' | 'PAUSED';

export type TaskType = 'train' | 'eval' | 'export' | 'deploy' | 'workflow';

/** 拉起方式：只有两种，连不连 server 是 payload 属性而非 runner */
export type RunnerKind = 'local' | 'ray';

export interface TaskItem {
  /** {type}-{yyyymmdd-HHMMSS}-{short_uuid}，不可变 */
  id: string;
  type: TaskType;
  /** 用户可改的显示名 */
  label: string;
  status: TaskStatus;
  createdAt: string;
  runner: RunnerKind;
  /** 连了 twinkle-server 的任务：显示服务端地址，且可暂停/继续 */
  serverUrl?: string;
  model: string;
  /** 训练进度，仅 train */
  step?: number;
  totalSteps?: number;
  /** deploy 起来后进程写回 runtime.json 的实际地址 */
  endpoint?: string;
  /** 失败原因摘要 */
  error?: string;
}

export interface MetricPoint {
  step: number;
  loss: number;
  lr: number;
  gradNorm: number;
}

export interface CheckpointItem {
  /** 目录名 */
  name: string;
  /** 所属训练任务 id，即血缘的上一跳 */
  fromTask: string;
  step: number;
  createdAt: string;
}

export interface EvalResult {
  name: string;
  /** 各数据集得分，雷达图的轴 */
  scores: Record<string, number>;
}

export interface ChatMessage {
  role: 'user' | 'assistant';
  content: string;
}

export interface Conversation {
  id: string;
  title: string;
  updatedAt: string;
}
