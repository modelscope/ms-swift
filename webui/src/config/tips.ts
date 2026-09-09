/**
 * 底部轮换 tips。
 *
 * 借的是游戏读条时那块小字的路子：反正要等，不如顺手讲点有用的。
 * 所以每条都尽量做到两件事——读着有意思，而且说的是真事。
 *
 * level 用来配合设置里的「只看通俗的」：
 *  - basic：不需要背景知识，看完能直接用
 *  - deep：涉及具体算法或数值细节，新手看了容易误用
 */
export interface Tip {
  text: string;
  level: 'basic' | 'deep';
}

export const TIPS: Tip[] = [
  /* ---- 心态类：等的时候最需要的其实是这个 ---- */
  { text: '炼丹需要有耐心，这次不行还有下次。', level: 'basic' },
  { text: '一次只改一个变量。同时动学习率和 batch size，你不会知道是哪个起了作用。', level: 'basic' },
  { text: '先用最小的模型把流程跑通，再换大的——调试 0.5B 比调试 72B 便宜一百倍。', level: 'basic' },
  { text: '训练曲线好看不等于模型好用，记得留一批你自己看得懂的样本做人工抽查。', level: 'basic' },
  { text: '评测集一旦被你反复看过很多次，它就慢慢变成训练集了。', level: 'basic' },

  /* ---- 数据类 ---- */
  { text: 'loss 降不下去，先别急着换模型，去看看数据里有没有把答案抄在 prompt 里。', level: 'basic' },
  { text: '数据里混进 10% 的脏样本，往往比少掉 50% 的干净样本更伤模型。', level: 'basic' },
  { text: 'tokenizer 换了就等于换了语言，别把两个模型的分词器混着用。', level: 'basic' },

  /* ---- 旋钮类 ---- */
  { text: '学习率是最值得先调的那个旋钮，比换模型架构划算得多。', level: 'basic' },
  { text: 'LoRA 的 rank 从 8 开始试通常就够了，翻到 64 很多时候只是让显存占用更好看。', level: 'basic' },
  { text: 'warmup 不是玄学：训练最开始梯度方向最不可靠，先小步走稳一点。', level: 'basic' },
  { text: 'temperature 调到 0 不会让模型变聪明，只会让它变得更固执。', level: 'basic' },
  { text: '梯度累积能换来更大的等效 batch，但换不来更快的训练——步数没少，只是合并了更新。', level: 'deep' },

  /* ---- 显存类 ---- */
  {
    text: '显存不够的处理顺序：先降 batch，再开梯度累积，再开 gradient checkpointing，最后才考虑换卡。',
    level: 'basic',
  },
  { text: 'gradient checkpointing 是拿重算换显存：激活值不留着了，反向时再算一遍。', level: 'deep' },
  {
    text: 'flash attention 省的主要是显存搬运，不是计算量——它把注意力分块算，不把整个矩阵落地。',
    level: 'deep',
  },
  { text: 'bf16 通常比 fp16 更值得选：指数位和 fp32 一样多，训练时不容易溢出。', level: 'deep' },

  /* ---- 出错类 ---- */
  { text: 'loss 突然变成 NaN，先怀疑学习率，再怀疑数据里混进了空样本。', level: 'basic' },
  {
    text: '断点续训前确认优化器状态也存下来了。只存了权重，等于回去重新 warmup 一遍。',
    level: 'deep',
  },
  { text: '模型答得又长又客气，不代表它答对了——长度相关的奖励要格外小心。', level: 'basic' },

  /* ---- 强化学习类 ---- */
  { text: 'reward hacking 不是模型学坏了，是奖励函数写漏了。', level: 'basic' },
  {
    text: 'GRPO 里如果一组采样全对或者全错，这组的优势会是 0——这一组等于白跑。',
    level: 'deep',
  },
  {
    text: '多轮采样最容易出的 bug 不是模型不会用工具，而是轨迹和奖励对错了位——而且它不会报错。',
    level: 'deep',
  },
  {
    text: '过滤轨迹的时候，记得优势要在过滤之前算好。先过滤再算，组内归一化的分母就变了。',
    level: 'deep',
  },
  { text: '参数量翻十倍，推理成本也跟着翻十倍，效果往往只涨一点。先把数据和 prompt 榨干。', level: 'basic' },

  /* ---- 工程类 ---- */
  { text: 'checkpoint 的保存间隔，按「你能接受重跑多久」来定，不是按整数好看来定。', level: 'basic' },
  { text: '跑长任务前先跑 10 步看看——大部分崩溃都发生在前 10 步。', level: 'basic' },
];

/** 按设置挑出可用的 tips */
export function visibleTips(beginnerOnly: boolean): Tip[] {
  return beginnerOnly ? TIPS.filter((t) => t.level === 'basic') : TIPS;
}
