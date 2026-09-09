import type { ModuleKey } from '@/theme/modules';
import { MODULES } from '@/theme/modules';
import type { ActionState, AiMessage, ProposalView } from './AiChatPanel';
import { messageId } from './AiChatPanel';

/**
 * 新建页的 AI 协助层。训练、评测、导出、部署四个新建页共用这一份。
 *
 * 跟编排页那份（pages/workflow/aiAssist.ts）是同一条规矩：AI 只提议，不直接改。
 * 差别只在提议的对象——那边改图上节点的参数，这边改表单里的字段。
 *
 * 四个模块合在一个文件里，是因为它们的问题有一半是重的（显存、慢、这些参数怎么填、
 * 日志在哪），只有另一半各自不同。拆成四份，重的那一半就要抄四遍。
 *
 * 接真后端时把 askConfigAi() 换成一次请求即可，界面和字段模型都不用动。
 */

/** 开关型字段的两个取值。页面和剧本必须用同一份字面量，所以在这里定死 */
export const ON = '开';
export const OFF = '关';

/** 助手能读、能改的一个表单字段 */
export interface ConfigField {
  /**
   * 剧本用它指认字段，命名跟命令行参数对齐（learning_rate 而不是 lr）。
   * 这样 AI 说的词、卡片上的词、右边预览里那条命令用的是同一个词。
   */
  key: string;
  label: string;
  /** 当前值。一律字符串——助手只负责显示和回写，不参与计算 */
  value: string;
  /**
   * 提议被应用时怎么写回去。字段自己带 setter，页面就不用再写一份
   * key → setState 的分发表；不给 apply 就是只读：AI 能引用它，但提不出改它的建议。
   */
  apply?: (value: string) => void;
}

/** 助手唯一能提的建议：把某个字段改成某个值。做得这么窄才敢让它一键应用 */
export interface ConfigAction {
  key: string;
  to: string;
}

/** 新建页的一条消息。消息的形状是通用的，只有提议的类型是这一页特有的 */
export type ConfigMessage = AiMessage<ConfigAction>;

export interface ConfigContext {
  module: ModuleKey;
  fields: ConfigField[];
}

const field = (ctx: ConfigContext, key: string) => ctx.fields.find((f) => f.key === key);
const valueOf = (ctx: ConfigContext, key: string) => field(ctx, key)?.value ?? '';
const numOf = (ctx: ConfigContext, key: string) => Number(valueOf(ctx, key));
const isOn = (ctx: ConfigContext, key: string) => valueOf(ctx, key) === ON;

/** 一段还没变成消息的回答 */
interface Draft {
  text: string;
  actions?: ConfigAction[];
}

/** 一条剧本：问题里命中 match 就出这段回答 */
interface Topic {
  match: RegExp;
  reply: (ctx: ConfigContext) => Draft;
}

/** 一条体检结论。有 action 的意味着「这条我能顺手帮你改」 */
interface Finding {
  text: string;
  action?: ConfigAction;
}

/*
 * 配置体检。
 *
 * 这几条不是凑数的：每一条都是「表单填得下去、命令拼得出来、但一跑就出事」的组合——
 * 类型检查和预览都拦不住，只有人（或者这里）盯着字段之间的关系才看得出来。
 *
 * Partial 是因为只有四个新建页会走这个外壳，对话页和编排页不在其中。
 */
const REVIEW: Partial<Record<ModuleKey, (ctx: ConfigContext) => Finding[]>> = {
  train: (ctx) => {
    const out: Finding[] = [];
    if (!valueOf(ctx, 'dataset')) {
      out.push({ text: '数据集一个都没选，命令里就不会有 --dataset，任务起来会直接失败。' });
    }
    if (valueOf(ctx, 'train_type') === 'full' && numOf(ctx, 'learning_rate') >= 1e-4) {
      out.push({
        text: '全参微调配 1e-4 的学习率大了一个量级——动的是全部权重，这个步长很容易一上手就把模型带崩。',
        action: { key: 'learning_rate', to: '1e-5' },
      });
    }
    if (numOf(ctx, 'per_device_train_batch_size') >= 4 && numOf(ctx, 'max_length') >= 4096) {
      out.push({
        text: `batch ${valueOf(ctx, 'per_device_train_batch_size')} 配 ${valueOf(ctx, 'max_length')} 的长度，显存占用是这两个数的乘积再乘一个平方项，单卡 24G 上大概率 OOM。`,
        action: { key: 'max_length', to: '2048' },
      });
    }
    return out;
  },

  eval: (ctx) =>
    valueOf(ctx, 'eval_dataset')
      ? []
      : [{ text: '评测集一个都没勾，--eval_dataset 会是空的，evalscope 直接报参数错误。' }],

  export: (ctx) => {
    const out: Finding[] = [];
    if (valueOf(ctx, 'quant_method') !== 'none' && !isOn(ctx, 'merge_lora')) {
      out.push({
        text: '要量化但没合并 LoRA：量化会拿基座权重去做，微调出来的那部分被丢在外面，导出的模型跟没训过一样。',
        action: { key: 'merge_lora', to: ON },
      });
    }
    if (isOn(ctx, 'push_to_hub')) {
      out.push({
        text: '推送到 Hub 需要 MODELSCOPE_API_TOKEN，没配的话前面几步都跑完了、最后一步才失败——比较费时间。',
      });
    }
    return out;
  },

  deploy: (ctx) => {
    const port = numOf(ctx, 'port');
    return port > 0
      ? [
          {
            text: `端口写死成 ${port} 了。两个部署撞同一个端口时第二个起不来，而且失败发生在进程拉起之后，只有日志里看得到。`,
            action: { key: 'port', to: '0' },
          },
        ]
      : [];
  },
};

/*
 * 各模块的剧本。
 *
 * 剧本不是随便编的：每条都对应一个真实会被问到的问题，回答里引用的是表单上
 * 当前的真实取值，所以看着才像真在读这份配置。
 */
const TOPICS: Partial<Record<ModuleKey, Topic[]>> = {
  train: [
    {
      match: /oom|显存|内存不够|爆了|out of memory/,
      reply: (ctx) => {
        const bs = valueOf(ctx, 'per_device_train_batch_size');
        const len = valueOf(ctx, 'max_length');
        const actions: ConfigAction[] = [
          { key: 'per_device_train_batch_size', to: '1' },
          { key: 'max_length', to: '1024' },
        ];
        if (valueOf(ctx, 'train_type') === 'full') actions.push({ key: 'train_type', to: 'lora' });
        return {
          text: [
            '按代价从小到大来：',
            `1. 先降 batch size（现在 ${bs || '未填'}）——最便宜，只影响单卡一次塞多少`,
            `2. 再压 max_length（现在 ${len || '未填'}）。显存跟序列长度不是线性的，attention 那部分是平方，长文本上它比 batch 管用`,
            '3. 用梯度累积把等效 batch 补回来，效果基本等价，就是慢一点',
            '4. 还不够就换 LoRA，或者上并行',
            '',
            '前两条我能直接改，后两条要在完整参数里配。',
          ].join('\n'),
          actions,
        };
      },
    },
    {
      match: /lora|全参|full|微调类型|哪种|冻结/,
      reply: () => ({
        text: [
          'LoRA 只训一小份低秩增量，基座权重冻着：显存省一大截，产物是几十 MB 的适配器，可以针对不同任务各训一个挂着用。',
          '全参把所有权重都动，效果上限更高，但显存要几倍，产物是一整个模型。',
          '',
          '判断依据是数据量和目标：几千到几万条、只是想让模型换个说法或者会一件具体的事，LoRA 够了；',
          '十万条以上、或者要动模型的基础能力，才值得上全参。',
        ].join('\n'),
        actions: [{ key: 'train_type', to: 'lora' }],
      }),
    },
    {
      match: /学习率|learning_rate|\blr\b|步长/,
      reply: (ctx) => {
        const type = valueOf(ctx, 'train_type');
        return {
          text: [
            `LoRA 常用 1e-4 这个量级，全参要小一个量级，1e-5 上下——动的参数量差好几个数量级，步长得跟着降。`,
            `你现在是${type === 'full' ? '全参' : 'LoRA'}，学习率 ${valueOf(ctx, 'learning_rate') || '未填'}。`,
            '',
            '判断的依据不是数值本身，是看 loss 曲线：一路平着不降是太小，抖得看不出趋势或者直接飞了是太大。',
          ].join('\n'),
          actions: type === 'full' ? [{ key: 'learning_rate', to: '1e-5' }] : [],
        };
      },
    },
    {
      match: /数据集|dataset|数据/,
      reply: () => ({
        text: [
          '多选会按你勾的顺序拼进 --dataset，重复样本不会自动去重。',
          '混多个数据集之前先确认格式一致——它们得能被同一个 template 解析，不然报错会出现在某一条样本上，很难定位。',
          '',
          '数据的影响比任何超参都大。真调不出效果的时候，先回来看数据，别在学习率上耗。',
        ].join('\n'),
      }),
    },
    {
      match: /续训|resume|接着训|断了|中断/,
      reply: () => ({
        text: [
          '「从 checkpoint 续训」会把优化器状态和 step 一起接上，等于这次训练没断过。',
          '这跟「拿 ckpt 当基座重新训一遍」不是一回事——后者应该填到基座模型那一栏，学习率也会从头 warmup。',
          '',
          '血缘记在 lineage.json 里，所以之后在评测页还能顺着查回「这个模型是哪次训练的第几步」。',
        ].join('\n'),
      }),
    },
    {
      match: /暂停|继续|server|远端|停一下/,
      reply: (ctx) => ({
        text: [
          '只有连 twinkle-server 的训练能暂停、继续：训练循环在本地，算子下沉到 server，进程本身停得下来。',
          '本地训练就是一个进程，停掉就是停掉，只能从最近一个 checkpoint 重来。',
          `你现在${isOn(ctx, 'twinkle_server') ? '已经连了 server，中途可以停' : '是本地训练，起了就得跑到底'}。`,
        ].join('\n'),
        actions: [{ key: 'twinkle_server', to: ON }],
      }),
    },
    {
      match: /跑通|调试|试一下|小规模|冒烟|快速验证|慢|久/,
      reply: () => ({
        text: [
          '第一次跑先别管效果，先用最小规模把链路验通：1 轮、batch 1、长度 512。',
          '十分钟内能看到 loss 在降、能看到 checkpoint 落盘，就说明这条 run.sh 是通的。',
          '然后把这三个放回去重新起一次——反正配置能存草稿，改回来不费事。',
        ].join('\n'),
        actions: [
          { key: 'num_train_epochs', to: '1' },
          { key: 'per_device_train_batch_size', to: '1' },
          { key: 'max_length', to: '512' },
        ],
      }),
    },
    {
      match: /怎么填|不会|新手|推荐|默认|帮我配|帮我填|第一次|起手/,
      reply: () => ({
        text: [
          '一份能直接跑的起手配置：LoRA + 学习率 1e-4 + 3 轮 + batch 4 + 长度 2048。',
          '',
          '理由：LoRA 显存友好、产物小、试错代价低；1e-4 是 LoRA 的常用量级；',
          '3 轮足够看出 loss 有没有在降，不够可以从 checkpoint 续训，不用一开始就赌一个大数。',
        ].join('\n'),
        actions: [
          { key: 'train_type', to: 'lora' },
          { key: 'learning_rate', to: '1e-4' },
          { key: 'num_train_epochs', to: '3' },
          { key: 'per_device_train_batch_size', to: '4' },
          { key: 'max_length', to: '2048' },
        ],
      }),
    },
  ],

  eval: [
    {
      match: /评测集|数据集|gsm8k|ceval|mmlu|humaneval|选哪个|挂哪些/,
      reply: () => ({
        text: [
          '几个常见的：gsm8k 数学推理、ceval 中文综合、mmlu 英文综合、humaneval 代码。',
          '',
          '挑的原则是一个同域 + 一个通用：同域的看有没有练出效果，通用的看有没有练废——',
          '微调最常见的代价是灾难性遗忘，只测同域的话，模型别的能力掉了你也看不见。',
        ].join('\n'),
        actions: [{ key: 'eval_dataset', to: 'gsm8k ceval' }],
      }),
    },
    {
      match: /慢|久|快一点|时间/,
      reply: (ctx) => {
        const sets = valueOf(ctx, 'eval_dataset').split(' ').filter(Boolean);
        return {
          text: [
            '评测时间约等于 题量 × 每题生成长度，跟模型大小的关系反而没那么大。',
            `你现在挂了 ${sets.length || 0} 个集。`,
            '先只留一个跑通，确认 evalscope 装好了、分数能落到详情页，再一次性全挂上。',
          ].join('\n'),
          actions: sets.length > 1 ? [{ key: 'eval_dataset', to: sets[0] }] : [],
        };
      },
    },
    {
      match: /evalscope|装|失败|127|报错/,
      reply: () => ({
        text: [
          '评测这一步依赖 evalscope，它不是默认装上的。',
          '没装的话任务会以 exit_code=127 结束，日志里只有一行 command not found——看着像环境炸了，其实就是缺包。',
          '',
          '先在跑任务那台机器上 pip install evalscope，再回来提交。',
        ].join('\n'),
      }),
    },
    {
      match: /结果|分数|怎么看|对比|报告/,
      reply: () => ({
        text: [
          '跑完在评测详情页看：各评测集的分数摊成一张表，同一个 ckpt 的多次评测并排放。',
          '',
          '单看一个分数没有意义。至少要有两个数才能说事：基座的分数，和上一版的分数。',
          '所以第一次评测建议先测基座，把基线留下来。',
        ].join('\n'),
      }),
    },
    {
      match: /ckpt|checkpoint|被测|模型|哪一步/,
      reply: () => ({
        text: [
          '被测对象是训练任务产出的 checkpoint，下拉里带着 step 和来源任务。',
          '注意它是那一步的权重，不一定是最好的那一步——训练还在跑的时候，最新的 ckpt 未必比中间某个强。',
          '要挑就多测几个 step，看曲线，别只测最后一个。',
        ].join('\n'),
      }),
    },
    {
      match: /怎么填|不会|新手|推荐|默认|帮我配|帮我填|起手/,
      reply: () => ({
        text: [
          '起手挂两个：gsm8k 加 ceval，一个看推理、一个看中文综合，跑得也不算久。',
          '等这条链路验通了，再按你的实际任务换成同域的评测集。',
        ].join('\n'),
        actions: [{ key: 'eval_dataset', to: 'gsm8k ceval' }],
      }),
    },
  ],

  export: [
    {
      match: /顺序|先|合并.*量化|量化.*合并|merge/,
      reply: (ctx) => ({
        text: [
          '顺序是固定的：先合并 LoRA，再量化。',
          '反过来的话，量化的是基座权重，微调那部分还在适配器里没进去，导出的模型跟没训过一样——',
          '而且这个错不报任何异常，只能靠评测分数不对才发现。',
          '',
          `你现在合并是${isOn(ctx, 'merge_lora') ? '开着的' : '关着的'}，量化选的是 ${valueOf(ctx, 'quant_method')}。`,
        ].join('\n'),
        actions: [{ key: 'merge_lora', to: ON }],
      }),
    },
    {
      /* 摆在讲量化方式那条前面：「量化会掉多少点」里有「量化」二字，让它先说才答得对 */
      match: /精度|掉|效果|变差|损失/,
      reply: () => ({
        text: [
          'int4 上通常掉几个点，具体多少跟任务有关——生成类比选择题敏感。',
          '',
          '所以量化完必须再评一次，拿量化前的分数当基线。',
          '跳过这一步，你就只知道模型变小了，不知道它变笨了多少。',
        ].join('\n'),
      }),
    },
    {
      match: /awq|gptq|bnb|bitsandbytes|量化|int4|int8|选哪个/,
      reply: () => ({
        text: [
          'AWQ 和 GPTQ 都是 int4、都要拿校准数据离线跑一遍，产物推理快、显存省，AWQ 对指令模型通常掉点更少。',
          'BitsAndBytes 是 int8，加载时量化、不用校准，胜在省事，但推理速度没什么优势。',
          '',
          '选 int4 是为了省显存和提速；int8 更像是「先让它装得下」的过渡。',
        ].join('\n'),
      }),
    },
    {
      match: /hub|推送|token|modelscope|上传/,
      reply: () => ({
        text: [
          '推送需要 MODELSCOPE_API_TOKEN 这个环境变量，hub_model_id 要写成 org/name 的形式。',
          '私有仓得先在 Hub 上建好，这里不会替你创建。',
          '',
          '建议先不推，本地确认模型能加载、评测分数正常，再单独跑一次导出推上去。',
        ].join('\n'),
      }),
    },
    {
      match: /怎么填|不会|新手|推荐|默认|帮我配|帮我填|起手/,
      reply: () => ({
        text: [
          '起手：合并 LoRA，不量化。先拿到一个完整、能直接部署的模型。',
          '量化是第二步，而且要单独跑一次，前后各评一次分数——两件事分开做，出问题时才知道是哪一步的锅。',
        ].join('\n'),
        actions: [
          { key: 'merge_lora', to: ON },
          { key: 'quant_method', to: 'none' },
        ],
      }),
    },
  ],

  deploy: [
    {
      match: /vllm|lmdeploy|pytorch|\bpt\b|引擎|backend|选哪个|快/,
      reply: (ctx) => ({
        text: [
          'vLLM 吞吐最高，PagedAttention 把显存碎片管起来了，适合有并发的场景。',
          'LMDeploy 在国产卡和长文本上有优势。',
          'PyTorch 是兜底——什么模型都能起，就是慢，一般只用来验证「这个模型到底能不能加载」。',
          '',
          `你现在选的是 ${valueOf(ctx, 'infer_backend')}。`,
        ].join('\n'),
        actions: [{ key: 'infer_backend', to: 'vllm' }],
      }),
    },
    {
      match: /端口|port|冲突|占用/,
      reply: () => ({
        text: [
          '留 0 的意思是让进程自己找一个空闲端口，绑定成功后写回 runtime.json，列表里的服务地址就是从那儿读的。',
          '',
          '写死端口的问题不是「不能这么干」，是失败得晚：两个部署撞上时，第二个进程已经拉起来了才绑定失败，',
          '页面上看到的是任务起了又挂了，得翻日志才知道是端口的事。',
        ].join('\n'),
        actions: [{ key: 'port', to: '0' }],
      }),
    },
    {
      match: /显存|oom|内存|爆/,
      reply: () => ({
        text: [
          '部署这边调的是 gpu_memory_utilization 和 max_model_len，跟训练那套参数没关系。',
          'vLLM 默认会把显存吃到 90%——它是预分配 KV cache，不是真用了那么多，但同一张卡上还要跑别的东西时就得压下来。',
          '',
          '装不下再考虑量化后的模型，那是导出页的事。',
        ].join('\n'),
      }),
    },
    {
      match: /模型|adapter|lora|填什么/,
      reply: () => ({
        text: [
          '这里可以填基座模型，也可以填导出后的目录。',
          '要部署 LoRA 微调的结果，得用合并过的模型——没合并的适配器要走 --adapters，这一版界面只给了 --model。',
          '所以顺序是：先去导出页合并，再回来部署。',
        ].join('\n'),
      }),
    },
    {
      match: /完成|状态|停止|一直在跑|结束/,
      reply: () => ({
        text: [
          '部署任务没有「完成」态：起来就一直是运行中，直到你停掉它，停止就算正常结束。',
          '所以列表里看到「已停止」不代表失败，判断依据是 exit_code——0 是你停的，非 0 才是它自己挂的。',
        ].join('\n'),
      }),
    },
    {
      match: /怎么填|不会|新手|推荐|默认|帮我配|帮我填|起手/,
      reply: () => ({
        text: [
          '起手：vLLM + 端口留 0。',
          'vLLM 是这三个里最不容易出问题的，端口交给进程自己挑能免掉一整类「起了又挂」的排查。',
        ].join('\n'),
        actions: [
          { key: 'infer_backend', to: 'vllm' },
          { key: 'port', to: '0' },
        ],
      }),
    },
  ],
};

/** 四个模块都会被问到的事。放在模块剧本后面匹配，让更具体的先说话 */
const SHARED: Topic[] = [
  {
    match: /检查|看看|有没有问题|问题吗|合理|review|靠谱/,
    reply: (ctx) => {
      const found = REVIEW[ctx.module]?.(ctx) ?? [];
      if (!found.length) {
        return {
          text: [
            '这份配置我没看出明显的坑：字段之间不冲突，命令拼得出来。',
            '剩下的就不是配置层面能判断的了——效果好不好得跑完看指标。',
          ].join('\n'),
        };
      }
      return {
        text: [`看出 ${found.length} 处：`, '', ...found.map((f, i) => `${i + 1}. ${f.text}`)].join('\n'),
        actions: found.map((f) => f.action).filter((a): a is ConfigAction => !!a),
      };
    },
  },
  {
    match: /run\.sh|日志|目录|落在哪|文件|复现/,
    reply: () => ({
      text: [
        '提交后会在任务目录下生成 run.sh、配置文件，跑起来之后多出 output.log 和 exit_code。',
        'run.sh 负责重定向日志、执行命令、结束时写退出码——所以页面上能给出的失败信息就是 exit_code 加日志尾部。',
        '',
        '想复现一次不用经过界面：进那个目录 bash run.sh 就行。这也是这份预览要摆在右边的原因。',
      ].join('\n'),
    }),
  },
  {
    match: /草稿|提交|保存|存下来/,
    reply: () => ({
      text: [
        '存为草稿只留这份配置，不建目录、不起进程；提交运行才会生成任务目录并拉起。',
        '所以拿不准的时候先存草稿，回头接着改。',
        '',
        '（这一版两个按钮都是示意，还没接后端。）',
      ].join('\n'),
    }),
  },
  {
    match: /checkpoint|ckpt|血缘|lineage/,
    reply: () => ({
      text: [
        'checkpoint 是训练途中写盘的一份权重加优化器状态。它不单独占一个页面，只在需要选它的地方以下拉出现：续训、评测、导出。',
        '',
        '每个 ckpt 的来源记在 lineage.json 里，所以「这个模型是哪次训练的第几步、用的什么数据」是能查回去的。',
        '这条链断了的话，几周后没人说得清线上那个模型是怎么来的。',
      ].join('\n'),
    }),
  },
];

/**
 * 假 AI。按关键词匹配剧本，匹配不到就出兜底。
 *
 * 提议在这里统一过一道筛：引用不到的字段、只读的字段、值本来就等于建议值的，
 * 一律丢掉。这样每条剧本都可以放心地把一整套建议写全，不用自己判断当前状态。
 */
export function askConfigAi(question: string, ctx: ConfigContext): ConfigMessage {
  const q = question.toLowerCase();
  const topic = [...(TOPICS[ctx.module] ?? []), ...SHARED].find((t) => t.match.test(q));
  const draft = topic ? topic.reply(ctx) : fallback(ctx);

  const proposed = draft.actions ?? [];
  const actions = proposed.filter((a) => {
    const f = field(ctx, a.key);
    return !!f?.apply && f.value !== a.to;
  });
  /*
   * 提了一套建议、而表单本来就是这么填的——得把这个说出来。
   * 不说的话，用户看到一段「建议你 A + B + C」却没有任何卡片，会以为是坏了。
   */
  const satisfied =
    proposed.length > 0 &&
    actions.length === 0 &&
    proposed.every((a) => field(ctx, a.key)?.value === a.to);

  return {
    id: messageId(),
    role: 'assistant',
    text: satisfied ? `${draft.text}\n\n你现在填的就是这些值，不用改。` : draft.text,
    actions: actions.length ? actions : undefined,
    states: actions.map(() => 'pending' as ActionState),
  };
}

/**
 * 开场白。
 *
 * 第一句就把体检结果说了，而不是「你好，我能帮你做什么」——
 * 用户点开助手的时候手上有一份填了一半的表单，最有用的一句话是「这里有个坑」。
 */
export function configIntro(ctx: ConfigContext): ConfigMessage {
  const found = REVIEW[ctx.module]?.(ctx) ?? [];
  const label = MODULES[ctx.module].label;
  const first = found[0];
  /* 体检结论里那条能顺手改的，直接附成一张卡片；改不了的就只说 */
  const actions =
    first?.action && field(ctx, first.action.key)?.apply ? [first.action] : [];

  const text = [
    `这份新建${label}的配置我能读到，${ctx.fields.length} 项。`,
    '',
    first ? `先说一件事：${first.text}` : '现在这些字段之间没看出冲突。',
    '',
    '我能帮的：解释某个字段是什么、给一套能直接跑的起手值、按你的机器和目标调参数。',
    '改动都是提议，你点应用才会落到表单上。',
  ].join('\n');

  return {
    id: messageId(),
    role: 'assistant',
    text,
    actions: actions.length ? actions : undefined,
    states: actions.map(() => 'pending' as ActionState),
  };
}

/** 兜底。说清楚能问什么，比说「我不明白」有用 */
function fallback(ctx: ConfigContext): Draft {
  return {
    text: [
      `这道题我没有现成的答案。换个说法试试，或者问这些：`,
      '',
      '· 这个字段是干什么的、填多少合适',
      '· 帮我配一套能直接跑的起手值',
      '· 检查一下这份配置有没有问题',
      '· 提交之后会生成什么、日志在哪',
      '',
      `顺带一提，${MODULES[ctx.module].desc}。`,
    ].join('\n'),
  };
}

/**
 * 快捷追问。
 *
 * 跟剧本放在同一个文件里是有意的：这几句必须能命中上面的 match，
 * 分开放就会出现「点了按钮 AI 说不明白」——那比不给按钮更糟。
 */
export function configSuggestions(module: ModuleKey): string[] {
  const shared = ['检查一下这份配置'];
  const own: Partial<Record<ModuleKey, string[]>> = {
    train: ['帮我配一套起手值', '显存不够怎么办', 'LoRA 还是全参'],
    eval: ['该挂哪些评测集', '结果怎么看', '为什么这么慢'],
    export: ['合并和量化的顺序', '量化会掉多少点', '起手怎么配'],
    deploy: ['引擎选哪个', '端口为什么留 0', '显存不够怎么办'],
  };
  return [...(own[module] ?? []), ...shared];
}

/** 提议在卡片上的样子。字段名和当前值都得摆出来，用户才知道自己点的是什么 */
export function describeConfigAction(a: ConfigAction, fields: ConfigField[]): ProposalView {
  const f = fields.find((x) => x.key === a.key);
  const cur = f?.value;
  return { label: `${f?.label ?? a.key}：${cur ? `${cur} → ${a.to}` : `填 ${a.to}`}` };
}
