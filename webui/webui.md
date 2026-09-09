webui 支持 swift-twinkle 的全流程使用：
1. 训练
	A. auto-research
	B. 自进化，RSI
	C. 组件化拼接流程，并支持训练后直接导出-量化-部署-评测
2. 部署
	A. 训练前后的推理对比（base 与 tuned 同时加载显存翻倍，需明确是同时加载还是串行跑完再比）
	B. 数据集批量推理
	C. agent对话：工具&skills 的支持
    D. nohup 拉起的本地部署
3. 评测
	A. 评测对比
	B. 历史评测管理
	C. 雷达图
4. ckpt 管理、训练管理
	A. 不做独立 ckpt 页面，ckpt 只在模型选择下拉中展示
	B. 文件中记录 ckpt 血缘：base model、数据集、父任务 id、产出 ckpt 列表
5. 界面风格可变
	A. 定制化两条路：页面部分可 vibecoding 重新编写；或直接更换主题
	B. 通过参数可以控制哪些界面、功能需要展示（前端隐藏≠后端禁用，后端必须独立校验，否则是越权漏洞）
6. 支持展示和编辑 python 代码，开启训练
7. 训练、部署、评测启动后不因为 webui 的停止而停止，下次启动后，所有状态可以回显

web-ui 和所有的命令、功能的交互完全由文件来管理。每个租户有自己的文件夹，单租户模式时，也有一个默认文件夹
存储分两层，不能混：
- 文件是真相（source of truth）：任务进程只写自己目录下的 metrics.jsonl / output.log / exit_code / runtime.json，天然单写者无需加锁
- sqlite 只是 WebUI 的索引缓存（支持分页、按状态筛选、排序），**只由 WebUI 单进程写，并且必须放本地盘、不能放共享盘**（sqlite 在 NFS/CPFS 上文件锁不可靠，多写者会损坏数据库）。索引丢了可以全量扫目录重建
单写者原则的两个推论（状态判定逻辑只存在 WebUI 里一份，不分散到多进程）：
- **status 不落盘**。它是 WebUI 从 exit_code + 查活推导出来的，存下来就会有两个真相。任务进程完全不知道 sqlite 的存在，也不负责报告自己的状态
- **meta.json 和 runtime.json 分开**。meta.json 由 WebUI 独占写（label、runner、handle 等）；任务进程只写 runtime.json（它只能自己知道的事实，如 deploy 实际 bind 的 endpoint）。两边各自单写，不争同一个文件
文件类型：
租户文件夹根目录由环境变量 SWIFT_WEBUI_HOME 或启动参数指定，默认 ~/.swift/webui。不放在 MODELSCOPE_CACHE（那是 modelscope 的模型缓存，混放会被清缓存波及），模型权重仍走 MODELSCOPE_CACHE
多机模式假定 SWIFT_WEBUI_HOME 在共享盘上（NFS/CPFS），启动时检测并提示
各租户文件夹：
harness：
1. skills：skills 列表，可以加载使用
2. prompts：prompt 列表，可以存储加载使用

状态：每个任务目录一个 meta.json，它是唯一真相（不用汇总 status.txt，单文件多任务并发写会撕裂）
meta.json（WebUI 写）：id、type、label（用户可改的显示名）、created_at、runner（local/ray）、handle（pgid / ray job_id）、start_time（防 pgid 复用误判）、swift+twinkle 版本、若连 twinkle-server 则额外记 server 地址与 run_id（仅用于展示）
runtime.json（任务进程写）：endpoint 等只有它自己知道的事实
两者写入都用临时文件 + os.replace 保证原子

成功/失败的判定：ps 只能告诉你「活/不活」，判不了成功还是失败，所以必须落盘退出码。四种任务统一包一层 run.sh wrapper，落在任务目录里，负责：cd 到任务目录、注入环境变量、重定向 output.log（stdout+stderr 合并）、执行 payload（shell 命令 / python 脚本 / 生成的脚本，三者之一）、最后 `echo $? > exit_code`
- 有 exit_code 且 =0 → DONE
- 有 exit_code 且 ≠0 → FAILED；但若同时存在 stopped_by_user 标记（stop 时先写）→ DONE。被信号杀死时 $? = 128+signum（SIGTERM→143、SIGKILL→137）
- 无 exit_code + 进程活 → RUNNING
- 无 exit_code + 进程死 → FAILED（被 OOM kill / 断电 / SIGKILL，wrapper 没来得及写）
- 有 paused_by_user 标记 + 进程死 → PAUSED（可 resume，不是终态）。注意标记要区分 paused_by_user 和 stopped_by_user 两种意图，不能只是一个布尔
wrapper 两个必须注意的点：
- **不能用 exec 跑 payload**，exec 会把 shell 本身替掉，之后就没人写 exit_code 了
- **stop 先发 SIGTERM 而不是 SIGKILL**，给 wrapper 机会写 exit_code（顺带让训练进程有机会保存 checkpoint）；超时（如 30s）未退再 SIGKILL
run.sh 落盘的额外好处：历史任务可以脱离 WebUI 直接 `bash run.sh` 重跑
ray 模式直接用 JobSubmissionClient 自带的 SUCCEEDED/FAILED/STOPPED，不用自己判（entrypoint 也就是 bash run.sh）
连 twinkle-server 的训练也是本地跑一个 python 脚本（参 cookbook/client/twinkle/*.py），同样被 run.sh 包住、同样有 exit_code 和 pgid、同样 ps 得到，判定规则完全一致，没有例外分支

model:
每个远端模型的 url、api，或 twinkle-server 地址
这里就是 run.sh 要注入的环境变量来源：TWINKLE_SERVER_URL / TWINKLE_SERVER_TOKEN / TWINKLE_MODEL_ID（cookbook 里的脚本用 dotenv 读 .env + os.environ.get 兜底，run.sh 直接 export 即可）

workflow:
1. yaml 的 workflow 流程（**定义**，可反复运行），包含训练流程、eval、lora 合并、deploy 等过程
2. 运行**实例**单独建目录 workflow-{yyyymmdd-HHMMSS}-{short_uuid}：含 meta.json、logs、快照一份当时的 yaml（定义后续被改不影响已跑完的实例）、各节点对应的子任务 id 列表

train：
每个子文件夹以 train-{yyyymmdd-HHMMSS}-{short_uuid} 命名（eval/export/deploy/workflow 同此规则）：不撞名、不可变、不承载语义，显示名放 meta.json 的 label
目录布局对齐 twinkle_client/auto/connection.py 已有的 run_dir 约定（同名同格式），让 autoresearch 的 agent 和 WebUI 读同一批文件，monitor/connection 的 tail 与读取逻辑直接复用，不写两套：
1. run.sh：统一 wrapper，JobRunner 只拉起它，不关心 payload 是哪种
2. train.py 当前脚本 + train_v{N}.py 归档旧版本（autoresearch 的版本管理就靠这个）；shell 的 swift 命令或 yaml+生成脚本同位置存放
3. output.log：stdout+stderr 合并（由 run.sh 重定向），增量 tail 读
4. outputs，包含了输出的 checkpoints；连 twinkle-server 时 ckpt 存在 server 侧的 training_run 下，本地只记一个远端路径
5. meta.json（WebUI 写）+ runtime.json（任务进程写）
6. exit_code：run.sh 写入的退出码，成功/失败判定的依据
7. metrics.jsonl：每行 {step, total_steps, loss, grad_norm, lr, epoch, eta}，WebUI 读它画曲线和进度条。注意它不是框架自动产出的，是训练脚本主动写的（参 twinkle_client/auto/runtime.py，buffering=1 行缓冲）：
   - python 脚本 / yaml 生成脚本路径：直接复用 twinkle_client.auto.runtime，**不用改 swift**。连 twinkle-server 的训练循环也跑在本地脚本里，同属这类
   - shell 的 swift 命令路径：必须在 swift 内部新增 metrics writer，**这才是需要改 swift 的部分**
8. lineage.json：ckpt 血缘

eval：
每个子文件夹以 eval-{yyyymmdd-HHMMSS}-{short_uuid} 命名
1. shell 的evalscope 命令，python 的自定义脚本、yaml 的界面组件流程+生成的 python 脚本，三者之一
2. output.log
3. outputs，包含了输出的评测结果
4. meta.json + exit_code（同 train）

export:
每个子文件夹以 export-{yyyymmdd-HHMMSS}-{short_uuid} 命名
1. shell 的 swift export 命令、python 的自定义脚本、yaml 的界面组件流程+生成的 python 脚本，三者之一
2. output.log
3. outputs，包含了输出结果
4. meta.json + exit_code（同 train）

deploy：
每个子文件夹以 deploy-{yyyymmdd-HHMMSS}-{short_uuid} 命名
deploy 只走 local runner（endpoint 固定 localhost，多机下跑到别的节点就不是 localhost 了）
端口自动分配：不由 WebUI 先探测空闲端口再拉起（探测到 bind 之间有竞态，两个 deploy 会撞同一端口），而是让被拉起的进程自己选端口并写回 runtime.json 的 endpoint，WebUI 轮询等它出现；启动失败（端口被外部占用等）靠 exit_code 反馈
状态：起来后一直 RUNNING（无 DONE），只能被杀死；stop 时先写 stopped_by_user 标记，所以杀死后判为 DONE 而不是 FAILED
1. shell 的 swift 命令，python 的自定义脚本、yaml 的界面组件流程+生成的 python 脚本，三者之一
2. output.log
3. harness：
    skills
    prompts
    messages 对话列表
4. meta.json + exit_code（同 train）

模块：
tenantmanager：
租户管理，根据单租户和多租户给出 token，以及生成短 uuid 的用户文件夹目录
- get_tenant_token
- get_tenant_dir
多租户安全需单独详细设计（待办）：token 只存 hash、下发方式与有效期、登录页、前端如何携带（Header）；前端传来的任务 id 必须校验，不能直接拼路径（防 ../ 穿越）；租户磁盘配额 + ckpt 保留策略（save_total_limit）+ 定期清理；停止和删除的审计记录

taskmanager：

根据当前租户，给出资源
- get_train_list
- get_eval_list
- get_export_list
- get_deploy_list
- get_workflow_list
可以分页，筛选项包含：运行状态（运行中、报错、已结束），按时间降序，走 sqlite 索引
状态优先读 meta.json + exit_code，还在 RUNNING 的再向对应 JobRunner 查活，回写 meta.json 和 sqlite 索引后返回（不用全局 ps 扫描，多机下 ps 查不到其它节点）

- stop_train
- stop_eval
- stop_export
- stop_deploy
- stop_workflow
统一走 JobRunner.stop(handle)，停止前先写 stopped_by_user 标记。不用命令关键词匹配（会误杀前缀相同的任务和用户自己在终端跑的同名命令，且 torchrun 子进程杀不净会留僵尸显存）。workflow 按记录的 task 依次停止
连 twinkle-server 的任务 stop 语义不同，必须区别对待：杀掉本地 client 进程后，**server 侧仍在 GPU 里保留 model/optimizer 状态，显存不释放**（见 twinkle_client/auto/connection.py 开头说明）。那边的约定是：SIGKILL = pause（用同一个 adapter_name 起新 client 即可续跑），SIGTERM = graceful stop（保存 checkpoint）。所以：
- stop 还要额外调 server API 释放资源，否则显存泄漏

- pause_train
- resume_train
只对连 twinkle-server 的训练提供（本地 GPU 训练的状态就在本进程里，杀了就没了，只能 stop 存 ckpt 后从 ckpt 重启，是有损的，不算 resume）
pause = 写 paused_by_user 标记后 SIGKILL 本地 client，server 侧保留 model/optimizer 状态（所以 pause 不释放显存，要在 UI 上说清楚）
resume = 用同一个 adapter_name 起新 client，无损续跑。adapter_name 必须稳定且落在 meta.json 里，否则找不回 server 侧状态
resume 复用同一个任务目录（不新建），三个必须处理的细节：
- run.sh 重定向 output.log 用 >> 而不是 >，否则 resume 会把之前的日志截掉
- 先删掉 exit_code，否则会被判成已结束；同时清 paused_by_user 标记
- pgid / start_time 变了要回写 meta.json；记一个 attempts 计数便于排查
metrics.jsonl 天然追写，step 从 pause 点继续，前端画曲线要能容忍 resume 处的断点

- delete_train
- delete_eval
- delete_export
- delete_deploy
- delete_workflow
如果任务还在运行，则先停止再删除。软删只保元数据和小文件（移入 .trash/ 并记录操作者，延迟物理清理），outputs 直接物理删除立即释放空间（ckpt 动辄几十 GB，软删不释放会让配额很快打满且用户困惑）。同步删 sqlite 索引

- create_train
- create_eval
- create_export
- create_deploy
- create_workflow
创建文件夹，写入 run.sh 和 payload，通过 JobRunner 拉起 run.sh（JobRunner 不关心 payload 是 shell 命令、python 脚本还是生成脚本，入口只有一个）。webui 进程自身不当 driver（twinkle.initialize() 是模块级全局状态，且用户 actor 无 lifetime='detached'，driver 死则训练全死）。JobRunner 只需两种实现：
- local：本机起独立 driver 进程，os.setsid 建进程组，pgid + start_time 落 meta.json，停止用 os.killpg
- ray：多机集群用 Ray Job Submission API（JobSubmissionClient），自带 job_id/status/tail_job_logs/stop_job；传 config json 而不是拼 shell
连 twinkle-server 的「remote 训练」不是第三种 runner：它就是一个本地 python 脚本（参 cookbook/client/twinkle/*.py），用 init_twinkle_client(base_url, api_key) 连服务端，**训练循环跑在本地进程里**，只有算子下沉到 server。所以它照样被 run.sh 包、照样有 pgid 和 exit_code、照样 ps 得到，走 local runner 即可。连不连 server 只是 payload 的一个属性（记在 meta.json 里用于展示服务端地址），不是 runner 的维度
日志回显用 SSE + 按字节 offset 增量拉取，不每次全量读；长训练日志要有滚动或截断策略
不做排队：多人并发提交不调度，撞卡由用户自己负责。仅在提交前检查显存并给告警（多模型对比、训练前后对比都是 N 倍显存，同样靠这个告警）

workflowmanager
组件用于将 workflow 组件转换为代码。
界面的组件可能是子组件（例如dataset、dataloader、model、eval 等），也可能包含了进程级别组件（export、deploy、eval 等），实际进行组合 workflow 时，
子组件需要组合为脚本，进程组件单独为 workflow 的节点。注意两个 eval 的区别：一个是 evalscope 命令，一个是 eval 子组件
组件列表涵盖了 twinkle、twinkle-agentic 的所有组件，如有无法支持的另算
另外增加代码组件，可以编辑 python 代码
增加一个循环组件，适配 dataloader 等循环输出组件，不需要 if 组件，有 if 的情况下使用代码组件
注意需要在 workflowmanager 中定义好每个组件的输入和输出，以及参数
参数表单不手写：组件配置本身是 dataclass/pydantic，用 TypeAdapter(Config).json_schema() 生成表单和校验（实测 18 个 config 类 643 字段全部可生成）
v1 先定组件白名单，不追求一次覆盖 twinkle/twinkle-agentic 全量
图形与代码同步：图形改动重新生成代码，也允许手改代码。生成时落一份代码指纹，手改后指纹不匹配就在 UI 提示「图形和代码已不一致」。此时再动图形弹确认框，二选一：fork（复制成新任务，新 id + 记 forked_from，保留手改版）或覆盖（丢弃手改）
不做 python → yaml 的反向解析（用户写了任意表达式、循环、条件后无法还原成组件图）
前提是 yaml→代码的生成必须稳定：同一份 yaml 永远产出同一份代码，含顺序和格式。需 golden test 锁住（yaml → 代码 → 逐字节比对基准文件），否则某天换了 dict 遍历顺序或格式化器，全量历史任务会集体报「不一致」
节点间数据传递用引用语法：${train-xxx.outputs.best_checkpoint}
失败后支持从失败节点续跑，不强制整体重跑
循环组件要在 yaml schema 里明确循环体内变量的作用域（对外是否可见）

- parse_workflow: 将组件转为合适的 yaml，这个方法调用后，才能调用 taskmanager 的create_workflow

evalmanager：
- get_eval_results
- compare_eval_results
evalscope 是 optional extra（requirements/eval.txt），缺失时要检测并提示安装
对比和雷达图的前提是 metric 名对齐：跨 benchmark 不可直接比，需显式定义可比维度，雷达图只在同 benchmark 组内出

autoresearch：
不新建任务树，而是在一个训练任务内做版本管理（同 twinkle_client.auto）：train.py 是当前脚本，每次 agent 改写先把旧版归档为 train_v{N}.py，meta.json 记 script_version。这样列表不会被自动任务淹没，RSI 的多层任务树先不做

skills:（复用 twinkle_client.skills 的 SkillManager + SkillProvider，已有 local/modelscope provider，只需补 provider，不重写）
- airesearchskill
- modelscope-skills
- huggingface-skills

tools:
- 仿照 twinkle_client.auto 的设计

chatmanager:

- create_chat(skills, tools, prompts, model): id
- delete_chat(id)
- chat(response): yield

页面列表：
- 默认：对话页面，可配置远端 api，或 twinkle-server 地址，或本地已启动的部署，可加载 skills，tools，可以支持 websearch
    - 多模型对比模式：单输入，多模型共同输出，需选择一个，然后才能进行后续多模型输出
    - 左侧可以选择或新建不同对话，后者删除对话
    - autoresearch：如果用户给出了训练需求，在训练创建后，在训练页面可以看到对应列表
- 训练：进入训练历史管理页面
    - 训练配置页面：列出模型列表和主要参数，可以开启训练，训练页面可以跳转到指标页面，实时更新指标，也可以跳转到 log 页面
- 评测：进入评测列表管理页面
    - 评测页面：评测参数，可以跳转 log 页面，也可以跳转评测指标页面，做评测横向对比
- 导出：同训练
- 部署：同训练
- workflow 编排
