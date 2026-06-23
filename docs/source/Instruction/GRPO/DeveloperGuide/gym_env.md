# GYM环境训练

GYM 风格的环境训练把"模型 → 环境 → 奖励"这条链路封装成一个抽象接口，让 LLM 像 Agent 一样与环境进行多轮交互，每一步的奖励直接由环境给出，无需再单独写 reward 函数从轨迹里反推。本文先介绍接口，再用一个完整的自定义示例（FrozenLake）说明如何接入训练。

## Gym 接口

GYM 源自 [Gymnasium库](https://github.com/Farama-Foundation/Gymnasium)。在 ms-swift 中我们定义了如下接口：

```python
class Env(ABC):

    def __init__(self, env_config):
        """env_config 来自数据集每行的 env_config 列，可承载初始化参数"""
        self.env_config = env_config

    @abstractmethod
    async def reset(self, config: RolloutInferRequest) -> Tuple[str, Dict[str, Any], str]:
        """
        Returns:
            - observation: 作为首轮 user 消息发送给模型
            - info: 调试/日志信息，记录到 completions.jsonl
            - system_message: 本条轨迹的 system prompt
        """
        pass

    @abstractmethod
    async def step(self, action: Messages) -> Tuple[str, float, bool, Dict[str, Any]]:
        """
        Args:
            action: 截止当前的完整对话消息，最后一条即模型最新回复
        Returns:
            - next_observation: 下一轮 user 消息
            - reward: 当前 step 奖励
            - done: 轨迹是否结束
            - info: 调试/日志信息
        """
        pass

    @abstractmethod
    async def close(self):
        """释放资源"""
        pass
```

`reset` 接收到的 `RolloutInferRequest` 包含数据集行的 `messages`、`data_dict`（额外列，包括 `env_config`）等。完整示例参见 [入参示例](./multi_turn.md#多轮规划器-multiturnscheduler)。

> 如果需要在每轮 rollout 之间额外控制对话历史（例如动态压缩、注入额外提示），推荐直接继承 `MultiTurnScheduler` 并实现 `on_trajectory_start` / `on_turn_end` hook，或重写 `step` / `run` 方法，详见[多轮训练文档](./multi_turn.md#自定义多轮交互逻辑)。

## 启动训练

使用内置的 [gym_scheduler](https://github.com/modelscope/ms-swift/blob/main/swift/rollout/multi_turn.py) 把 env 串到多轮 rollout 中。

`GYMScheduler` 基于通用 hook 协议实现：
- 继承 `MultiTurnScheduler`，无需自定义 `run` 方法
- 实现 `on_trajectory_start`（调用 `env.reset`）和 `on_turn_end`（调用 `env.step`）
- 同时适用于 server mode（`run()`）和 colocate mode（`run_multi_turn()`）

用户自定义的 env 通过 `--external_plugins your_plugin.py` 加载，plugin 里执行 `envs['my_env'] = MyEnv` 完成注册（下文 FrozenLake 示例完整演示）。

**Colocate 模式**:

```bash
megatron rlhf \
    --rlhf_type grpo \
    --vllm_mode colocate \
    --external_plugins examples/megatron/grpo/multi_turn/frozen_lake_plugin.py \
    --multi_turn_scheduler gym_scheduler \
    --gym_env frozen_lake \
    --use_gym_env true \
    --max_turns 10 \
    ...

# swift rlhf 同理
```


**Server 模式**

```bash
swift rollout \
    --model xxx \
    --use_gym_env true \
    --external_plugins examples/megatron/grpo/multi_turn/frozen_lake_plugin.py \
    --multi_turn_scheduler gym_scheduler \
    --gym_env frozen_lake \
    --max_turns 10

# trainer 侧需要加 --vllm_server_pass_dataset true，把 env_config 等额外列透传给 rollout 端
megatron rlhf --vllm_mode server --vllm_server_pass_dataset true ...
# or swift rlhf --vllm_mode server --vllm_server_pass_dataset true ...
```


环境选择有两种方式：
- 通过 `--gym_env env_name` 全局指定（同一脚本里所有 prompt 共用一个 env）；
- 在每行数据的 `env_config.name` 中指定（适用于多环境混合场景，每条数据可指向不同 env，会覆盖 `--gym_env`）。

## 示例：从零写一个 FrozenLake 环境

<img src="https://gymnasium.farama.org/_images/frozen_lake.gif" width="220" alt="FrozenLake 环境示意图（来源：Gymnasium 官方文档）" />

[FrozenLake](https://gymnasium.farama.org/environments/toy_text/frozen_lake/) 是 OpenAI Gym 中的经典任务：智能体从起点出发，需要穿过一片冰湖到达终点，途中要避开冰窟。原始环境如上图所示。下面以纯文本版本（把上图网格直接渲染成 ASCII 字符）为例。

以下完整代码参考完整代码：[frozen_lake_plugin](https://github.com/modelscope/ms-swift/blob/main/examples/megatron/grpo/multi_turn/frozen_lake_plugin.py)。

**1. 定义 Env**

每条数据派生一张随机 4x4 地图（随机洞 + 随机 S/G 位置，BFS 校验保证可解）。单元含义：`S` 起点 / `G` 终点 / `H` 冰窟（踩到=失败）/ `F` 安全冰面 / `P` 玩家当前位置。

```python
class FrozenLakeEnv(Env):
    def __init__(self, env_config):
        super().__init__(env_config)
        self.size = int(env_config.get('size', 4))
        self.p = float(env_config.get('p', 0.8))
        seed = env_config.get('seed')
        self.seed = int(seed) if seed is not None else None

    async def reset(self, config: RolloutInferRequest):
        self.grid = generate_random_map(size=self.size, p=self.p, seed=self.seed)
        ...
        return observation, {'seed': self.seed}, SYSTEM_PROMPT

    async def step(self, action: Messages):
        move = _parse_action(action[-1]['content'])  # <action>up|down|left|right</action>
        # 推进一格、判断 G / H；外层 max_turns 由 scheduler 兜底
        if cell == 'G': return obs, 1.0, True, {'status': 'goal'}
        if cell == 'H': return obs, 0.0, True, {'status': 'hole'}
        ...
```

**2. GYMScheduler 的 hook 实现**

框架内置的 `GYMScheduler` 基于多轮 hook 完成了控制逻辑：

```python
class GYMScheduler(MultiTurnScheduler):
    def on_trajectory_start(self, requests):
        # 为每个请求创建 env，调用 env.reset，注入初始 observation
        for req in requests:
            env = self._create_env(req.data_dict.get('env_config', {}))
            observation, info, system_message = env.reset(req)
            req.messages = [system_msg, user_msg(observation)]
            self._envs[req.uuid] = env

    def on_turn_end(self, req, response_choice, current_turn):
        # 调用 env.step，累积 reward，返回 done + rollout_infos
        next_obs, reward, done, info = env.step(deepcopy(req.messages))
        self._total_rewards[req.uuid] += reward
        return {
            'done': done,
            'rollout_infos': {
                'total_reward': self._total_rewards[req.uuid],
                'step_rewards': [...],
            }
        }

    def step(self, req, response_choice, current_turn):
        # 注入下一帧 observation 到 user message
        if self._pending_obs.get(req.uuid):
            req.messages.append({'role': 'user', 'content': next_obs})
        return {'infer_request': req}
```

用户只需实现 Env 接口，无需关心多轮控制细节。

**3. 注册**

将 env 类挂到 swift 的 `envs` 注册表里。`--external_plugins` 在训练启动时会 import 该文件，注册随之生效：

```python
# examples/megatron/grpo/multi_turn/frozen_lake_plugin.py
from swift.rollout.gym_env import Env, envs

class FrozenLakeEnv(Env):
    ...

envs['frozen_lake'] = FrozenLakeEnv
```

**4. 准备数据集**

数据集在这里仅作占位符处理，数据构造由环境生成，和 `env_config.seed`来控制地图生成的随机性：

```json
{"messages":[{"role":"user","content":"<placeholder>"}],"env_config":{"seed":0}}
{"messages":[{"role":"user","content":"<placeholder>"}],"env_config":{"seed":1}}
...
{"messages":[{"role":"user","content":"<placeholder>"}],"env_config":{"seed":127}}
```

**5. （可选）叠加自定义 reward**

设置 `--use_gym_env true` 后，env 给出的 `total_reward` 会自动作为一路奖励参与训练，无需再写 reward 函数。如果想在此之外再叠加自定义信号（如格式/长度等），通过 `--reward_funcs` 传入即可，gym 奖励会作为额外一列与 reward_funcs 拼在一起，由 `--reward_weights` 统一加权。例如同时启用一个格式校验 reward：

```bash
megatron rlhf ... --use_gym_env true --reward_funcs format --reward_weights 0.2 1.0
# reward_weights 末位对应 gym 的 total_reward
```

**6. 训练**

运行脚本参考：[`examples/megatron/grpo/multi_turn/frozen_lake.sh`](https://github.com/modelscope/ms-swift/blob/main/examples/megatron/grpo/multi_turn/frozen_lake.sh)


## OpenEnv 环境训练

[OpenEnv](https://github.com/huggingface/openenv) 是 HuggingFace 开源的 Agentic RL 环境框架，通过 WebSocket 与环境服务器交互。与上文 FrozenLake 的本地 `Env` 接口不同，OpenEnv 将环境逻辑放在独立的服务进程中，swift 通过 `OpenEnvScheduler` + `OpenEnvWrapper` 与之通信。

### 架构对比

| 特性 | 内置 Gym (`GYMScheduler`) | OpenEnv (`OpenEnvScheduler`) |
|------|--------------------------|------------------------------|
| 环境运行位置 | 训练进程内（Python 对象） | 独立服务器（WebSocket 通信） |
| 环境接口 | 继承 `Env`，实现 `reset/step/close` | 服务器提供 HTTP/WebSocket API |
| 注册方式 | `--external_plugins` + `envs` 注册表 | `--external_plugins` + `multi_turns` 注册表 |
| 适用场景 | 轻量本地环境（FrozenLake 等） | 复杂服务端环境（TextArena、CARLA 等） |
| 并发控制 | 无需 | 内置 Semaphore 限制并发连接 |

### OpenEnvScheduler

`OpenEnvScheduler` 继承 `GYMScheduler`，将本地 `Env` 替换为 `OpenEnvWrapper`（WebSocket 客户端）。核心设计：

- **`_create_env`**：创建 `OpenEnvWrapper`，连接 OpenEnv 服务器
- **`on_trajectory_start`**：为每个请求创建 wrapper，调用 `reset()`，用 Semaphore 限制并发（默认 4）
- **`on_turn_end`**：解析模型输出，调用 `wrapper.step()`，累积奖励
- **`parse_action`**（可覆盖）：将模型文本解析为 action dict，默认 `json.loads`
- **`format_observation`**（可覆盖）：将服务器返回的 observation 格式化为字符串，默认 `json.dumps`

用户通过继承 `OpenEnvScheduler` 并覆盖 `parse_action`、`format_observation`、`on_trajectory_start`、`on_turn_end` 来适配具体环境。

### 示例：Sudoku 环境

以 TextArena Sudoku 为例，模型需要通过 `[row col number]` 格式下棋，在 9x9 数独棋盘上填入数字。完整代码参考：[sudoku_scheduler.py](https://github.com/modelscope/ms-swift/blob/main/examples/train/grpo/plugin/openenv/sudoku_scheduler.py)。

**1. 启动 OpenEnv 服务器**

安装 OpenEnv 和 Sudoku 环境包（textarena 和 nltk 会作为依赖自动安装）：

```bash
pip install openenv
pip install git+https://huggingface.co/spaces/openenv/sudoku
```

使用提供的启动脚本启动本地服务器（默认端口 8000）。`MAX_CONCURRENT_ENVS` 需 ≥ 训练时的 `num_generations`：

```bash
TEXTARENA_ENV_ID=Sudoku-v0 MAX_CONCURRENT_ENVS=8 python examples/train/grpo/plugin/openenv/start_sudoku_server.py
```

数据集中将 `base_url` 指向本地服务器地址：

```json
{"messages":[{"role":"user","content":"Play"}],"env_config":{"name":"openenv","base_url":"http://127.0.0.1:8000"}}
```

**2. 自定义 Scheduler**

继承 `OpenEnvScheduler`，实现 Sudoku 专用的动作解析、观察格式化和多组件奖励：

```python
from swift.rollout.multi_turn import OpenEnvScheduler

class SudokuScheduler(OpenEnvScheduler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._last_content_len = {}  # 内容差分跟踪

    async def on_trajectory_start(self, requests):
        # 创建环境、解析棋盘、生成 hints
        # hints 包括「保证正确的走法」和候选数字列表
        ...

    async def on_turn_end(self, infer_request, response_choice, current_turn):
        # 解析 [row col number]，step 环境
        # 计算 5 路奖励：空格选择 / 合法移动 / 重复惩罚 / 进度 / 正确
        # 返回更新后的棋盘 + hints 作为下一轮观察
        ...

    def parse_action(self, text):
        import re
        match = re.search(r'\[\s*(\d+)\s+(\d+)\s+(\d+)\s*\]', text)
        if match:
            row, col, num = match.groups()
            return {"message": f"[{row} {col} {num}]"}
        return {"message": "[1 1 1]"}
```

**多组件奖励系统**（参考 [TRL Sudoku 示例](https://github.com/huggingface/trl/blob/main/examples/notebooks/openenv_sudoku_grpo.ipynb)）：

| 奖励组件 | 计算方式 | 作用 |
|---------|---------|------|
| `empty_cell_reward` | 目标是空格 +1 / 覆盖已有 -1 | 引导模型选择合法位置 |
| `valid_move_reward` | 合法新走法 +1 / 警告 -0.5 / 无效 0 | 鼓励合法操作 |
| `repetition_reward` | 重复走法指数惩罚（-2^n，上限 -10） | 避免重复 |
| `progress_reward` | (已填充 - 初始) / (81 - 初始) | 衡量解题进度 |
| `correct_reward` | 环境返回的二值奖励 | 完全解出 |

组合奖励 = 各组件均值之和，提供比单一二值奖励更密集的学习信号。

**3. Hints 系统**

每轮交互中，scheduler 解析当前棋盘状态，为模型提供提示：

- **GUARANTEED MOVES**：只有一个候选数字的空格（可直接填入）
- **Other options**：2-3 个候选数字的空格
- **MOVES ALREADY TRIED**：已尝试过的走法（避免重复）

**4. 准备数据集**

数据集仅作占位符，实际棋盘由环境服务器生成。`base_url` 指向 OpenEnv 托管地址：

```json
{"messages":[{"role":"user","content":"Play"}],"env_config":{"name":"openenv","base_url":"http://127.0.0.1:8000"}}
```

**5. 注册 Scheduler**

`sudoku_scheduler.py` 末尾已包含注册代码，通过 `--external_plugins` 加载即可：

```python
# sudoku_scheduler.py 末尾
from swift.rollout.multi_turn import multi_turns
multi_turns['sudoku_scheduler'] = SudokuScheduler
```

**6. 启动训练**

```bash
swift rlhf \
    --rlhf_type grpo \
    --model Qwen/Qwen3.5-4B \
    --dataset examples/train/grpo/plugin/openenv/sudoku.jsonl \
    --external_plugins examples/train/grpo/plugin/openenv/sudoku_scheduler.py \
    --enable_thinking false \
    --max_completion_length 256 \
    --use_gym_env true \
    --multi_turn_scheduler sudoku_scheduler \
    --max_turns 20 \
    --use_vllm true \
    --vllm_mode colocate \
    ...
```

运行脚本参考：[`examples/train/grpo/plugin/openenv/run_grpo_sudoku.sh`](https://github.com/modelscope/ms-swift/blob/main/examples/train/grpo/plugin/openenv/run_grpo_sudoku.sh)

### 注意事项

- **vLLM 模式**：以上示例使用 `--vllm_mode colocate`，vLLM 与训练共享 GPU。若使用 `--vllm_mode server`，需额外启动 `swift rollout` 作为 vLLM 服务器，且 `--multi_turn_scheduler` 和 `--max_turns` 参数应传给 `swift rlhf` 而非 `swift rollout`。
- **并发会话数**：`start_sudoku_server.py` 的 `MAX_CONCURRENT_ENVS` 需 ≥ 训练时的 `num_generations`。默认的 `python -m textarena_env.server.app` 只支持 1 个并发会话。
- **enable_thinking**：Sudoku 等环境不需要 CoT 推理，建议设置 `--enable_thinking false` 以减少 token 消耗。
- **同步 I/O**：`OpenEnvWrapper` 的 `reset()`/`step()` 是同步 WebSocket 调用。`OpenEnvScheduler` 的子类应使用 `asyncio.to_thread()` 包装这些调用以避免阻塞事件循环。


参考资料:

- https://gymnasium.farama.org/environments/toy_text/frozen_lake/
- https://github.com/alibaba/ROLL/tree/main/roll/pipeline/agentic/env/frozen_lake
- https://github.com/huggingface/openenv
- https://github.com/huggingface/trl/blob/main/examples/notebooks/openenv_sudoku_grpo.ipynb
