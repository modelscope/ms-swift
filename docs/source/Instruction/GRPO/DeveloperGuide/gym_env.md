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


参考资料:

- https://gymnasium.farama.org/environments/toy_text/frozen_lake/
- https://github.com/alibaba/ROLL/tree/main/roll/pipeline/agentic/env/frozen_lake
