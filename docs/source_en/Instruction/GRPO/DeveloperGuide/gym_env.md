# GYM Environment Training

GYM-style environment training wraps the "model → environment → reward" chain behind an abstract interface, letting the LLM interact with the environment as an Agent over multiple turns. The reward of each step is produced directly by the environment, so you don't need a separate reward function to infer it from the trajectory. This document first introduces the interface, then walks through a complete custom example (FrozenLake) showing how to plug it into training.

## Gym interface

GYM originates from the [Gymnasium library](https://github.com/Farama-Foundation/Gymnasium). In ms-swift we define the following interface:

```python
class Env(ABC):

    def __init__(self, env_config):
        """env_config comes from the env_config column of each dataset row and carries initialization arguments."""
        self.env_config = env_config

    @abstractmethod
    async def reset(self, config: RolloutInferRequest) -> Tuple[str, Dict[str, Any], str]:
        """
        Returns:
            - observation: sent to the model as the first user message
            - info: debug/log information, recorded in completions.jsonl
            - system_message: system prompt for this trajectory
        """
        pass

    @abstractmethod
    async def step(self, action: Messages) -> Tuple[str, float, bool, Dict[str, Any]]:
        """
        Args:
            action: the complete conversation messages so far; the last one is the model's latest reply
        Returns:
            - next_observation: next user message
            - reward: reward for the current step
            - done: whether the trajectory is finished
            - info: debug/log information
        """
        pass

    @abstractmethod
    async def close(self):
        """Release resources."""
        pass
```

The `RolloutInferRequest` received by `reset` contains the dataset row's `messages`, `data_dict` (extra columns including `env_config`), etc. See the [input example](./multi_turn.md#multiturnscheduler) for the full structure.

> If you need extra control over the conversation history between turns (e.g. dynamic compression, injecting hints), subclass `MultiTurnScheduler` and implement `on_trajectory_start` / `on_turn_end` hooks, or override `step` / `run` — see the [multi-turn doc](./multi_turn.md#customising-the-interaction-logic).

## Launching training

Use the built-in [gym_scheduler](https://github.com/modelscope/ms-swift/blob/main/swift/rollout/multi_turn.py) to wire the env into multi-turn rollout.

`GYMScheduler` is based on the generic hook protocol:
- Inherits `MultiTurnScheduler` — no need to override the `run` method
- Implements `on_trajectory_start` (calls `env.reset`) and `on_turn_end` (calls `env.step`)
- Works with both server mode (`run()`) and colocate mode (`run_multi_turn()`)

User-defined envs are loaded via `--external_plugins your_plugin.py`; the plugin runs `envs['my_env'] = MyEnv` to register them (the FrozenLake example below demonstrates the full pattern).

The built-in `GYMScheduler` completes the control logic via hooks:

```python
class GYMScheduler(MultiTurnScheduler):
    def on_trajectory_start(self, requests):
        # Create an env for each request, call env.reset, inject initial observation
        for req in requests:
            env = self._create_env(req.data_dict.get('env_config', {}))
            observation, info, system_message = env.reset(req)
            req.messages = [system_msg, user_msg(observation)]
            self._envs[req.uuid] = env

    def on_turn_end(self, req, response_choice, current_turn):
        # Call env.step, accumulate reward, return done + rollout_infos
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
        # Inject the next observation into a user message
        if self._pending_obs.get(req.uuid):
            req.messages.append({'role': 'user', 'content': next_obs})
        return {'infer_request': req}
```

**Colocate mode**:

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

# swift rlhf works the same way
```


**Server mode**

```bash
swift rollout \
    --model xxx \
    --use_gym_env true \
    --external_plugins examples/megatron/grpo/multi_turn/frozen_lake_plugin.py \
    --multi_turn_scheduler gym_scheduler \
    --gym_env frozen_lake \
    --max_turns 10

# On the trainer side, add --vllm_server_pass_dataset true so the env_config column reaches the rollout server.
megatron rlhf --vllm_mode server --vllm_server_pass_dataset true ...
# or swift rlhf --vllm_mode server --vllm_server_pass_dataset true ...
```

Two ways to select the environment:
- Set it globally via `--gym_env env_name` (recommended — one env for the whole script);
- Or specify it per dataset row via `env_config.name` (for mixed-env workloads; overrides `--gym_env`).

## Example: writing a FrozenLake environment from scratch

<img src="https://gymnasium.farama.org/_images/frozen_lake.gif" width="220" alt="FrozenLake environment (image from Gymnasium docs)" />

[FrozenLake](https://gymnasium.farama.org/environments/toy_text/frozen_lake/) is a classic task from OpenAI Gym: the agent starts at the start cell, must cross a frozen lake to reach the goal, and avoid holes along the way. The original environment is illustrated above. The walkthrough below uses a text-only version of it (the same grid rendered as ASCII).

Full source: [frozen_lake_plugin](https://github.com/modelscope/ms-swift/blob/main/examples/megatron/grpo/multi_turn/frozen_lake_plugin.py).

**1. Define the Env**

Each dataset row produces a freshly generated random 4x4 map (random holes + random S/G positions, BFS-validated to be solvable). Cell meanings: `S` start / `G` goal / `H` hole (stepping in = fail) / `F` safe ice / `P` player's current position.

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
        # Advance one cell, check G / H; the outer max_turns is enforced by the scheduler.
        if cell == 'G': return obs, 1.0, True, {'status': 'goal'}
        if cell == 'H': return obs, 0.0, True, {'status': 'hole'}
        ...
```

**2. Register**

Hook the env class into swift's `envs` registry. `--external_plugins` imports the file at startup, so the registration takes effect automatically:

```python
# examples/megatron/grpo/multi_turn/frozen_lake_plugin.py
from swift.rollout.gym_env import Env, envs

class FrozenLakeEnv(Env):
    ...

envs['frozen_lake'] = FrozenLakeEnv
```

**3. Prepare the dataset**

The dataset is just a placeholder here — the actual data is constructed by the env, with `env_config.seed` controlling map-generation randomness:

```json
{"messages":[{"role":"user","content":"<placeholder>"}],"env_config":{"seed":0}}
{"messages":[{"role":"user","content":"<placeholder>"}],"env_config":{"seed":1}}
...
{"messages":[{"role":"user","content":"<placeholder>"}],"env_config":{"seed":127}}
```

**4. (Optional) Blend in extra rewards**

With `--use_gym_env true`, the env-provided `total_reward` is automatically added as one reward column — no reward function is required. To mix in additional signals (e.g. format/length checks), just pass them via `--reward_funcs`; the gym reward is appended as an extra column and blended with the reward_funcs through `--reward_weights`. For example, also enabling a format reward:

```bash
megatron rlhf ... --use_gym_env true --reward_funcs format --reward_weights 0.2 1.0
# the last entry of reward_weights corresponds to the gym total_reward
```

**5. Train**

Runnable script: [`examples/megatron/grpo/multi_turn/frozen_lake.sh`](https://github.com/modelscope/ms-swift/blob/main/examples/megatron/grpo/multi_turn/frozen_lake.sh)

During training, observe `rollout_infos.num_turns` (steps per trajectory) and the reward mean in the logs. `--log_completions true` writes full conversations to `completions.jsonl`, so you can verify the model outputs in the `<action>...</action>` format turn by turn.

References:

- https://gymnasium.farama.org/environments/toy_text/frozen_lake/
- https://github.com/alibaba/ROLL/tree/main/roll/pipeline/agentic/env/frozen_lake
