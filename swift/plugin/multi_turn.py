import asyncio
from abc import ABC
from copy import deepcopy
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

from swift.plugin import ContextManager, Env, context_managers, envs
from swift.utils import remove_response

if TYPE_CHECKING:
    from swift.llm.infer.protocol import (ChatCompletionResponse, ChatCompletionResponseChoice, RequestConfig,
                                          RolloutOutput)
    from swift.llm.template import RolloutInferRequest
    from swift.llm.infer.infer_engine import GRPOVllmEngine
    from swift.llm.utils import Messages


class RolloutScheduler(ABC):
    # Single Turn Rollout Scheduler
    def __init__(self,
                 infer_engine: Optional['GRPOVllmEngine'] = None,
                 max_turns: Optional[int] = None,
                 *args,
                 **kwargs):
        self.infer_engine = infer_engine
        self.max_turns = max_turns

    async def async_infer(self,
                          infer_requests: List[Union['RolloutInferRequest', Dict[str, Any]]],
                          request_config: 'RequestConfig',
                          *,
                          use_tqdm: Optional[bool] = None,
                          **kwargs) -> List['ChatCompletionResponse']:
        assert request_config.n == 1

        async def _infer_async_single(infer_request: Union['RolloutInferRequest', Dict[str, Any]],
                                      request_config: 'RequestConfig', **kwargs):
            from swift.llm.template import RolloutInferRequest
            if isinstance(infer_request, Dict):
                infer_request = RolloutInferRequest(**infer_request)

            return await self.run(infer_request, request_config, **kwargs)

        tasks = [_infer_async_single(infer_request, request_config, **kwargs) for infer_request in infer_requests]
        if use_tqdm is None:
            use_tqdm = len(infer_requests) > 1
        return await self.infer_engine._batch_infer_stream(tasks, request_config.stream, use_tqdm)

    async def run(self, infer_request: 'RolloutInferRequest', request_config: 'RequestConfig',
                  **kwargs) -> 'RolloutOutput':
        response: 'ChatCompletionResponse' = await self.infer_engine.infer_async(infer_request, request_config,
                                                                                 **kwargs)
        response_token_ids = response.choices[0].token_ids
        response_loss_mask = [1] * len(response_token_ids)
        return RolloutOutput(
            response=response,
            messages=infer_request.messages,
            response_token_ids=[response_token_ids],
            response_loss_mask=[response_loss_mask],
            rollout_infos={'num_turns': 1})

    def __getattr__(self, key: str):
        try:
            return object.__getattribute__(self, key)
        except AttributeError:
            pass

        try:
            infer_engine = object.__getattribute__(self, 'infer_engine')
            if hasattr(infer_engine, key):
                return getattr(infer_engine, key)

        except AttributeError:
            raise AttributeError(f'{type(self).__name__} object has no attribute {key}')

    @property
    def engine(self):
        return self.infer_engine


class MultiTurnScheduler(RolloutScheduler, ABC):
    """
    Abstract base class for multi-turn rollout scheduling.

    Provides default implementation for multi-turn conversation management with two customization approaches:

    1. FULL CUSTOMIZATION:
       Override the `run()` method to implement completely custom multi-turn logic.
       - Gives full control over the rollout process
       - Must handle all turn management and termination logic

    2. PARTIAL CUSTOMIZATION:
       Implement the required `step()` method and optionally override `check_finished()`
       - Uses MultiTurnScheduler's run() method infrastructure
       - Only need to implement turn transition logic in step()
       - Optionally customize termination conditions

    Note: You must implement at least one of these approaches in your subclass.

    Options:
        - If each round's response_token_ids are included in the RolloutOutput,
          the Trainer can skip encoding the completion text into token_ids when calculating loss.
          This avoids potential training inconsistencies due to asymmetric encode/decode behavior.
          See: https://github.com/0russwest0/Agent-R1/issues/30#issuecomment-2826155367

        - If both response_token_ids and response_loss_mask are returned in the RolloutOutput,
          you can manually control the loss mask for each token.
          The Trainer will use the provided loss_mask values directly when computing the loss.
          Note: Returning response_loss_mask requires that response_token_ids are also returned,
          as the two must be aligned in length for correct loss computation.

        You can refer to MathTipsScheduler as an example of how to use response_token_ids and response_loss_mask.

    Loss mask configuration:
        During rollout, some parts of the completion (e.g., environment observations embedded in completion)
        may need to be masked out from loss computation.
        There are two supported strategies:

        1. Use the built-in `loss_scale` parameter in ms-swift and do not return response token ids.
        2. Return response_token_ids along with a corresponding response_loss_mask (of equal length) to indicate the loss mask for each token. # noqa
    """

    async def run(self, infer_request: 'RolloutInferRequest', request_config: 'RequestConfig',
                  **kwargs) -> Union['RolloutOutput', List['RolloutOutput']]:
        """Execute multi-turn conversation rollout with built-in turn management logic.

        This implements the default multi-turn interaction flow, which you can override
        to customize the conversation handling behavior. The default logic:

        1. Manages conversation turns and stopping conditions
        2. Handles message accumulation across turns
        3. Tracks response tokens and loss masks
        4. Supports early stopping conditions

        Args:
            infer_request: The initial inference request containing messages
            request_config: Configuration for the inference request
            **kwargs: Additional inference parameters

        Returns:
            RolloutOutput containing the complete conversation history and metadata,
            or a list of outputs for batched requests

        Customization Points:
            - Override check_finished() to change stopping conditions
            - Override step() to customize turn-to-turn transitions
            - Subclass to completely change multi-turn behavior

        Example:
            class CustomScheduler(MultiTurnScheduler):
                async def run(self, *args, **kwargs):
                    # Custom multi-turn logic here
                    ...
        """

        current_request = infer_request
        current_turn = 1
        rollout_infos = {}
        total_response_ids = []
        total_response_loss_mask = []
        while True:
            messages = current_request.messages
            if current_turn == 1 or not messages[-1]['content']:
                # If it's the first turn or the last message content is empty(dummy), remove the response
                remove_response(messages)

            # Get model response
            response: 'ChatCompletionResponse' = await self.infer_engine.infer_async(
                current_request, request_config, **kwargs)
            response_choice: 'ChatCompletionResponseChoice' = response.choices[0]

            # Update conversation history
            completion = response_choice.message.content
            if messages[-1]['role'] == 'assistant':
                messages[-1]['content'] += completion
            else:
                messages.append({'role': 'assistant', 'content': completion})

            # Check stopping conditions
            should_stop = self.check_finished(current_request, response_choice, current_turn)

            if self.max_turns:
                should_stop = should_stop or (current_turn >= self.max_turns)

            if should_stop:
                return RolloutOutput(
                    response=response,
                    messages=messages,
                    respones_token_ids=total_response_ids,
                    response_loss_mask=total_response_loss_mask,
                    rollout_infos=rollout_infos,
                )

            # Prepare next turn
            ret = self.step(current_request, response_choice, current_turn)
            current_request: 'RolloutInferRequest' = ret['infer_request']

            # Track response tokens and masks
            return_token_id = False
            if 'response_token_ids' in ret:
                total_response_ids.append(ret['response_token_ids'])
                return_token_id = True

            if 'response_loss_mask' in ret:
                assert return_token_id, 'You must return response_token_ids if you want to return response_loss_mask'
                assert len(ret['response_loss_mask']) == len(ret['response_token_ids']), \
                    'response_loss_mask must have the same length as response_token_ids'
                total_response_loss_mask.append(ret['response_loss_mask'])

            if 'rollout_infos' in ret:
                rollout_infos = {**rollout_infos, **ret['rollout_infos']}

            if current_request.messages[-1]['role'] == 'assistant':
                # Add a dummy response to allow engine to continue generating
                current_request.messages.append({'role': 'assistant', 'content': None})

            current_turn += 1

    def step(self, infer_request: 'RolloutInferRequest', response_choice: 'ChatCompletionResponseChoice',
             current_turn: int) -> Dict:
        """
        Handles transition between conversation turns.

        Args:
            infer_request: Current inference request
            response_choice: Response from current turn
            current_turn: Current turn number

        Returns:
            Dict[str, Any]: A dictionary containing inference results with the following structure:
                - infer_request (required): Main inference request object
                - response_token_ids (Optional[List[List[int]]]): Token IDs of responses for each rollout turn
                - response_loss_scale (Optional[List[List[int]]]): Loss scaling factors for responses in each rollout turn # noqa
                - rollout_infos (Optional[Dict[str, Any]]): Additional metadata (must be serializable)

        """
        raise NotImplementedError(
            'Please implement the `step` method in your MultiTurnScheduler subclass, or override the `run` method.')

    def check_finished(self, infer_request: 'RolloutInferRequest', response_choice: 'ChatCompletionResponseChoice',
                       current_turn: int) -> bool:
        """
        Default termination logic for checking if a multi-turn rollout should end.

        This method is invoked by:
        - The base class MultiTurnScheduler.run() method, OR
        - Custom run() methods when explicitly called

        Note: This is the default implementation that can be overridden by subclasses for custom termination logic.

        Termination Conditions:
        1. When response hits length limit (finish_reason == 'length')
        2. When conversation reaches max_turns (if max_turns is set)

        Args:
            infer_request: The inference request object
            response_choice: Contains generation results including finish_reason
            current_turn: Current conversation turn count

        Returns:
            bool: True to terminate conversation, False to continue
        """
        if response_choice.finish_reason == 'length':
            return True
        if self.max_turns and current_turn >= self.max_turns:
            return True
        return False


class ThinkingModelScheduler(MultiTurnScheduler):
    """
    Scheduler for Thinking class models that handle multi-turn reasoning.

    For Thinking models, the assistant response format is:
    "<think> think content </think> <answer> answer content </answer>"

    This scheduler:
    1. Parses think and answer content from assistant responses
    2. Only keeps the think content from the last round
    3. Processes each round's history separately
    4. Returns List[RolloutOutput] with one output per round
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _parse_think_answer(self, content: str) -> tuple[str, str]:
        """
        Parse think and answer content from assistant response.

        Args:
            content: Assistant response content

        Returns:
            tuple: (think_content, answer_content)
        """
        think_content = ''
        answer_content = ''

        # Parse think content
        think_start = content.find('<think>')
        think_end = content.find('</think>')
        if think_start != -1 and think_end != -1:
            think_content = content[think_start + 7:think_end].strip()

        # Parse answer content
        answer_start = content.find('<answer>')
        answer_end = content.find('</answer>')
        if answer_start != -1 and answer_end != -1:
            answer_content = content[answer_start + 8:answer_end].strip()

        return think_content, answer_content

    def _is_thinking_template(self) -> bool:
        """
        Check if the model's template is a ThinkingTemplate.

        Returns:
            bool: True if the template is a ThinkingTemplate or its subclass
        """
        if not hasattr(self.infer_engine, 'default_template'):
            return False

        template = self.infer_engine.default_template
        from swift.llm.template.template.utils import ThinkingTemplate

        return isinstance(template, ThinkingTemplate)

    def _build_round_history(self, original_messages: 'Messages', round_num: int, think_content: str) -> 'Messages':
        """
        Build history for a specific round, keeping only the think content from the last round.

        Args:
            original_messages: Original conversation messages
            round_num: Current round number
            think_content: Think content to include

        Returns:
            Messages: History for this specific round
        """
        from copy import deepcopy

        # If this is a thinking template, use the template's method to prepare messages
        if self._is_thinking_template():
            # Create a mock inputs object to use the template's _swift_prepare_inputs method
            class MockInputs:

                def __init__(self, messages):
                    self.messages = deepcopy(messages)

            mock_inputs = MockInputs(original_messages)

            # Set up the template for inference mode
            template = self.infer_engine.default_template
            original_is_training = getattr(template, 'is_training', False)
            template.is_training = False

            # Use the template's method to prepare messages
            template._swift_prepare_inputs(mock_inputs)

            # Restore original training state
            template.is_training = original_is_training

            return mock_inputs.messages
        else:
            # Fallback to manual processing for non-thinking templates
            round_messages = []

            # Process messages in original order
            for i, msg in enumerate(original_messages):
                if msg['role'] == 'assistant' and isinstance(msg['content'], str) and i != len(original_messages) - 1:
                    # For assistant messages
                    assistant_no_think = msg['content'].split('</think>')[-1].strip()
                    round_messages.append(assistant_no_think)
                else:
                    round_messages.append(deepcopy(msg))

            return round_messages

    async def run(self, infer_request: 'RolloutInferRequest', request_config: 'RequestConfig',
                  **kwargs) -> List['RolloutOutput']:
        """
        Execute multi-turn conversation for Thinking models.

        Args:
            infer_request: The initial inference request containing messages
            request_config: Configuration for the inference request
            **kwargs: Additional inference parameters

        Returns:
            List[RolloutOutput]: List of outputs, one for each round
        """
        from swift.llm.infer.protocol import RolloutOutput

        current_request = infer_request
        current_turn = 1
        rollout_outputs = []
        last_think_content = ''

        while True:
            messages = current_request.messages
            if current_turn == 1 or not messages[-1]['content']:
                # If it's the first turn or the last message content is empty(dummy), remove the response
                remove_response(messages)

            # Get model response
            response: 'ChatCompletionResponse' = await self.infer_engine.infer_async(
                current_request, request_config, **kwargs)
            response_choice: 'ChatCompletionResponseChoice' = response.choices[0]

            # Parse think and answer content
            completion = response_choice.message.content
            think_content, answer_content = self._parse_think_answer(completion)

            # Update last think content
            if think_content:
                last_think_content = think_content

            # Update conversation history
            if messages[-1]['role'] == 'assistant':
                messages[-1]['content'] += completion
            else:
                messages.append({'role': 'assistant', 'content': completion})

            # Build history for this round
            round_history = self._build_round_history(messages, current_turn, last_think_content)

            # Create RolloutOutput for this round
            round_output = RolloutOutput(
                response=response,
                messages=round_history,
                rollout_infos={
                    'num_turns': current_turn,
                    'think_content': think_content,
                    'answer_content': answer_content,
                    'round_number': current_turn
                })
            rollout_outputs.append(round_output)

            # Check stopping conditions
            should_stop = self.check_finished(current_request, response_choice, current_turn)

            if self.max_turns:
                should_stop = should_stop or (current_turn >= self.max_turns)

            if should_stop:
                break

            # Prepare next turn
            ret = self.step(current_request, response_choice, current_turn)
            current_request: 'RolloutInferRequest' = ret['infer_request']

            if current_request.messages[-1]['role'] == 'assistant':
                # Add a dummy response to allow engine to continue generating
                current_request.messages.append({'role': 'assistant', 'content': None})

            current_turn += 1

        return rollout_outputs


class MathTipsScheduler(MultiTurnScheduler):
    tips_prompt = 'But wait... It seems I made a mistake,'

    def __init__(self, tokenizer, *args, **kwargs):
        from .orm import MathAccuracy
        self.tokenizer = tokenizer
        super().__init__(*args, **kwargs)
        self.acc_func = kwargs.get('acc_function', MathAccuracy())

    def check_finished(self, infer_request: 'RolloutInferRequest', response_choice: 'ChatCompletionResponseChoice',
                       current_turn: int) -> bool:
        last_completion = infer_request.messages[-1]['content']
        # we only give tips once
        if self.tips_prompt in last_completion:
            return True
        solution = infer_request.data_dict['solution']

        acc = self.acc_func([last_completion], [solution])[0]
        if acc == 1:
            return True

        return super().check_finished(infer_request, response_choice, current_turn)

    def step(self, infer_request: 'RolloutInferRequest', response_choice: 'ChatCompletionResponseChoice',
             current_turn: int) -> Dict:
        completion = response_choice.message.content
        if '<answer>' in completion:
            completion = completion[:completion.index('<answer>')]
        if '</think>' in completion:
            completion = completion[:completion.index('</think>')]
        completion += self.tips_prompt
        if infer_request.messages[-1]['role'] == 'assistant':
            if not infer_request.messages[-1]['content']:
                # Multi-turn continuation: pop the dummy input we add in last turn
                infer_request.messages.pop(-1)
            infer_request.messages[-1]['content'] = completion
        else:
            infer_request.messages.append({'role': 'assistant', 'content': completion})

        return {'infer_request': infer_request}


class MathTipsMultiTurnScheduler(MultiTurnScheduler):
    from .orm import MathAccuracy
    tips_prompt = 'The answer is not correct, It seems You made a mistake, you need to recheck very carefully.'
    acc_func = MathAccuracy()

    def check_finished(self, infer_request: 'RolloutInferRequest', response_choice: 'ChatCompletionResponseChoice',
                       current_turn: int) -> bool:

        last_query = infer_request.messages[-2]['content']
        # we only give tips once
        if self.tips_prompt in last_query:
            return True

        completion = response_choice.message.content
        solution = infer_request.data_dict['solution']
        acc = self.acc_func([completion], [solution])[0]
        if acc == 1:
            return True

        return super().check_finished(infer_request, response_choice, current_turn)

    def step(self, infer_request: 'RolloutInferRequest', response_choice: 'ChatCompletionResponseChoice',
             current_turn: int) -> Dict:
        infer_request.messages.append({'role': 'user', 'content': self.tips_prompt})

        return {'infer_request': infer_request}


class GYMScheduler(RolloutScheduler):

    def __init__(self,
                 infer_engine: 'GRPOVllmEngine',
                 gym_env: Optional[str] = None,
                 context_manager_name: Optional[str] = None,
                 max_turns: Optional[int] = None,
                 **kwargs):
        super().__init__(infer_engine, max_turns, **kwargs)
        self.gym_env_name = gym_env
        self.context_manager_name = context_manager_name

    async def _create_env(self, env_config: Dict) -> Env:
        """Create environment instance from configuration."""
        env_name = env_config.get('name', self.gym_env_name)
        if env_name not in envs:
            raise ValueError(f"Environment '{env_name}' not found. Available: {list(envs.keys())}")
        return envs[env_name](env_config)

    async def _create_context_manager(self, ctx_config: Dict) -> ContextManager:
        """Create context manager from configuration."""
        ctx_name = ctx_config.get('name', self.context_manager_name)

        if not ctx_name:
            ctx_name = 'dummyContextManager'

        return context_managers[ctx_name](ctx_config)

    async def _close_env_async(self, env: Env):
        """Safely close environment with async support."""
        try:
            if hasattr(env, 'close') and asyncio.iscoroutinefunction(env.close):
                await env.close()
            elif hasattr(env, 'close'):
                env.close()
        except Exception:
            # Handle any exceptions during environment closure
            pass

    async def run(self, infer_request: 'RolloutInferRequest', request_config: 'RequestConfig',
                  **kwargs) -> 'ChatCompletionResponse':
        from swift.llm.infer.protocol import ChatCompletionResponse, ChatCompletionResponseChoice
        """
        Execute the gym environment-based rollout:
        1. Initialize environment and context manager
        2. Run multi-turn interactions between LLM and environment
        3. Collect trajectory information and rewards
        """
        # Extract configurations from request
        env_config = infer_request.data_dict.get('env_config', {})
        ctx_config = infer_request.data_dict.get('ctx_config', {})

        # Create environment and context manager
        env = await self._create_env(env_config)
        context_manager = await self._create_context_manager(ctx_config)

        try:
            # Initialize environment
            observation, info, system_message = await env.reset(infer_request)

            # Build initial messages
            messages: 'Messages' = []
            if system_message:
                messages.append({'role': 'system', 'content': system_message})
            messages.append({'role': 'user', 'content': observation})

            current_request = deepcopy(infer_request)
            current_turn = 1
            done = False
            total_reward = 0.0
            step_rewards = []
            trajectory_id = f'{id(infer_request)}_{hash(str(infer_request))}'
            trajectory_info = [info]

            while not done and current_turn <= (self.max_turns or float('inf')):
                # Apply context management (e.g., history compression)
                messages = context_manager.manage_context(messages, trajectory_id)
                current_request.messages = messages
                remove_response(current_request.messages)

                response: 'ChatCompletionResponse' = await self.infer_engine.infer_async(
                    current_request, request_config, **kwargs)
                response_choice: 'ChatCompletionResponseChoice' = response.choices[0]
                completion = response_choice.message.content
                messages.append({'role': 'assistant', 'content': completion})

                # Execute environment step
                next_obs, reward, done, step_info = await env.step(deepcopy(messages))

                # Update trajectory information
                total_reward += reward
                step_rewards.append(reward)
                trajectory_info.append(step_info)

                # Prepare for next turn
                if not done:
                    messages.append({'role': 'user', 'content': next_obs})
                    current_request.messages = messages
                    current_turn += 1

            final_choice = ChatCompletionResponseChoice(
                index=response_choice.index,
                message=response_choice.message,
                finish_reason=response_choice.finish_reason,
                logprobs=response_choice.logprobs)

            last_response = ChatCompletionResponse(
                model=self.infer_engine.model_name,
                choices=[final_choice],
                usage=response.usage,
                id=f'gym_{trajectory_id}')

            return RolloutOutput(
                response=last_response,
                messages=messages,
                rollout_infos={
                    'num_turns': current_turn,
                    'trajectory_id': trajectory_id,
                    'total_reward': total_reward,
                    'step_rewards': step_rewards,
                    'trajectory_info': trajectory_info
                })

        finally:
            # Ensure environment is properly closed
            await self._close_env_async(env)


multi_turns = {
    'base_scheduler': RolloutScheduler,
    'math_tip_trick': MathTipsScheduler,
    'math_tip_trick_multi_turn': MathTipsMultiTurnScheduler,
    'gym_scheduler': GYMScheduler,
    'thinking_scheduler': ThinkingModelScheduler,
}
