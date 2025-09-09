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
                          **kwargs) -> List['RolloutOutput']:
        """
        Perform asynchronous batched inference for multiple rollout requests.

        This method serves as the main entry point for multi-round training inference.
        It executes the `run` method for each inference request concurrently and
        aggregates the results into a single flattened list.

        Each inference request can be either a `RolloutInferRequest` instance or a
        dictionary that can be converted into one. The results from all requests are
        collected asynchronously using the underlying inference engine.

        Args:
            infer_requests (List[Union[RolloutInferRequest, Dict[str, Any]]]):
                A list of inference requests. Each request can be either:
                    - A `RolloutInferRequest` object.
                    - A dictionary containing the fields required to initialize a
                    `RolloutInferRequest`.
            request_config (RequestConfig):
                Configuration object specifying inference settings. Must satisfy
                `request_config.n == 1`, as only single-response generation is supported.
            use_tqdm (Optional[bool], optional):
                Whether to display a progress bar during batch inference.
                If `None`, it defaults to `True` when there are multiple requests,
                otherwise `False`.
            **kwargs:
                Additional arguments forwarded to the underlying `run` method.

        Returns:
            List[RolloutOutput]:
                A list of RolloutOutput objects corresponding to the provided inference requests.

        Raises:
            AssertionError:
                If `request_config.n` is not equal to `1`.

        Notes:
            - Internally, this method converts dict-based requests into
            `RolloutInferRequest` instances.
            - Uses `infer_engine._batch_infer_stream` to perform concurrent execution.
            - The returned list is guaranteed to be flattened, even if individual
            tasks return lists of responses.
        """

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
        # Execute all tasks and flatten the results
        results = await self.infer_engine._batch_infer_stream(tasks, request_config.stream, use_tqdm, None)
        # Flatten the results since each task may return a list
        flattened_results = []
        for result in results:
            if isinstance(result, list):
                flattened_results.extend(result)
            else:
                flattened_results.append(result)
        return flattened_results

    async def run(self, infer_request: 'RolloutInferRequest', request_config: 'RequestConfig',
                  **kwargs) -> 'RolloutOutput':
        from swift.llm.infer.protocol import RolloutOutput
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
            if hasattr(infer_engine.engine, key):
                return getattr(infer_engine.engine, key)

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

        This implements the default multi-turn interaction flow that can be overridden
        to customize conversation handling behavior. The default logic provides:

        1. Automatic conversation turn management and stopping conditions
        2. Seamless message accumulation across multiple turns
        3. Response token tracking and loss mask management
        4. Configurable early stopping mechanisms

        Args:
            infer_request: The initial inference request containing conversation messages
            request_config: Configuration parameters for the inference request
            **kwargs: Additional inference parameters passed to the engine

        Returns:
            RolloutOutput containing the complete conversation history and metadata,
            or a list of outputs for batched requests

        Customization Approaches:
            - Override check_finished() to implement custom stopping criteria
            - Override step() to customize turn-to-turn transition logic
            - Override this entire run() method for completely custom multi-turn behavior

        Important Notes:
            - Method overriding is only supported when using server mode (swift rollout)
              with vllm_use_async_engine=True
            - Custom implementations must maintain async/await compatibility
            - Ensure proper handling of conversation state across turns

        Example:
            class CustomScheduler(MultiTurnScheduler):
                async def run(self, infer_request, request_config, **kwargs):
                    # Implement custom multi-turn conversation logic
                    # Must return RolloutOutput or List[RolloutOutput]
                    ...
        """
        from swift.llm.infer.protocol import RolloutOutput
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
            is_continuation = False
            if messages[-1]['role'] == 'assistant':
                messages[-1]['content'] += completion
                is_continuation = True
            else:
                messages.append({'role': 'assistant', 'content': completion})

            # Check stopping conditions
            should_stop = self.check_finished(current_request, response_choice, current_turn)

            # double-check if user forget to judge the max_turns
            if self.max_turns:
                should_stop = should_stop or (current_turn >= self.max_turns)

            if should_stop:
                if is_continuation and total_response_ids:
                    # for continuation and total_response_ids is not empty
                    # we need to extend the last turn's response_token_ids and response_loss_mask
                    total_response_ids[-1].extend(response_choice.token_ids)
                    if total_response_loss_mask:
                        total_response_loss_mask[-1].extend([1] * len(response_choice.token_ids))

                return RolloutOutput(
                    response=response,
                    messages=messages,
                    response_token_ids=total_response_ids,
                    response_loss_mask=total_response_loss_mask,
                    rollout_infos={
                        **rollout_infos, 'num_turns': current_turn
                    },
                )

            # Prepare next turn
            ret = self.step(current_request, response_choice, current_turn)
            current_request: 'RolloutInferRequest' = ret['infer_request']

            # Track response tokens and masks
            return_token_id = False
            if 'response_token_ids' in ret:
                if is_continuation and total_response_ids:
                    total_response_ids[-1].extend(ret['response_token_ids'])
                else:
                    total_response_ids.append(ret['response_token_ids'])
                return_token_id = True

            if 'response_loss_mask' in ret:
                assert return_token_id, 'You must return response_token_ids if you want to return response_loss_mask'
                assert len(ret['response_loss_mask']) == len(ret['response_token_ids']), \
                    'response_loss_mask must have the same length as response_token_ids'
                if is_continuation and total_response_loss_mask:
                    total_response_loss_mask[-1].extend(ret['response_loss_mask'])
                else:
                    total_response_loss_mask.append(ret['response_loss_mask'])

            if 'rollout_infos' in ret:
                # Always overwrite the rollout info for this step.
                # If you need to keep all step-wise details, switch to append or merge instead.
                rollout_infos.update(ret['rollout_infos'])

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


class ThinkingModelTipsScheduler(MultiTurnScheduler):
    """
    Scheduler for multi-turn reasoning with Thinking class models.

    Key Features:
    1. Parses both "think" and "answer" content from each assistant response.
    2. For each round, only the "think" content from the last round is retained in the message history.
    3. Each round's conversation history is processed independently.
    4. Returns a list of RolloutOutput objects, one for each round.
    5. Please set `--loss_scale last_round` for training last round response.

    The scheduler will automatically inject a tip prompt if the answer is incorrect, encouraging the model to recheck its reasoning. # noqa
    """
    from .orm import MathAccuracy
    tips_prompt = 'The answer is not correct, It seems You made a mistake, you need to recheck very carefully.'
    acc_func = MathAccuracy()

    async def run(self, infer_request: 'RolloutInferRequest', request_config: 'RequestConfig',
                  **kwargs) -> List['RolloutOutput']:
        """
        Execute multi-turn inference for Thinking models.

        Args:
            infer_request (RolloutInferRequest): The initial inference request containing the conversation history.
            request_config (RequestConfig): Configuration for the inference request.
            **kwargs: Additional arguments for the inference engine.

        Returns:
            List[RolloutOutput]: A list of RolloutOutput objects, one for each reasoning round.
        """
        from swift.llm.infer.protocol import RolloutOutput

        current_request = infer_request
        current_turn = 1
        rollout_outputs = []

        while True:
            messages = current_request.messages
            # Obtain model response for the current turn
            response: 'ChatCompletionResponse' = await self.infer_engine.infer_async(
                current_request, request_config, **kwargs)
            response_choice: 'ChatCompletionResponseChoice' = response.choices[0]
            completion = response_choice.message.content

            # Append the assistant's response to the message history
            messages.append({'role': 'assistant', 'content': completion})

            # Construct the message history for this round, keeping only the last "think" content
            messages_with_last_think = self._build_messages(messages)

            # Create a RolloutOutput for the current round
            round_output = RolloutOutput(
                response=response,
                messages=messages_with_last_think,
                response_token_ids=response_choice.token_ids,
                rollout_infos={'num_turns': current_turn})
            # Store the output for this round
            rollout_outputs.append(round_output)

            # Determine whether to stop the multi-turn reasoning
            should_stop = self.check_finished(current_request, response_choice, current_turn)

            if should_stop:
                break

            # Prepare for the next turn by updating the inference request
            ret = self.step(current_request, response_choice, current_turn)
            current_request: 'RolloutInferRequest' = ret['infer_request']
            current_turn += 1

        return rollout_outputs

    def check_finished(self, infer_request: 'RolloutInferRequest', response_choice: 'ChatCompletionResponseChoice',
                       current_turn: int) -> bool:

        last_query = infer_request.messages[-2]['content']
        # tips once
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

    def _is_thinking_template(self) -> bool:
        if not hasattr(self.infer_engine, 'default_template'):
            return False

        template = self.infer_engine.default_template
        from swift.llm.template.template.utils import ThinkingTemplate

        return isinstance(template, ThinkingTemplate)

    def _build_messages(self, original_messages: 'Messages') -> 'Messages':
        """
        Build history for a specific round, keeping only the think content from the last round.

        Args:
            original_messages: Original conversation messages

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
            # _swift_prepare_inputs will remove historical thinking content when in train mode, patch the mode here
            original_mode = template.mode
            template.mode = 'train'
            # Use the template's method to prepare messages
            template._swift_prepare_inputs(mock_inputs)
            # Restore original mode
            template.mode = original_mode

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


class MathTipsScheduler(MultiTurnScheduler):
    tips_prompt = 'But wait... It seems I made a mistake,'

    def __init__(self, *args, **kwargs):
        from .orm import MathAccuracy
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


class GYMScheduler(RolloutScheduler):

    def __init__(self, infer_engine: 'GRPOVllmEngine', max_turns: Optional[int] = None, **kwargs):
        super().__init__(infer_engine, max_turns, **kwargs)
        self.gym_env_name = kwargs.get('gym_env', None)
        self.context_manager_name = kwargs.get('context_manager', None)

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

        if ctx_name not in context_managers:
            raise ValueError(f"Context manager '{ctx_name}' not found. Available: {list(context_managers.keys())}")

        return context_managers[ctx_name](ctx_config)

    async def _close_env_async(self, env: Env):
        """Safely close environment with async support."""
        if env is None:
            return

        try:
            if hasattr(env, 'close') and asyncio.iscoroutinefunction(env.close):
                await env.close()
            elif hasattr(env, 'close'):
                env.close()
        except Exception as e:
            # Log the exception but don't raise it to avoid masking other errors
            import logging
            logging.warning(f'Error closing environment: {e}')

    async def run(self, infer_request: 'RolloutInferRequest', request_config: 'RequestConfig',
                  **kwargs) -> 'RolloutOutput':
        from swift.llm.infer.protocol import RolloutOutput
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
        env = None
        context_manager = None
        try:
            env = await self._create_env(env_config)
            context_manager = await self._create_context_manager(ctx_config)

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
            trajectory_id = infer_request.uuid
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

            return RolloutOutput(
                response=response,
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
            if env is not None:
                await self._close_env_async(env)


multi_turns = {
    'math_tip_trick': MathTipsScheduler,
    'gym_scheduler': GYMScheduler,
    'thinking_tips_scheduler': ThinkingModelTipsScheduler,
}
