"""
EvalScope integration utilities for ms-swift models.

This module provides a custom ModelAPI implementation that enables batch inference
for evaluation tasks using ms-swift's PtEngine. It implements an asynchronous
batch processing system to improve throughput when evaluating models.
"""

from concurrent.futures import Future
from dataclasses import dataclass
from queue import Empty, Queue
from threading import Thread
from typing import Any, Dict, List, Optional, Tuple, Union

from evalscope.api.messages import ChatMessage as EvalChatMessage
from evalscope.api.model import GenerateConfig, ModelAPI, ModelOutput, ModelUsage
from evalscope.api.registry import register_model_api
from evalscope.api.tool import ToolChoice, ToolInfo
from evalscope.models.utils.openai import chat_choices_from_openai

from ..infer import PtEngine, RequestConfig
from ..template import InferRequest


@dataclass
class BatchInferInput:
    """
    Container for batch inference input data.

    Holds all necessary information for a single inference request
    that will be processed as part of a batch.
    """
    ms_input: InferRequest  # ms-swift format request
    ms_config: RequestConfig  # ms-swift format configuration
    batch_size: int  # desired batch size for this request
    engine: PtEngine  # inference engine to use


@dataclass
class _QueueItem:
    """
    Internal queue item for batch processing.

    Pairs a batch input with its corresponding future for result delivery.
    """
    input: BatchInferInput
    future: Future[ModelOutput]  # will be resolved with the inference result


# Global variables for batch processing
# These maintain the shared batch processing infrastructure across all model instances
batch_thread: Optional[Thread] = None  # background thread for processing batches
batch_queue: Queue[_QueueItem] = Queue()  # queue of pending inference requests


@register_model_api('swift_custom')
class EvalModel(ModelAPI):
    """
    Custom ModelAPI implementation for ms-swift models with batch inference support.

    This class integrates ms-swift's PtEngine with EvalScope's evaluation framework,
    providing efficient batch processing for improved evaluation throughput.
    """

    def __init__(
            self,
            model_name: str,
            base_url: Optional[str] = None,
            api_key: Optional[str] = None,
            config: GenerateConfig = GenerateConfig(),
            **model_args: Any,
    ):
        """
        Initialize the EvalModel with ms-swift backend.

        Args:
            model_name: Name of the model for identification
            base_url: Not used in this implementation (for API compatibility)
            api_key: Not used in this implementation (for API compatibility)
            config: Generation configuration with batch settings
            **model_args: Additional arguments including 'model' and 'template'
        """
        super().__init__(
            model_name=model_name,
            base_url=base_url,
            api_key=api_key,
            config=config,
        )

        # Extract model-specific arguments from kwargs
        # This pattern allows us to collect known arguments while preserving unknown ones
        def collect_model_arg(name: str) -> Optional[Any]:
            nonlocal model_args
            value = model_args.get(name, None)
            if value is not None:
                model_args.pop(name)
            return value

        # Extract required model parameters
        self.model = collect_model_arg('model')  # model path or identifier
        self.template = collect_model_arg('template')  # conversation template
        self.max_batch_size = collect_model_arg('max_batch_size')  # maximum batch size

        # Initialize the inference engine with batch support
        self.engine = PtEngine.from_model_template(self.model, self.template, max_batch_size=self.max_batch_size)

    def generate(
        self,
        input: List[EvalChatMessage],
        tools: List[ToolInfo],
        tool_choice: ToolChoice,
        config: GenerateConfig,
    ) -> ModelOutput:
        """
        Generate model response using batch inference.

        This method queues the request for batch processing and waits for the result.
        The actual inference is performed asynchronously in a background thread.

        Args:
            input: List of chat messages forming the conversation
            tools: Available tools for function calling (if supported)
            tool_choice: Tool selection strategy
            config: Generation configuration

        Returns:
            ModelOutput containing the generated response
        """
        # Ensure the background batch processing thread is running
        global batch_thread
        if batch_thread is None:
            batch_thread = Thread(target=_process_batches, daemon=True)
            batch_thread.start()

        # Convert EvalScope format to ms-swift format
        ms_input = convert_request(input, tools)
        ms_config = convert_config(config)

        # Package the request for batch processing
        batch_input = BatchInferInput(
            ms_input=ms_input, ms_config=ms_config, batch_size=config.batch_size, engine=self.engine)

        # Create a future to receive the result asynchronously
        future = Future[ModelOutput]()

        # Queue the request for batch processing
        batch_queue.put(_QueueItem(input=batch_input, future=future))

        # Block until the result is available
        return future.result()


def _process_batches() -> None:
    """
    Background thread function that processes batched inference requests.

    This function runs continuously, collecting requests from the queue and
    processing them in batches for improved efficiency. It uses a timeout-based
    approach to balance between batch size and latency.
    """
    while True:
        # Collect requests from the queue until timeout or batch size limit
        inputs: List[Tuple[BatchInferInput, Future[ModelOutput]]] = []

        while True:
            try:
                # Wait for new requests with a 2-second timeout
                item = batch_queue.get(timeout=2)
                inputs.append((item.input, item.future))

                # Check if we've reached the desired batch size
                if len(inputs) == item.input.batch_size:
                    break  # Process this batch now

            except Empty:
                # No more requests in queue, process what we have
                break

        # Skip processing if no requests were collected
        if len(inputs) == 0:
            continue

        try:
            # Prepare batch inputs for ms-swift inference
            ms_inputs = [item[0].ms_input for item in inputs]
            ms_config = inputs[0][0].ms_config  # use first config for the batch
            engine = inputs[0][0].engine  # use first engine for the batch

            # Perform batch inference using ms-swift engine
            completions = engine.infer(ms_inputs, ms_config, use_tqdm=False)

            # Process results and deliver them to waiting futures
            for i, (batch_input, future) in enumerate(inputs):
                completion = completions[i]

                # Convert ms-swift response to EvalScope format
                choices = chat_choices_from_openai(completion, tools=[])
                result = ModelOutput(
                    model=completion.model,
                    choices=choices,
                    usage=(ModelUsage(
                        input_tokens=completion.usage.prompt_tokens,
                        output_tokens=completion.usage.completion_tokens,
                        total_tokens=completion.usage.total_tokens,
                    ) if completion.usage else None),
                )

                # Deliver the result to the waiting caller
                future.set_result(result)

        except Exception as ex:
            # If batch processing fails, propagate the error to all waiting futures
            for _, future in inputs:
                future.set_exception(ex)


def convert_config(config: GenerateConfig) -> RequestConfig:
    """
    Convert EvalScope GenerateConfig to ms-swift RequestConfig.

    Maps configuration parameters between the two frameworks, ensuring
    compatibility while maintaining the same generation behavior.

    Args:
        config: EvalScope generation configuration

    Returns:
        RequestConfig: ms-swift compatible configuration
    """
    return RequestConfig(
        max_tokens=config.max_tokens,
        temperature=config.temperature,
        top_k=config.top_k,
        top_p=config.top_p,
        presence_penalty=config.presence_penalty,
        frequency_penalty=config.frequency_penalty,
        seed=config.seed,
        stream=False,  # batch processing doesn't support streaming
        logprobs=config.logprobs,
        top_logprobs=config.top_logprobs)


def convert_request(messages: List[EvalChatMessage], tools: List[ToolInfo]) -> InferRequest:
    """
    Convert EvalScope request format to ms-swift InferRequest format.

    Transforms the message and tool format from EvalScope's representation
    to the format expected by ms-swift's inference engine.

    Args:
        messages: List of chat messages in EvalScope format
        tools: List of available tools in EvalScope format

    Returns:
        InferRequest: ms-swift compatible request object
    """
    # Convert tools to ms-swift format
    tools_list = []
    if len(tools) > 0:
        tools_list = [tool.model_dump(exclude_none=True) for tool in tools]

    # Convert messages to ms-swift format
    ms_messages = []
    for message in messages:
        ms_messages.append(message.model_dump(exclude_none=True))

    return InferRequest(
        messages=ms_messages,
        tools=tools_list,
    )
