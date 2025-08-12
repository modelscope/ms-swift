"""
Example of using reasoning_parser

This example demonstrates how to use reasoning_parser in Swift's VllmEngine to support reasoning models.
"""

from swift.llm import InferRequest, RequestConfig, VllmEngine


def main(engine: VllmEngine):
    # Create inference request
    infer_request = InferRequest(messages=[{'role': 'user', 'content': '9.11 and 9.8, which is greater?'}])

    # Configure request parameters
    request_config = RequestConfig(
        max_tokens=8192,
        temperature=0.7,
        stream=False  # Non-streaming inference
    )

    # Execute inference
    responses = engine.infer(infer_requests=[infer_request], request_config=request_config)

    # Process responses
    for response in responses:
        if hasattr(response, 'choices') and response.choices:
            choice = response.choices[0]
            message = choice.message

            print('=== Reasoning Content ===')
            if message.reasoning_content:
                print(f'Reasoning steps: {message.reasoning_content}')
            else:
                print('No reasoning content detected')

            print('\n=== Final Answer ===')
            print(f'Answer: {message.content}')

            print('\n=== Finish Reason ===')
            print(f'Reason: {choice.finish_reason}')


def streaming_example(engine: VllmEngine):
    """Streaming inference example"""
    infer_request = InferRequest(messages=[{'role': 'user', 'content': 'Calculate the result of 15 + 27'}])

    request_config = RequestConfig(
        max_tokens=8192,
        temperature=0.7,
        stream=True  # Enable streaming inference
    )

    # Streaming inference
    responses = engine.infer(infer_requests=[infer_request], request_config=request_config)

    print('=== Streaming Inference Results ===')
    for chunk in responses[0]:  # responses[0] is the streaming generator
        if chunk and chunk.choices:
            choice = chunk.choices[0]
            delta = choice.delta

            if delta.reasoning_content:
                print(f'Reasoning: {delta.reasoning_content}', end='', flush=True)

            if delta.content:
                print(f'Content: {delta.content}', end='', flush=True)

    print('\n=== Inference Complete ===')


if __name__ == '__main__':
    # Initialize VllmEngine with reasoning_parser enabled
    engine = VllmEngine(
        model_id_or_path='Qwen/Qwen3-8B',
        reasoning_parser='qwen3',  # Specify reasoning parser
        gpu_memory_utilization=0.9,
    )

    print('=== Non-streaming Inference Example ===')
    main(engine)

    print('\n' + '=' * 50 + '\n')

    print('=== Streaming Inference Example ===')
    streaming_example(engine)
