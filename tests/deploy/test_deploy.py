from types import SimpleNamespace

from swift.infer_engine.protocol import ChatCompletionResponseStreamChoice, ChatCompletionStreamResponse, DeltaMessage
from swift.pipelines.infer.deploy import SwiftDeploy


def test_post_process_reasoning_only_stream_chunk():
    deploy = object.__new__(SwiftDeploy)
    deploy.args = SimpleNamespace()
    request_info = {'response': ''}
    response = ChatCompletionStreamResponse(
        model='test',
        choices=[
            ChatCompletionResponseStreamChoice(
                index=0,
                delta=DeltaMessage(content=None, reasoning_content='thinking'),
                finish_reason=None,
            )
        ],
    )

    processed = deploy._post_process(request_info, response)

    assert processed is response
    assert processed.choices[0].delta.content is None
    assert processed.choices[0].delta.reasoning_content == 'thinking'
    assert request_info['response'] == ''
