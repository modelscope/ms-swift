import os


def test_client():
    from swift.llm import sampling_main, SamplingArguments
    import json
    base_url = 'https://dashscope.aliyuncs.com/compatible-mode/v1'
    api_key = os.environ.get('OPENAI_API_KEY')
    engine_kwargs = json.dumps({
        'base_url': base_url,
        'api_key': api_key,
    })
    dataset = 'tastelikefeet/competition_math#5'
    system = """A conversation between User and Assistant. The user asks a question, and the Assistant solves it.
        The assistant first thinks about the reasoning process in the mind and then provides the user
        with the answer. The reasoning process and answer are enclosed
        within <think> </think> and <answer> </answer> tags, respectively,
        i.e., <think> reasoning process here </think> <answer> answer here </answer>."""
    args = SamplingArguments(
        sampler_type='distill',
        sampler_engine='client',
        model='deepseek-r1',
        dataset=dataset,
        num_return_sequences=1,
        stream=True,
        system=system,
        temperature=0.6,
        top_p=0.95,
        engine_kwargs=engine_kwargs,
    )
    sampling_main(args)


if __name__ == '__main__':
    test_client()
