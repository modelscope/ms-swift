from swift.llm import SamplingArguments, sampling_main


def test_sampling():
    sampling_main(
        SamplingArguments(
            model='LLM-Research/Meta-Llama-3.1-8B-Instruct',
            sampler_engine='pt',
            num_return_sequences=5,
            dataset='AI-ModelScope/alpaca-gpt4-data-zh#5'))


if __name__ == '__main__':
    test_sampling()
