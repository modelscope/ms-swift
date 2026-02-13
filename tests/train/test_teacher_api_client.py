"""
Test script for TeacherAPIClient with vLLM backend.

This script tests the TeacherAPIClient's ability to fetch logprobs from:
1. swift deploy with vLLM backend
2. Standalone vLLM server (vllm serve)

Usage:
    python test_teacher_api_client.py  # Run all tests
    python test_teacher_api_client.py --parse-only  # Only test format parsing
"""
import argparse
import os
import time
import multiprocessing

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')


def wait_for_server(base_url: str, timeout: int = 120) -> bool:
    """Wait for server to be ready."""
    import requests
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            for endpoint in ['/health', '/v1/models']:
                resp = requests.get(f'{base_url}{endpoint}', timeout=5)
                if resp.status_code == 200:
                    print(f'Server is ready at {base_url}')
                    return True
        except Exception:
            pass
        time.sleep(2)
    print(f'Timeout waiting for server at {base_url}')
    return False


def test_api_client_logprobs(base_url: str):
    """Test TeacherAPIClient logprobs fetching."""
    from swift.rlhf_trainers import TeacherAPIClient
    from transformers import AutoTokenizer

    print(f'\n{"=" * 60}')
    print(f'Testing TeacherAPIClient')
    print(f'Base URL: {base_url}')
    print('=' * 60)

    # Initialize client
    client = TeacherAPIClient(
        base_url=base_url,
        top_logprobs=10,
        timeout=60.0,
    )

    # Check server health
    is_healthy = client.check_server_health()
    print(f'Server health check: {"OK" if is_healthy else "FAILED"}')
    if not is_healthy:
        print('Skipping test due to server health check failure')
        return False

    # Prepare test input
    tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2-0.5B-Instruct', trust_remote_code=True)
    test_text = 'Hello, how are you today?'
    input_ids = tokenizer.encode(test_text, add_special_tokens=True)

    print(f'\nTest text: "{test_text}"')
    print(f'Token IDs: {input_ids}')
    print(f'Number of tokens: {len(input_ids)}')

    # Test synchronous API
    print('\n--- Testing synchronous get_logprobs_sync ---')
    try:
        logprobs_tensor, indices_tensor = client.get_logprobs_sync(
            input_ids=[input_ids], top_logprobs=5)

        print(f'Logprobs tensor shape: {logprobs_tensor.shape}')
        print(f'Indices tensor shape: {indices_tensor.shape}')

        # Check for valid logprobs
        valid_count = (logprobs_tensor > float('-inf')).sum().item()
        print(f'Valid logprob entries: {valid_count}')

        if valid_count > 0:
            print('\nSample logprobs for first position:')
            for k in range(min(5, indices_tensor.shape[-1])):
                token_id = indices_tensor[0, 0, k].item()
                logprob = logprobs_tensor[0, 0, k].item()
                if token_id > 0 and logprob > float('-inf'):
                    token_str = tokenizer.decode([token_id])
                    print(f'  Top-{k + 1}: token_id={token_id} ("{token_str}"), logprob={logprob:.4f}')
            print('\nSync test: PASSED')
            return True
        else:
            print('\nSync test: FAILED (no valid logprobs)')
            return False

    except Exception as e:
        print(f'Sync test: FAILED with error: {e}')
        import traceback
        traceback.print_exc()
        return False


def test_with_swift_deploy_vllm(port: int = 8100):
    """Test with swift deploy using vLLM backend."""
    from swift import DeployArguments, deploy_main

    print('\n' + '=' * 60)
    print('Starting swift deploy with vLLM backend...')
    print('=' * 60)

    mp = multiprocessing.get_context('spawn')
    args = DeployArguments(
        model='Qwen/Qwen2-0.5B-Instruct',
        infer_backend='vllm',
        port=port,
        verbose=False,
        vllm_max_model_len=4096,
    )

    process = mp.Process(target=deploy_main, args=(args, ))
    process.start()

    try:
        base_url = f'http://localhost:{port}'
        if wait_for_server(base_url):
            result = test_api_client_logprobs(base_url)
            return result
        return False
    finally:
        process.terminate()
        process.join(timeout=10)
        if process.is_alive():
            process.kill()


def test_logprobs_format_parsing():
    """Test parsing of vLLM logprobs response format."""
    print('\n' + '=' * 60)
    print('Testing logprobs format parsing')
    print('=' * 60)

    from swift.rlhf_trainers import TeacherAPIClient

    client = TeacherAPIClient(base_url='http://localhost:8000', top_logprobs=5)

    # Test vLLM response parsing with token_id keys
    vllm_response = {
        'choices': [{
            'logprobs': {
                'top_logprobs': [
                    {
                        '123': -0.5,
                        '456': -1.2,
                        '789': -2.0
                    },
                    {
                        '44': -0.1,
                        '55': -2.5,
                        '66': -3.0
                    },
                ]
            }
        }]
    }

    result = client._parse_response(vllm_response, seq_len=2, topk=3)
    print(f'Parsing result indices: {result["indices"]}')
    print(f'Parsing result values: {result["values"]}')
    assert len(result['values']) == 2, 'Expected 2 positions'
    assert len(result['values'][0]) == 3, 'Expected 3 top logprobs per position'
    assert result['indices'][0][0] == 123, f'Expected token ID 123, got {result["indices"][0][0]}'
    print('Format parsing: PASSED')

    return True


def main():
    parser = argparse.ArgumentParser(description='Test TeacherAPIClient')
    parser.add_argument('--parse-only', action='store_true', help='Only test format parsing (no server needed)')
    args = parser.parse_args()

    results = {}

    # Test format parsing (no server needed)
    print('\n' + '#' * 60)
    print('# Testing format parsing')
    print('#' * 60)
    try:
        results['format_parsing'] = test_logprobs_format_parsing()
    except Exception as e:
        print(f'Format parsing test failed: {e}')
        import traceback
        traceback.print_exc()
        results['format_parsing'] = False

    if args.parse_only:
        print('\n' + '=' * 60)
        print('Test Summary (parse-only mode):')
        print('=' * 60)
        for test, passed in results.items():
            print(f'  {test}: {"PASSED" if passed else "FAILED"}')
        return

    # Test with swift deploy
    print('\n' + '#' * 60)
    print('# Testing with vLLM backend')
    print('#' * 60)
    try:
        results['vllm'] = test_with_swift_deploy_vllm()
    except Exception as e:
        print(f'vLLM test failed: {e}')
        import traceback
        traceback.print_exc()
        results['vllm'] = False

    # Print summary
    print('\n' + '=' * 60)
    print('Test Summary:')
    print('=' * 60)
    for test, passed in results.items():
        print(f'  {test}: {"PASSED" if passed else "FAILED"}')

    all_passed = all(results.values())
    print(f'\nOverall: {"ALL TESTS PASSED" if all_passed else "SOME TESTS FAILED"}')
    return all_passed


if __name__ == '__main__':
    main()
