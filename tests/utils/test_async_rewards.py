"""Unit tests for async reward functions.

Tests the async reward function implementation including:
- AsyncORM class functionality
- Event loop daemon thread management
- Performance comparison between sync and async reward functions
"""
import asyncio
import time
import unittest
from typing import List


class TestAsyncRewardFunctions(unittest.TestCase):
    """Test async reward function utilities and base class."""

    def test_start_and_shutdown_event_loop_in_daemon(self):
        """Test that we can start and shutdown an event loop in a daemon thread."""
        from swift.utils import start_event_loop_in_daemon, shutdown_event_loop_in_daemon

        # Start the event loop
        thread, loop, ready_event = start_event_loop_in_daemon(name='TestLoop')

        # Wait for the loop to be ready
        self.assertTrue(ready_event.wait(timeout=5))

        # Verify the loop is running
        self.assertTrue(loop.is_running())
        self.assertTrue(thread.is_alive())

        # Shutdown the event loop
        shutdown_event_loop_in_daemon(thread, loop)

        # Give the thread time to finish
        thread.join(timeout=2)
        self.assertFalse(thread.is_alive())

    def test_run_async_function_in_daemon_loop(self):
        """Test running an async function in the daemon event loop."""
        from swift.utils import start_event_loop_in_daemon, shutdown_event_loop_in_daemon

        thread, loop, ready_event = start_event_loop_in_daemon(name='TestLoop')
        ready_event.wait(timeout=5)

        async def async_add(a, b):
            await asyncio.sleep(0.01)  # Simulate async work
            return a + b

        # Run the async function in the daemon loop
        future = asyncio.run_coroutine_threadsafe(async_add(2, 3), loop)
        result = future.result(timeout=5)

        self.assertEqual(result, 5)

        shutdown_event_loop_in_daemon(thread, loop)

    def test_async_orm_base_class(self):
        """Test that AsyncORM can be subclassed and used correctly."""
        from swift.rewards import AsyncORM

        class TestAsyncReward(AsyncORM):

            async def __call__(self, completions: List[str], **kwargs) -> List[float]:
                await asyncio.sleep(0.01)
                return [float(len(c)) for c in completions]

        reward_func = TestAsyncReward()

        # Check that it's detected as async
        self.assertTrue(asyncio.iscoroutinefunction(reward_func.__call__))

        # Run in an event loop to verify it works
        async def run_test():
            result = await reward_func(['hello', 'world!'])
            return result

        result = asyncio.get_event_loop().run_until_complete(run_test())
        self.assertEqual(result, [5.0, 6.0])

    def test_async_reward_is_detected(self):
        """Test that async reward functions are correctly detected."""
        from swift.rewards import AsyncORM

        class SyncReward:

            def __call__(self, completions, **kwargs):
                return [1.0] * len(completions)

        class AsyncReward(AsyncORM):

            async def __call__(self, completions, **kwargs):
                return [1.0] * len(completions)

        sync_func = SyncReward()
        async_func = AsyncReward()

        # Check detection
        self.assertFalse(asyncio.iscoroutinefunction(sync_func))
        self.assertFalse(asyncio.iscoroutinefunction(sync_func.__call__))
        self.assertTrue(asyncio.iscoroutinefunction(async_func.__call__))


class TestAsyncRewardPerformance(unittest.TestCase):
    """Test performance benefits of async reward functions."""

    def test_parallel_async_execution(self):
        """Test that multiple async reward functions execute in parallel."""
        from swift.utils import start_event_loop_in_daemon, shutdown_event_loop_in_daemon

        thread, loop, ready_event = start_event_loop_in_daemon(name='PerfTestLoop')
        ready_event.wait(timeout=5)

        sleep_time = 0.1  # 100ms per call
        num_calls = 5

        async def slow_async_func(idx):
            await asyncio.sleep(sleep_time)
            return idx

        # Test sequential execution time
        start_seq = time.time()
        for i in range(num_calls):
            future = asyncio.run_coroutine_threadsafe(slow_async_func(i), loop)
            future.result(timeout=5)
        sequential_time = time.time() - start_seq

        # Test parallel execution time using asyncio.gather
        async def run_parallel():
            tasks = [slow_async_func(i) for i in range(num_calls)]
            return await asyncio.gather(*tasks)

        start_par = time.time()
        future = asyncio.run_coroutine_threadsafe(run_parallel(), loop)
        results = future.result(timeout=5)
        parallel_time = time.time() - start_par

        shutdown_event_loop_in_daemon(thread, loop)

        # Verify results
        self.assertEqual(results, list(range(num_calls)))

        # Parallel should be significantly faster than sequential
        # Sequential: ~num_calls * sleep_time = 0.5s
        # Parallel: ~sleep_time = 0.1s
        # Allow some margin for overhead
        self.assertLess(parallel_time, sequential_time * 0.5)

        print('\nPerformance test results:')
        print(f'  Sequential time: {sequential_time:.3f}s (expected ~{sleep_time * num_calls:.1f}s)')
        print(f'  Parallel time:   {parallel_time:.3f}s (expected ~{sleep_time:.1f}s)')
        print(f'  Speedup:         {sequential_time / parallel_time:.1f}x')

    def test_async_reward_function_batch_performance(self):
        """Test performance of async reward function with batch processing."""
        from swift.rewards import AsyncORM
        from swift.utils import start_event_loop_in_daemon, shutdown_event_loop_in_daemon

        sleep_per_item = 0.05  # 50ms per item
        batch_size = 8

        class SlowSyncReward:

            def __call__(self, completions, **kwargs):
                rewards = []
                for c in completions:
                    time.sleep(sleep_per_item)  # Blocking sleep
                    rewards.append(float(len(c)))
                return rewards

        class FastAsyncReward(AsyncORM):

            async def __call__(self, completions, **kwargs):

                async def score_single(text):
                    await asyncio.sleep(sleep_per_item)  # Non-blocking sleep
                    return float(len(text))

                # Process all in parallel
                tasks = [score_single(c) for c in completions]
                return await asyncio.gather(*tasks)

        completions = [f'text_{i}' for i in range(batch_size)]

        # Test sync reward function
        sync_reward = SlowSyncReward()
        start_sync = time.time()
        sync_results = sync_reward(completions)
        sync_time = time.time() - start_sync

        # Test async reward function
        thread, loop, ready_event = start_event_loop_in_daemon(name='BatchPerfLoop')
        ready_event.wait(timeout=5)

        async_reward = FastAsyncReward()

        async def run_async():
            return await async_reward(completions)

        start_async = time.time()
        future = asyncio.run_coroutine_threadsafe(run_async(), loop)
        async_results = future.result(timeout=10)
        async_time = time.time() - start_async

        shutdown_event_loop_in_daemon(thread, loop)

        # Verify results are the same
        self.assertEqual(len(sync_results), len(async_results))
        self.assertEqual(sync_results, list(async_results))

        # Async should be significantly faster
        # Sync: ~batch_size * sleep_per_item = 0.4s
        # Async: ~sleep_per_item = 0.05s
        self.assertLess(async_time, sync_time * 0.5)

        print('\nBatch processing performance:')
        print(f'  Sync time:  {sync_time:.3f}s (expected ~{sleep_per_item * batch_size:.2f}s)')
        print(f'  Async time: {async_time:.3f}s (expected ~{sleep_per_item:.2f}s)')
        print(f'  Speedup:    {sync_time / async_time:.1f}x')


if __name__ == '__main__':
    unittest.main()
