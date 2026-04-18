"""Benchmark suite for TrajectoryCollector performance.

Run: python -m benchmarks.bench_collector
"""

import tempfile
import timeit
import sys

from sentric import TrajectoryCollector
from sentric._json import _has_orjson


def bench_add_message(n=1000):
    """Measure add_message throughput."""
    with tempfile.TemporaryDirectory() as tmpdir:
        collector = TrajectoryCollector(
            task_id="bench", domain="code",
            model={"name": "test", "version": "1", "provider": "local"},
            output_dir=tmpdir,
        )

        def run():
            for i in range(n):
                collector.add_message(role="user", content=f"Message {i}")

        elapsed = timeit.timeit(run, number=1)
        rate = n / elapsed
        per_msg_us = (elapsed / n) * 1_000_000
        print(f"add_message:  {rate:,.0f} msgs/sec  ({per_msg_us:.1f} μs/msg, n={n})")


def bench_save_episode(n_messages=100, iterations=50):
    """Measure save_episode latency."""
    times = []
    for _ in range(iterations):
        with tempfile.TemporaryDirectory() as tmpdir:
            collector = TrajectoryCollector(
                task_id="bench", domain="code",
                model={"name": "test", "version": "1", "provider": "local"},
                output_dir=tmpdir,
            )
            for i in range(n_messages):
                collector.add_message(role="user", content=f"Message {i}")
            collector.add_tokens(input_tokens=500, output_tokens=300)

            t = timeit.timeit(lambda: collector.save_episode(tmpdir), number=1)
            times.append(t)

    avg_ms = (sum(times) / len(times)) * 1000
    min_ms = min(times) * 1000
    max_ms = max(times) * 1000
    print(f"save_episode: avg={avg_ms:.2f}ms  min={min_ms:.2f}ms  max={max_ms:.2f}ms  ({n_messages} msgs, {iterations} iters)")


def bench_save_episode_async(n_messages=100, iterations=50):
    """Measure save_episode_async latency (submit time only)."""
    times = []
    futures = []
    with tempfile.TemporaryDirectory() as tmpdir:
        for _ in range(iterations):
            collector = TrajectoryCollector(
                task_id="bench", domain="code",
                model={"name": "test", "version": "1", "provider": "local"},
                output_dir=tmpdir,
            )
            for i in range(n_messages):
                collector.add_message(role="user", content=f"Message {i}")
            collector.add_tokens(input_tokens=500, output_tokens=300)

            t = timeit.timeit(lambda: futures.append(collector.save_episode_async(tmpdir)), number=1)
            times.append(t)

        # Wait for all futures
        for f in futures:
            f.result()

    avg_us = (sum(times) / len(times)) * 1_000_000
    print(f"save_async:   avg={avg_us:.1f}μs submit time  ({n_messages} msgs, {iterations} iters)")


def bench_memory(n_messages=1000):
    """Estimate memory per message."""
    import tracemalloc
    tracemalloc.start()

    with tempfile.TemporaryDirectory() as tmpdir:
        collector = TrajectoryCollector(
            task_id="bench", domain="code",
            model={"name": "test", "version": "1", "provider": "local"},
            output_dir=tmpdir,
        )
        snap1 = tracemalloc.take_snapshot()

        for i in range(n_messages):
            collector.add_message(role="user", content=f"Message {i} with some content to simulate real usage")

        snap2 = tracemalloc.take_snapshot()

    stats = snap2.compare_to(snap1, "lineno")
    total_bytes = sum(s.size_diff for s in stats if s.size_diff > 0)
    per_msg = total_bytes / n_messages
    print(f"memory:       {per_msg:.0f} bytes/msg  (total={total_bytes:,} bytes for {n_messages} msgs)")

    tracemalloc.stop()


if __name__ == "__main__":
    print(f"Python {sys.version}")
    print(f"orjson available: {_has_orjson()}")
    print("-" * 60)
    bench_add_message()
    bench_save_episode()
    bench_save_episode_async()
    bench_memory()
