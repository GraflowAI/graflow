"""
Channel Concurrency Example
============================

This example demonstrates how to safely share and update channel data
when tasks run in parallel using ``ParallelGroup``.

Problem
-------
A naive read-modify-write (``get`` -> compute -> ``set``) is **not** atomic.
When multiple tasks execute in parallel threads, updates can be lost because
two tasks may read the same value and overwrite each other's writes.

Solutions
---------
1. **``channel.atomic_add(key, amount)``** — Atomic numeric add (inc/dec).
   Backed by a per-key lock in MemoryChannel and ``INCRBYFLOAT`` in Redis.

2. **``channel.lock(key)``** — Advisory lock for arbitrary compound
   operations. ``MemoryChannel`` uses a per-key ``threading.RLock`` for
   in-process coordination; ``RedisChannel`` uses ``redis.lock.Lock``
   (SET NX + Lua release) for cross-client distributed coordination.

Expected Output
---------------
=== Channel Concurrency Demo ===

--- Unsafe parallel increment (race condition) ---
Expected counter: 500, Actual: <less than 500>
Updates lost!

--- Safe parallel increment with atomic_add() ---
Expected counter: 500, Actual: 500

--- Safe compound update with lock() ---
Overflow events: 5
Counter after resets: 0

Done!
"""

from graflow.core.context import TaskExecutionContext
from graflow.core.decorators import task
from graflow.core.task import ParallelGroup
from graflow.core.workflow import workflow


def demo_unsafe_increment() -> None:
    """Show that naive get/set loses updates in parallel execution."""
    import time

    print("--- Unsafe parallel increment (race condition) ---")

    num_workers = 5
    increments_per_worker = 100
    expected = num_workers * increments_per_worker

    with workflow("unsafe_demo") as ctx:

        @task(inject_context=True)
        def init_counter(context: TaskExecutionContext):
            context.get_channel().set("counter", 0)

        # Create worker tasks that use naive get/set
        workers = []
        for i in range(num_workers):

            @task(inject_context=True, id=f"unsafe_worker_{i}")
            def unsafe_worker(context: TaskExecutionContext):
                channel = context.get_channel()
                for _ in range(increments_per_worker):
                    val = channel.get("counter")
                    time.sleep(0)  # yield to trigger interleaving
                    channel.set("counter", val + 1)

            workers.append(unsafe_worker)

        @task(inject_context=True)
        def report(context: TaskExecutionContext):
            actual = context.get_channel().get("counter")
            print(f"  Expected counter: {expected}, Actual: {actual}")
            if actual < expected:
                print("  Updates lost!\n")
            else:
                print("  (Got lucky — no interleaving this run)\n")

        parallel = ParallelGroup(workers, name="unsafe_group")
        _ = init_counter >> parallel >> report
        ctx.execute("init_counter")


def demo_atomic_add() -> None:
    """Show that atomic_add() is safe for parallel numeric updates."""
    print("--- Safe parallel increment with atomic_add() ---")

    num_workers = 5
    increments_per_worker = 100
    expected = num_workers * increments_per_worker

    with workflow("add_demo") as ctx:

        @task(inject_context=True)
        def init_counter(context: TaskExecutionContext):
            context.get_channel().set("counter", 0)

        workers = []
        for i in range(num_workers):

            @task(inject_context=True, id=f"add_worker_{i}")
            def add_worker(context: TaskExecutionContext):
                channel = context.get_channel()
                for _ in range(increments_per_worker):
                    channel.atomic_add("counter", 1)

            workers.append(add_worker)

        @task(inject_context=True)
        def report(context: TaskExecutionContext):
            actual = context.get_channel().get("counter")
            print(f"  Expected counter: {expected}, Actual: {actual}\n")

        parallel = ParallelGroup(workers, name="add_group")
        _ = init_counter >> parallel >> report
        ctx.execute("init_counter")


def demo_advisory_lock() -> None:
    """Show lock() for compound read-modify-write that atomic_add() can't express."""
    print("--- Safe compound update with lock() ---")

    threshold = 10
    num_workers = 5
    increments_per_worker = 10

    with workflow("lock_demo") as ctx:

        @task(inject_context=True)
        def init(context: TaskExecutionContext):
            channel = context.get_channel()
            channel.set("counter", 0)
            channel.set("overflow_count", 0)

        workers = []
        for i in range(num_workers):

            @task(inject_context=True, id=f"lock_worker_{i}")
            def lock_worker(context: TaskExecutionContext):
                channel = context.get_channel()
                for _ in range(increments_per_worker):
                    # Advisory lock protects the entire read-modify-write block
                    with channel.lock("counter"):
                        val = channel.get("counter")
                        if val >= threshold:
                            channel.set("counter", 0)
                            channel.atomic_add("overflow_count", 1)
                        else:
                            channel.set("counter", val + 1)

            workers.append(lock_worker)

        @task(inject_context=True)
        def report(context: TaskExecutionContext):
            channel = context.get_channel()
            overflows = channel.get("overflow_count")
            counter = channel.get("counter")
            print(f"  Overflow events: {overflows}")
            print(f"  Counter after resets: {counter}\n")

        parallel = ParallelGroup(workers, name="lock_group")
        _ = init >> parallel >> report
        ctx.execute("init")


def main():
    print("=== Channel Concurrency Demo ===\n")
    demo_unsafe_increment()
    demo_atomic_add()
    demo_advisory_lock()
    print("Done!")


if __name__ == "__main__":
    main()


# ============================================================================
# Key Takeaways:
# ============================================================================
#
# 1. **channel.atomic_add(key, amount)**
#    - Atomic numeric add/subtract — no lost updates
#    - Initialises missing keys to 0 automatically
#    - MemoryChannel: per-key RLock; Redis: INCRBYFLOAT (server-side atomic)
#    - Use for counters, metrics, scores
#
# 2. **channel.lock(key)**
#    - Advisory lock for compound operations that atomic_add() can't express
#    - Wrap with ``with channel.lock(key):`` context manager
#    - MemoryChannel: per-key RLock; Redis: distributed lock for the same key
#    - Use for conditional updates and other compound read-modify-write logic
#
# 3. **When to use which**
#    - Simple counter?         → channel.atomic_add("counter", 1)
#    - Decrement?              → channel.atomic_add("counter", -1)
#    - Conditional update?     → with channel.lock("key"): ...
#    - Multi-key update?       → with channel.lock("key"): ...
#    - No concurrency concern? → channel.get() / channel.set() is fine
#
# ============================================================================
