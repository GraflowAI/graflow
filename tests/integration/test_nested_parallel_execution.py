"""Integration tests for nested ParallelGroup execution with Redis coordination.

Tests Phase 2 scenarios from redis_distributed_execution_redesign.md:
- Worker-local task execution via next_task()
- Nested ParallelGroup execution within Worker
- 2-3 levels of nested ParallelGroup
- Dynamic task addition that contains nested ParallelGroups
"""

import time

import pytest

from graflow.channels.base import Channel
from graflow.coordination.coordinator import CoordinationBackend
from graflow.core.context import TaskExecutionContext
from graflow.core.decorators import task
from graflow.core.workflow import workflow
from graflow.queue.distributed import DistributedTaskQueue
from graflow.utils.redis_utils import create_redis_client, extract_redis_config


@pytest.mark.integration
class TestNestedParallelExecution:
    """Tests for nested ParallelGroup execution scenarios."""

    def test_worker_next_task_local_execution(self, clean_redis):
        """
        Test: Worker-local task execution via next_task() works correctly.

        Scenario:
        1. Dispatch tasks via ParallelGroup
        2. Worker calls next_task() to add local tasks
        3. Added tasks execute within the Worker
        """
        from graflow.channels.factory import ChannelFactory

        redis_config = extract_redis_config(clean_redis)
        redis_config.setdefault("decode_responses", True)

        # Use Redis channel to track execution across serialization boundaries
        results_channel = ChannelFactory.create_channel(
            backend="redis",
            name="test_results",
            key_prefix="test_local",
            **redis_config,
        )
        results_channel.set("results", [])

        with workflow("worker_next_task_local_execution") as wf:
            @task("parallel_task_1", inject_context=True)
            def parallel_task_1(ctx: TaskExecutionContext):
                """Task executed in ParallelGroup that adds local task via next_task()"""
                # Add local task via next_task() within Worker
                @task("local_task_1")
                def local_task_1():
                    return "local_result_1"

                ctx.next_task(local_task_1)
                return "parallel_result_1"

            @task("parallel_task_2", inject_context=True)
            def parallel_task_2(ctx: TaskExecutionContext):
                """Task executed in ParallelGroup that adds local task via next_task()"""
                # Add local task via next_task() within Worker
                @task("local_task_2")
                def local_task_2():
                    return "local_result_2"

                ctx.next_task(local_task_2)
                return "parallel_result_2"

            # Setup Worker
            queue = DistributedTaskQueue(
                redis_client=create_redis_client(redis_config),
                key_prefix="test_local"
            )
            from graflow.worker.worker import TaskWorker
            worker = TaskWorker(queue, worker_id="test-worker-1", max_concurrent_tasks=2)
            worker.start()

            try:
                # Create ParallelGroup and execute via workflow
                parallel_group = (
                    parallel_task_1 | parallel_task_2
                ).set_group_name("parallel_local").with_execution(
                    CoordinationBackend.REDIS,
                    backend_config={**redis_config, "key_prefix": "test_local"},
                )
                wf.graph.add_node(parallel_group, parallel_group.task_id)

                wf.execute(start_node=parallel_group.task_id)

                # Wait for worker to process tasks
                time.sleep(3)

                # Verify execution by checking if tasks stored results in Redis
                # Since results are stored in ExecutionContext which is session-specific,
                # we verify by checking task completion count
                assert worker.tasks_processed == 2, f"Expected 2 tasks processed, got {worker.tasks_processed}"
                assert worker.tasks_succeeded == 2, f"Expected 2 tasks succeeded, got {worker.tasks_succeeded}"
                assert worker.tasks_failed == 0, f"Expected 0 tasks failed, got {worker.tasks_failed}"

            finally:
                worker.stop(timeout=5)
                results_channel.delete("results")

    def test_nested_parallel_group_level_1(self, clean_redis):
        """
        Test: Nested ParallelGroup execution within Worker dispatches
              child tasks with new graph_hash.

        Scenario:
        1. Outer ParallelGroup
        2. Worker creates new ParallelGroup
        3. Inner ParallelGroup is dispatched with new graph_hash
        """

        with workflow("nested_parallel_level_1") as wf:
            redis_config = extract_redis_config(clean_redis)
            redis_config.setdefault("decode_responses", True)

            def _append_result(channel: Channel, value: str):
                """Helper to append to results list in Redis."""
                current = channel.get("results") or []
                current.append(value)
                channel.set("results", current)

            @task("inner_task_o1_i1", inject_context=True)
            def inner_task_o1_i1(ctx: TaskExecutionContext):
                """Task in inner ParallelGroup"""
                channel = ctx.get_channel()
                _append_result(channel, "inner_o1_i1")
                return "inner_result_o1_i1"

            @task("inner_task_o1_i2")
            def inner_task_o1_i2(ctx: TaskExecutionContext):
                """Task in inner ParallelGroup"""
                channel = ctx.get_channel()
                _append_result(channel, "inner_o1_i2")
                return "inner_result_o1_i2"

            @task("inner_task_o2_i1", inject_context=True)
            def inner_task_o2_i1(ctx: TaskExecutionContext):
                """Task in inner ParallelGroup"""
                channel = ctx.get_channel()
                _append_result(channel, "inner_o2_i1")
                return "inner_result_o2_i1"

            @task("inner_task_o2_i2", inject_context=True)
            def inner_task_o2_i2(ctx: TaskExecutionContext):
                """Task in inner ParallelGroup"""
                channel = ctx.get_channel()
                _append_result(channel, "inner_o2_i2")
                return "inner_result_o2_i2"

            @task("outer_o1_start", inject_context=True)
            def outer_o1_start(ctx: TaskExecutionContext):
                channel = ctx.get_channel()
                _append_result(channel, "outer_o1_start")

            @task("outer_o1_end", inject_context=True)
            def outer_o1_end(ctx: TaskExecutionContext):
                channel = ctx.get_channel()
                _append_result(channel, "outer_o1_end")

            @task("outer_o2_start", inject_context=True)
            def outer_o2_start(ctx: TaskExecutionContext):
                channel = ctx.get_channel()
                _append_result(channel, "outer_o2_start")

            @task("outer_o2_end", inject_context=True)
            def outer_o2_end(ctx: TaskExecutionContext):
                channel = ctx.get_channel()
                _append_result(channel, "outer_o2_end")

            nested_group_o1 = (
                inner_task_o1_i1 | inner_task_o1_i2
            ).set_group_name("nested_o1").with_execution(
                CoordinationBackend.REDIS,
                backend_config={**redis_config, "key_prefix": "test_nested"},
            )

            nested_group_o2 = (
                inner_task_o2_i1 | inner_task_o2_i2
            ).set_group_name("nested_o2").with_execution(
                CoordinationBackend.REDIS,
                backend_config={**redis_config, "key_prefix": "test_nested"},
            )

            outer_task_o1 = (
                outer_o1_start >> nested_group_o1 >> outer_o1_end
            )

            outer_task_o2 = (
                outer_o2_start >> nested_group_o2 >> outer_o2_end
            )

            # Setup Workers (multiple for parallel processing)
            queue = DistributedTaskQueue(
                redis_client=create_redis_client(redis_config),
                key_prefix="test_nested"
            )
            from graflow.worker.worker import TaskWorker
            workers = [
                TaskWorker(queue, worker_id=f"test-worker-{i}", max_concurrent_tasks=2)
                for i in range(3)
            ]

            for w in workers:
                w.start()

            try:
                # Execute workflow
                outer_group = (
                    outer_task_o1 | outer_task_o2
                ).set_group_name("outer_group").with_execution(
                    CoordinationBackend.REDIS,
                    backend_config={**redis_config, "key_prefix": "test_nested"},
                )

                _result, exec_context = wf.execute(start_node=outer_group.task_id, ret_context=True)

                # Wait for all workers to process
                time.sleep(10)

                # Verify all tasks completed by checking worker metrics
                total_processed = sum(w.tasks_processed for w in workers)
                total_succeeded = sum(w.tasks_succeeded for w in workers)
                total_failed = sum(w.tasks_failed for w in workers)

                # Expected: 2 outer start + 2 outer end + 4 inner tasks + 2 nested groups = 10 tasks
                assert total_processed == 10, f"Expected 10 tasks processed, got {total_processed}"
                assert total_succeeded == 10, f"Expected 10 tasks succeeded, got {total_succeeded}"
                assert total_failed == 0, f"Expected 0 tasks failed, got {total_failed}"

                exec_context.get_channel().delete("results")

            finally:
                for w in workers:
                    w.stop(timeout=5)

    def test_nested_parallel_group_level_2_3(self, clean_redis):
        """
        Test: 2-3 levels of nested ParallelGroup work correctly.

        Scenario:
        Level 0 (Producer) → Level 1 (Worker) → Level 2 (Worker) → Level 3 (Worker)
        """
        with workflow("nested_parallel_level_2_3") as wf:
            redis_config = extract_redis_config(clean_redis)
            redis_config.setdefault("decode_responses", True)

            def _append_result(channel: Channel, value: str):
                """Helper to append to results list in Redis."""
                current = channel.get("results") or []
                current.append(value)
                channel.set("results", current)

            # Level 3 tasks
            @task("level3_L1_1_L2_1_L3_1", inject_context=True)
            def level3_L1_1_L2_1_L3_1(ctx: TaskExecutionContext):
                """Level 3: Deepest task"""
                channel = ctx.get_channel()
                _append_result(channel, "L3_L1_1_L2_1_L3_1")
                return "L3_result_L1_1_L2_1_L3_1"

            @task("level3_L1_1_L2_1_L3_2", inject_context=True)
            def level3_L1_1_L2_1_L3_2(ctx: TaskExecutionContext):
                """Level 3: Deepest task"""
                channel = ctx.get_channel()
                _append_result(channel, "L3_L1_1_L2_1_L3_2")
                return "L3_result_L1_1_L2_1_L3_2"

            # Level 2 task
            @task("level2_L1_1_L2_1", inject_context=True)
            def level2_L1_1_L2_1(ctx: TaskExecutionContext):
                """Level 2: Creates further nested ParallelGroup"""
                channel = ctx.get_channel()
                _append_result(channel, "L2_L1_1_L2_1_start")

                # Create Level 3 tasks
                l3_group = (
                    level3_L1_1_L2_1_L3_1 | level3_L1_1_L2_1_L3_2
                ).set_group_name("L3_group_L1_1_L2_1").with_execution(
                    CoordinationBackend.REDIS,
                    backend_config={**redis_config, "key_prefix": "test_deep"},
                )

                # Add Level 3 group to graph and execute
                ctx.next_task(l3_group)

                _append_result(channel, "L2_L1_1_L2_1_end")
                return "L2_result_L1_1_L2_1"

            @task("level2_L1_1_L2_2", inject_context=True)
            def level2_L1_1_L2_2(ctx: TaskExecutionContext):
                """Level 2: Creates further nested ParallelGroup"""
                channel = ctx.get_channel()
                _append_result(channel, "L2_L1_1_L2_2_start")
                _append_result(channel, "L2_L1_1_L2_2_end")
                return "L2_result_L1_1_L2_2"

            # Level 1 task
            @task("level1_L1_1", inject_context=True)
            def level1_L1_1(ctx: TaskExecutionContext):
                """Level 1: Creates nested ParallelGroup"""
                channel = ctx.get_channel()
                _append_result(channel, "L1_L1_1_start")

                # Create Level 2 tasks
                l2_group = (
                    level2_L1_1_L2_1 | level2_L1_1_L2_2
                ).set_group_name("L2_group_L1_1").with_execution(
                    CoordinationBackend.REDIS,
                    backend_config={**redis_config, "key_prefix": "test_deep"},
                )

                ctx.next_task(l2_group)

                _append_result(channel, "L1_L1_1_end")
                return "L1_result_L1_1"

            @task("level1_L1_2", inject_context=True)
            def level1_L1_2(ctx: TaskExecutionContext):
                """Level 1: Creates nested ParallelGroup"""
                channel = ctx.get_channel()
                _append_result(channel, "L1_L1_2_start")
                _append_result(channel, "L1_L1_2_end")
                return "L1_result_L1_2"

            # Setup Workers (multiple for deep parallelism)
            queue = DistributedTaskQueue(
                redis_client=create_redis_client(redis_config),
                key_prefix="test_deep"
            )
            from graflow.worker.worker import TaskWorker
            workers = [
                TaskWorker(queue, worker_id=f"test-worker-{i}", max_concurrent_tasks=4)
                for i in range(4)
            ]

            for w in workers:
                w.start()

            try:
                # Execute workflow with 2-3 levels of nesting
                l1_group = (
                    level1_L1_1 | level1_L1_2
                ).set_group_name("L1_group").with_execution(
                    CoordinationBackend.REDIS,
                    backend_config={**redis_config, "key_prefix": "test_deep"},
                )

                _last_result, exec_context = wf.execute(start_node=l1_group.task_id, ret_context=True)

                # Wait for deep nested execution
                time.sleep(10)

                # Retrieve results from channel
                channel = exec_context.get_channel()
                results = channel.get("results") or []

                # Verify all levels executed
                # Level 1
                assert "L1_L1_1_start" in results
                assert "L1_L1_1_end" in results
                assert "L1_L1_2_start" in results
                assert "L1_L1_2_end" in results

                # Level 2 (from L1_1)
                assert "L2_L1_1_L2_1_start" in results
                assert "L2_L1_1_L2_1_end" in results
                assert "L2_L1_1_L2_2_start" in results
                assert "L2_L1_1_L2_2_end" in results

                # Level 3 (at least some)
                l3_results = [r for r in results if r.startswith("L3_")]
                assert len(l3_results) >= 2  # At least 2 L3 tasks should execute

                # Cleanup
                channel.delete("results")

            finally:
                for w in workers:
                    w.stop(timeout=5)

    def test_dynamic_task_with_nested_parallel_group(self, clean_redis):
        """
        Test: Dynamically added task containing nested ParallelGroup
              executes correctly.

        Scenario:
        1. Task executing in ParallelGroup
        2. Dynamically add task via next_task()
        3. Added task contains ParallelGroup
        """
        with workflow("dynamic_nested_parallel") as wf:
            redis_config = extract_redis_config(clean_redis)
            redis_config.setdefault("decode_responses", True)

            def _append_result(channel: Channel, value: str):
                """Helper to append to results list in Redis."""
                current = channel.get("results") or []
                current.append(value)
                channel.set("results", current)

            @task("dynamic_nested_inner_from_t1_d1", inject_context=True)
            def dynamic_nested_inner_from_t1_d1(ctx: TaskExecutionContext):
                """Inner ParallelGroup of dynamically added task"""
                channel = ctx.get_channel()
                _append_result(channel, "dynamic_inner_from_t1_d1")
                return "dynamic_inner_result_from_t1_d1"

            @task("dynamic_nested_inner_from_t1_d2", inject_context=True)
            def dynamic_nested_inner_from_t1_d2(ctx: TaskExecutionContext):
                """Inner ParallelGroup of dynamically added task"""
                channel = ctx.get_channel()
                _append_result(channel, "dynamic_inner_from_t1_d2")
                return "dynamic_inner_result_from_t1_d2"

            @task("dynamic_task_from_t1", inject_context=True)
            def dynamic_task_from_t1(ctx: TaskExecutionContext):
                """Dynamically added task (contains ParallelGroup)"""
                channel = ctx.get_channel()
                _append_result(channel, "dynamic_from_t1_start")

                # Create ParallelGroup within this task
                inner_group = (
                    dynamic_nested_inner_from_t1_d1 | dynamic_nested_inner_from_t1_d2
                ).set_group_name("dynamic_inner_from_t1").with_execution(
                    CoordinationBackend.REDIS,
                    backend_config={**redis_config, "key_prefix": "test_dynamic"},
                )

                ctx.next_task(inner_group)

                _append_result(channel, "dynamic_from_t1_end")
                return "dynamic_result_from_t1"

            @task("initial_task_t1", inject_context=True)
            def initial_task_t1(ctx: TaskExecutionContext):
                """Initial task (dynamically adds task with nested group)"""
                channel = ctx.get_channel()
                _append_result(channel, "initial_t1_start")

                # Dynamically add task (which contains ParallelGroup)
                ctx.next_task(dynamic_task_from_t1)

                _append_result(channel, "initial_t1_end")
                return "initial_result_t1"

            @task("initial_task_t2", inject_context=True)
            def initial_task_t2(ctx: TaskExecutionContext):
                """Initial task (dynamically adds task with nested group)"""
                channel = ctx.get_channel()
                _append_result(channel, "initial_t2_start")
                _append_result(channel, "initial_t2_end")
                return "initial_result_t2"

            # Setup Workers
            queue = DistributedTaskQueue(
                redis_client=create_redis_client(redis_config),
                key_prefix="test_dynamic"
            )
            from graflow.worker.worker import TaskWorker
            workers = [
                TaskWorker(queue, worker_id=f"test-worker-{i}", max_concurrent_tasks=3)
                for i in range(3)
            ]

            for w in workers:
                w.start()

            try:
                # Execute workflow
                init_group = (
                    initial_task_t1 | initial_task_t2
                ).set_group_name("init_group").with_execution(
                    CoordinationBackend.REDIS,
                    backend_config={**redis_config, "key_prefix": "test_dynamic"},
                )

                _last_result, exec_context = wf.execute(start_node=init_group.task_id, ret_context=True)

                # Wait for dynamic nested execution
                time.sleep(8)

                # Retrieve results from channel
                channel = exec_context.get_channel()
                results = channel.get("results") or []

                # Verify dynamic execution with nested groups
                # Initial tasks
                assert "initial_t1_start" in results
                assert "initial_t1_end" in results
                assert "initial_t2_start" in results
                assert "initial_t2_end" in results

                # Dynamic tasks (added via next_task)
                assert "dynamic_from_t1_start" in results
                assert "dynamic_from_t1_end" in results

                # Dynamic inner tasks (ParallelGroup within dynamic task)
                assert "dynamic_inner_from_t1_d1" in results
                assert "dynamic_inner_from_t1_d2" in results

                # Cleanup
                channel.delete("results")

            finally:
                for w in workers:
                    w.stop(timeout=5)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "integration"])
