"""Integration tests for nested ParallelGroup execution with Redis coordination.

Tests Phase 2 scenarios from redis_distributed_execution_redesign.md:
- Worker-local task execution via next_task()
- Nested ParallelGroup execution within Worker
- 2-3 levels of nested ParallelGroup
- Dynamic task addition that contains nested ParallelGroups
"""

import time
from typing import List

import pytest

from graflow.coordination.coordinator import CoordinationBackend
from graflow.core.context import ExecutionContext
from graflow.core.decorators import task
from graflow.core.engine import WorkflowEngine
from graflow.core.workflow import workflow
from graflow.queue.redis import RedisTaskQueue
from graflow.worker.worker import TaskWorker


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
        results = []

        with workflow("worker_next_task_local_execution") as wf:
            @task("parallel_task_1", inject_context=True)
            def parallel_task_1(ctx: ExecutionContext):
                """Task executed in ParallelGroup that adds local task via next_task()"""
                results.append("parallel_1_start")

                # Add local task via next_task() within Worker
                @task("local_task_1")
                def local_task_1():
                    results.append("local_from_1")
                    return "local_result_1"

                ctx.next_task(local_task_1)
                results.append("parallel_1_end")
                return "parallel_result_1"

            @task("parallel_task_2", inject_context=True)
            def parallel_task_2(ctx: ExecutionContext):
                """Task executed in ParallelGroup that adds local task via next_task()"""
                results.append("parallel_2_start")

                # Add local task via next_task() within Worker
                @task("local_task_2")
                def local_task_2():
                    results.append("local_from_2")
                    return "local_result_2"

                ctx.next_task(local_task_2)
                results.append("parallel_2_end")
                return "parallel_result_2"

            # Setup Worker
            context = ExecutionContext(wf.graph)
            queue = RedisTaskQueue(
                context,
                redis_client=clean_redis,
                key_prefix="test_local"
            )
            worker = TaskWorker(queue, worker_id="test-worker-1", max_concurrent_tasks=2)
            worker.start()

            try:
                # Create ParallelGroup and execute via workflow
                parallel_group = (
                    parallel_task_1 | parallel_task_2
                ).set_group_name("parallel_local").with_execution(
                    CoordinationBackend.REDIS,
                    backend_config={"redis_client": clean_redis, "key_prefix": "test_local"},
                )
                wf.graph.add_node(parallel_group, parallel_group.task_id)

                wf.execute(start_node=parallel_group.task_id)

                # Wait for worker to process tasks
                time.sleep(3)

                # Verify results
                assert "parallel_1_start" in results
                assert "parallel_1_end" in results
                assert "local_from_1" in results

                assert "parallel_2_start" in results
                assert "parallel_2_end" in results
                assert "local_from_2" in results

            finally:
                worker.stop(timeout=5)

    def test_nested_parallel_group_level_1(self, clean_redis):
        """
        Test: Nested ParallelGroup execution within Worker dispatches
              child tasks with new graph_hash.

        Scenario:
        1. Outer ParallelGroup
        2. Worker creates new ParallelGroup
        3. Inner ParallelGroup is dispatched with new graph_hash
        """
        results = []

        with workflow("nested_parallel_level_1") as wf:
            @task("inner_task_o1_i1")
            def inner_task_o1_i1():
                """Task in inner ParallelGroup"""
                results.append("inner_o1_i1")
                return "inner_result_o1_i1"

            @task("inner_task_o1_i2")
            def inner_task_o1_i2():
                """Task in inner ParallelGroup"""
                results.append("inner_o1_i2")
                return "inner_result_o1_i2"

            @task("inner_task_o2_i1")
            def inner_task_o2_i1():
                """Task in inner ParallelGroup"""
                results.append("inner_o2_i1")
                return "inner_result_o2_i1"

            @task("inner_task_o2_i2")
            def inner_task_o2_i2():
                """Task in inner ParallelGroup"""
                results.append("inner_o2_i2")
                return "inner_result_o2_i2"

            @task("outer_o1_start")
            def outer_o1_start():
                results.append("outer_o1_start")

            @task("outer_o1_end")
            def outer_o1_end():
                results.append("outer_o1_end")

            @task("outer_o2_start")
            def outer_o2_start():
                results.append("outer_o2_start")

            @task("outer_o2_end")
            def outer_o2_end():
                results.append("outer_o2_end")

            nested_group_o1 = (
                inner_task_o1_i1 | inner_task_o1_i2
            ).set_group_name("nested_o1").with_execution(
                CoordinationBackend.REDIS,
                backend_config={"redis_client": clean_redis, "key_prefix": "test_nested"},
            )

            nested_group_o2 = (
                inner_task_o2_i1 | inner_task_o2_i2
            ).set_group_name("nested_o2").with_execution(
                CoordinationBackend.REDIS,
                backend_config={"redis_client": clean_redis, "key_prefix": "test_nested"},
            )

            outer_task_o1 = (
                outer_o1_start >> nested_group_o1 >> outer_o1_end
            )

            outer_task_o2 = (
                outer_o2_start >> nested_group_o2 >> outer_o2_end
            )

            # Setup Workers (multiple for parallel processing)
            context = ExecutionContext(wf.graph)
            queue = RedisTaskQueue(
                context,
                redis_client=clean_redis,
                key_prefix="test_nested"
            )
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
                    backend_config={"redis_client": clean_redis, "key_prefix": "test_nested"},
                )

                wf.graph.add_node(outer_group, outer_group.task_id)
                wf.execute(start_node=outer_group.task_id)

                # Wait for all workers to process
                time.sleep(5)

                # Verify nested execution
                # Outer tasks
                assert "outer_o1_start" in results
                assert "outer_o1_end" in results
                assert "outer_o2_start" in results
                assert "outer_o2_end" in results

                # Inner tasks from outer_o1
                assert "inner_o1_i1" in results
                assert "inner_o1_i2" in results

                # Inner tasks from outer_o2
                assert "inner_o2_i1" in results
                assert "inner_o2_i2" in results

            finally:
                for w in workers:
                    w.stop(timeout=5)

    def test_nested_parallel_group_level_2_3(self, clean_redis):
        """
        Test: 2-3 levels of nested ParallelGroup work correctly.

        Scenario:
        Level 0 (Producer) → Level 1 (Worker) → Level 2 (Worker) → Level 3 (Worker)
        """
        results: List[str] = []

        with workflow("nested_parallel_level_2_3") as wf:
            # Level 3 tasks
            @task("level3_L1_1_L2_1_L3_1")
            def level3_L1_1_L2_1_L3_1():
                """Level 3: Deepest task"""
                results.append("L3_L1_1_L2_1_L3_1")
                return "L3_result_L1_1_L2_1_L3_1"

            @task("level3_L1_1_L2_1_L3_2")
            def level3_L1_1_L2_1_L3_2():
                """Level 3: Deepest task"""
                results.append("L3_L1_1_L2_1_L3_2")
                return "L3_result_L1_1_L2_1_L3_2"

            # Level 2 task
            @task("level2_L1_1_L2_1", inject_context=True)
            def level2_L1_1_L2_1(ctx: ExecutionContext):
                """Level 2: Creates further nested ParallelGroup"""
                results.append("L2_L1_1_L2_1_start")

                # Create Level 3 tasks
                l3_group = (
                    level3_L1_1_L2_1_L3_1 | level3_L1_1_L2_1_L3_2
                ).set_group_name("L3_group_L1_1_L2_1").with_execution(
                    CoordinationBackend.REDIS,
                    backend_config={"redis_client": clean_redis, "key_prefix": "test_deep"},
                )

                l3_group.set_execution_context(ctx)
                ctx.graph.add_node(l3_group, l3_group.task_id)
                engine = WorkflowEngine()
                engine.execute(ctx, start_task_id=l3_group.task_id)

                results.append("L2_L1_1_L2_1_end")
                return "L2_result_L1_1_L2_1"

            @task("level2_L1_1_L2_2", inject_context=True)
            def level2_L1_1_L2_2(ctx: ExecutionContext):
                """Level 2: Creates further nested ParallelGroup"""
                results.append("L2_L1_1_L2_2_start")
                results.append("L2_L1_1_L2_2_end")
                return "L2_result_L1_1_L2_2"

            # Level 1 task
            @task("level1_L1_1", inject_context=True)
            def level1_L1_1(ctx: ExecutionContext):
                """Level 1: Creates nested ParallelGroup"""
                results.append("L1_L1_1_start")

                # Create Level 2 tasks
                l2_group = (
                    level2_L1_1_L2_1 | level2_L1_1_L2_2
                ).set_group_name("L2_group_L1_1").with_execution(
                    CoordinationBackend.REDIS,
                    backend_config={"redis_client": clean_redis, "key_prefix": "test_deep"},
                )

                l2_group.set_execution_context(ctx)
                ctx.graph.add_node(l2_group, l2_group.task_id)
                engine = WorkflowEngine()
                engine.execute(ctx, start_task_id=l2_group.task_id)

                results.append("L1_L1_1_end")
                return "L1_result_L1_1"

            @task("level1_L1_2", inject_context=True)
            def level1_L1_2(ctx: ExecutionContext):
                """Level 1: Creates nested ParallelGroup"""
                results.append("L1_L1_2_start")
                results.append("L1_L1_2_end")
                return "L1_result_L1_2"

            # Setup Workers (multiple for deep parallelism)
            context = ExecutionContext(wf.graph)
            queue = RedisTaskQueue(
                context,
                redis_client=clean_redis,
                key_prefix="test_deep"
            )
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
                    backend_config={"redis_client": clean_redis, "key_prefix": "test_deep"},
                )

                wf.graph.add_node(l1_group, l1_group.task_id)
                wf.execute(start_node=l1_group.task_id)

                # Wait for deep nested execution
                time.sleep(10)

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
        results = []

        with workflow("dynamic_nested_parallel") as wf:
            @task("dynamic_nested_inner_from_t1_d1")
            def dynamic_nested_inner_from_t1_d1():
                """Inner ParallelGroup of dynamically added task"""
                results.append("dynamic_inner_from_t1_d1")
                return "dynamic_inner_result_from_t1_d1"

            @task("dynamic_nested_inner_from_t1_d2")
            def dynamic_nested_inner_from_t1_d2():
                """Inner ParallelGroup of dynamically added task"""
                results.append("dynamic_inner_from_t1_d2")
                return "dynamic_inner_result_from_t1_d2"

            @task("dynamic_task_from_t1", inject_context=True)
            def dynamic_task_from_t1(ctx: ExecutionContext):
                """Dynamically added task (contains ParallelGroup)"""
                results.append("dynamic_from_t1_start")

                # Create ParallelGroup within this task
                inner_group = (
                    dynamic_nested_inner_from_t1_d1 | dynamic_nested_inner_from_t1_d2
                ).set_group_name("dynamic_inner_from_t1").with_execution(
                    CoordinationBackend.REDIS,
                    backend_config={"redis_client": clean_redis, "key_prefix": "test_dynamic"},
                )

                inner_group.set_execution_context(ctx)
                ctx.graph.add_node(inner_group, inner_group.task_id)
                engine = WorkflowEngine()
                engine.execute(ctx, start_task_id=inner_group.task_id)

                results.append("dynamic_from_t1_end")
                return "dynamic_result_from_t1"

            @task("initial_task_t1", inject_context=True)
            def initial_task_t1(ctx: ExecutionContext):
                """Initial task (dynamically adds task with nested group)"""
                results.append("initial_t1_start")

                # Dynamically add task (which contains ParallelGroup)
                ctx.next_task(dynamic_task_from_t1)

                results.append("initial_t1_end")
                return "initial_result_t1"

            @task("initial_task_t2", inject_context=True)
            def initial_task_t2(ctx: ExecutionContext):
                """Initial task (dynamically adds task with nested group)"""
                results.append("initial_t2_start")
                results.append("initial_t2_end")
                return "initial_result_t2"

            # Setup Workers
            context = ExecutionContext(wf.graph)
            queue = RedisTaskQueue(
                context,
                redis_client=clean_redis,
                key_prefix="test_dynamic"
            )
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
                    backend_config={"redis_client": clean_redis, "key_prefix": "test_dynamic"},
                )

                wf.graph.add_node(init_group, init_group.task_id)
                wf.execute(start_node=init_group.task_id)

                # Wait for dynamic nested execution
                time.sleep(8)

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

            finally:
                for w in workers:
                    w.stop(timeout=5)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "integration"])
